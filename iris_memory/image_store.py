"""Bounded image metadata and conservative upload reservations; no image bodies."""

import hashlib
import io
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from .config import Config
from .mapping import stable_key


@dataclass(frozen=True)
class Fingerprint:
    sha256: str
    dhash: int
    color: tuple[int, int, int]
    width: int
    height: int
    animated: bool


def inspect_image(data: bytes, config: Config) -> Fingerprint:
    # Pillow is supplied by AstrBot >=4.28. Do not change its global decoder limits.
    from PIL import Image, ImageOps

    try:
        with Image.open(io.BytesIO(data)) as source:
            width, height = source.size
            if (
                min(width, height) < config.image_min_side
                or width * height > config.image_max_pixels
            ):
                raise ValueError("image dimensions")
            animated = bool(getattr(source, "is_animated", False))
            if animated and config.image_skip_stickers:
                raise ValueError("animated sticker")
            source.verify()
        with Image.open(io.BytesIO(data)) as source:
            source.seek(0)
            frame = ImageOps.exif_transpose(source).convert("RGBA")
            # Composite alpha before comparing: invisible RGB does not imply content.
            rgb = Image.new("RGBA", frame.size, "white")
            rgb.alpha_composite(frame)
            rgb = rgb.convert("RGB")
            if (
                not animated
                and config.image_skip_solid
                and all(a == b for a, b in rgb.getextrema())
            ):
                raise ValueError("solid image")
            gray = rgb.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
            pixels = gray.tobytes()
            dhash = 0
            for y in range(8):
                for x in range(8):
                    dhash = (dhash << 1) | (pixels[y * 9 + x] > pixels[y * 9 + x + 1])
            color = tuple(rgb.resize((1, 1), Image.Resampling.BOX).getpixel((0, 0)))
            width, height = rgb.size
        return Fingerprint(hashlib.sha256(data).hexdigest(), dhash, color, width, height, animated)
    except Exception:
        # Decoder diagnostics may contain file metadata; expose only a fixed reason.
        raise ValueError("image rejected") from None


class ImageStore:
    def __init__(
        self,
        directory: Path,
        config: Config,
        namespace: str,
        *,
        clock: Callable[[], float] = time.time,
    ):
        directory.mkdir(parents=True, exist_ok=True)
        self.config, self.namespace, self.clock = config, namespace, clock
        self.db = sqlite3.connect(directory / "images.sqlite3", timeout=0.05)
        try:
            self.db.execute("PRAGMA auto_vacuum=FULL")
            self.db.execute("PRAGMA journal_mode=DELETE")
            pages = (
                config.image_cache_max_mb
                * 1024**2
                // self.db.execute("PRAGMA page_size").fetchone()[0]
            )
            self.db.execute(f"PRAGMA max_page_count={pages}")
            self.db.executescript("""
                CREATE TABLE IF NOT EXISTS images (
                    scope TEXT, entry TEXT, sha TEXT, description TEXT, media TEXT,
                    dhash TEXT, red INTEGER, green INTEGER, blue INTEGER,
                    width INTEGER, height INTEGER, animated INTEGER,
                    created REAL, accessed REAL,
                    PRIMARY KEY(scope, entry, sha, description)
                );
                CREATE TABLE IF NOT EXISTS quotas (
                    day TEXT, entry TEXT, used INTEGER, PRIMARY KEY(day, entry)
                );
            """)
            with self.db:
                self._prune()
            # Shrink after lowering the configured bound, without discarding quotas.
            if self.db.execute("PRAGMA page_count").fetchone()[0] > pages:
                raise ValueError("image store exceeds limit")
        except Exception:
            self.db.close()
            raise

    def _prune(self):
        self.db.execute(
            "DELETE FROM images WHERE created <= ?",
            (self.clock() - self.config.image_cache_ttl_hours * 3600,),
        )
        self.db.execute(
            "DELETE FROM images WHERE rowid IN (SELECT rowid FROM images ORDER BY accessed DESC, rowid DESC LIMIT -1 OFFSET ?)",
            (self.config.image_cache_max_entries,),
        )

    def lookup(self, entry: str, fp: Fingerprint, description: str) -> str | None:
        with self.db:
            self._prune()
            rows = self.db.execute(
                "SELECT rowid, sha, media, dhash, red, green, blue, width, height, animated FROM images WHERE scope=? AND entry=? AND description=? ORDER BY accessed DESC",
                (self.namespace, stable_key(entry), stable_key(description)),
            ).fetchall()
            for rowid, sha, media, dhash, r, g, b, width, height, animated in rows:
                exact = sha == fp.sha256
                similar = (
                    self.config.image_perceptual_reuse
                    and not fp.animated
                    and not animated
                    and abs(width * fp.height - fp.width * height)
                    <= max(width * fp.height, fp.width * height) * 0.02
                    and max(abs(a - b) for a, b in zip(fp.color, (r, g, b))) <= 12
                    and (fp.dhash ^ int(dhash, 16)).bit_count() <= self.config.image_hash_threshold
                )
                if exact or similar:
                    self.db.execute(
                        "UPDATE images SET accessed=? WHERE rowid=?", (self.clock(), rowid)
                    )
                    return media
        return None

    def save(self, entry: str, fp: Fingerprint, description: str, media: str) -> None:
        if not isinstance(media, str) or not 1 <= len(media) <= 200:
            raise ValueError("invalid media identifier")
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO images VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    self.namespace,
                    stable_key(entry),
                    fp.sha256,
                    stable_key(description),
                    media,
                    f"{fp.dhash:016x}",
                    *fp.color,
                    fp.width,
                    fp.height,
                    int(fp.animated),
                    self.clock(),
                    self.clock(),
                ),
            )
            self._prune()

    def reserve(self, entry: str) -> bool:
        # Across scopes: changing service/token must not reset plugin-wide quotas.
        day, key = datetime.fromtimestamp(self.clock()).date().isoformat(), stable_key(entry)
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            self.db.execute("DELETE FROM quotas WHERE day < ?", (day,))
            for scope, limit in (
                ("global", self.config.image_global_daily_quota),
                (key, self.config.image_entry_daily_quota),
            ):
                row = self.db.execute(
                    "SELECT used FROM quotas WHERE day=? AND entry=?", (day, scope)
                ).fetchone()
                if (row[0] if row else 0) >= limit:
                    return False
            for scope in ("global", key):
                self.db.execute(
                    "INSERT INTO quotas VALUES (?,?,1) ON CONFLICT(day,entry) DO UPDATE SET used=used+1",
                    (day, scope),
                )
        return True

    def discard(self, entry: str, media_ids: set[str]) -> None:
        with self.db:
            for media in media_ids:
                self.db.execute(
                    "DELETE FROM images WHERE scope=? AND entry=? AND media=?",
                    (self.namespace, stable_key(entry), media),
                )

    def close(self):
        self.db.close()
