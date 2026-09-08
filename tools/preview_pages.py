"""Development-only Pages harness with temporary SQLite and Fake host.

This is never imported or started by the AstrBot plugin. It exercises the real
Pages API; only the AstrBot iframe bridge/provider/platform is replaced.
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
import uvicorn

from iris_memory.api import PagesAPI
from iris_memory.control import Control
from iris_memory.identity import Identity
from iris_memory.errors import IrisError

ROOT = Path(__file__).resolve().parents[1]


class PreviewHost:
    def provider_list(self):
        return {"chat": [], "embedding": []}

    async def providers(self, control):
        return None, None

    async def check_persona_support(self):
        pass

    async def release_personas(self, control, only_umo=None):
        pass


@asynccontextmanager
async def lifespan(app):
    with TemporaryDirectory(prefix="iris-pages-preview-") as directory:
        c = Control(directory, PreviewHost())
        await c.start()
        app.state.control = c
        app.state.api = PagesAPI(c)
        identity = Identity(
            "chat:preview",
            "session:preview",
            "preview",
            "preview-realm",
            "tester",
            "测试会话",
            "preview:FriendMessage:tester",
            "preview-msg",
            False,
        )
        await c.remember_conversation(identity)
        try:
            yield
        finally:
            await c.close()


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def index():
    text = (ROOT / "pages/iris/index.html").read_text()
    bridge = """<script>window.AstrBotPluginPage={ready:async()=>({preview:true}),apiPost:async(endpoint,body)=>{const r=await fetch('/query',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const v=await r.json();if(!r.ok)throw new Error(v.message);return v;}};</script>"""
    return HTMLResponse(
        text.replace("</head>", bridge + "</head>").replace(
            "IRIS WORKSPACE", "IRIS · 临时测试环境"
        )
    )


@app.get("/{name}")
async def asset(name: str):
    if name not in {"app.js", "style.css"}:
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(ROOT / "pages/iris" / name)


@app.post("/query")
async def query(request: Request):
    data = await request.json()
    try:
        value = await request.app.state.api.dispatch(
            data["action"], data.get("data", {}), username="preview-admin"
        )
        return {"status": "ok", "data": value}
    except IrisError as exc:
        return JSONResponse(
            {"message": str(exc), "data": exc.as_dict()}, status_code=400
        )


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8769, log_level="warning")
