import importlib
import importlib.util
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType, SimpleNamespace
from urllib.parse import parse_qs, unquote, urlsplit

import pytest

from tests.test_bridge import Event, component, prepared


class Result:
    def __init__(self, text="", llm=False):
        self.chain = [component("plain", text=text)] if text else []
        self.llm = llm
        self.use_t2i_ = None

    def message(self, text):
        self.chain.append(component("plain", text=text))
        return self

    def is_llm_result(self):
        return self.llm


class HostEvent(Event):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.result = Result("回答", llm=True)

    def is_admin(self):
        return getattr(self, "admin", True)

    def get_self_id(self):
        return "bot-self"

    def get_message_str(self):
        return "".join(c.text for c in self.messages if c.type == "plain")

    def plain_result(self, text):
        return Result(text)

    def request_llm(self, **kwargs):
        return SimpleNamespace(**kwargs)


class Context:
    def __init__(self):
        self.registered_web_apis = []
        self.sent = []
        self.config = {
            "provider_ltm_settings": {"group_message_max_cnt": 0},
            "platform_settings": {"empty_mention_waiting": False},
            "t2i": False,
        }
        self.conversation_manager = SimpleNamespace(
            get_curr_conversation_id=self.conversation_id,
            new_conversation=self.conversation_id,
            get_conversation=self.conversation,
        )

    async def conversation_id(self, *args, **kwargs):
        return "conversation-1"

    async def conversation(self, *args, **kwargs):
        return SimpleNamespace(id="conversation-1")

    def get_config(self, **kwargs):
        return self.config

    def register_web_api(self, *args):
        self.registered_web_apis.append(args)

    async def send_message(self, origin, chain):
        self.sent.append((origin, chain))
        return True


@pytest.fixture
def plugin_module(monkeypatch, tmp_path):
    logs = []

    class Star:
        def __init__(self, context):
            self.context = context

    class Filter:
        EventMessageType = SimpleNamespace(ALL="all")
        PermissionType = SimpleNamespace(ADMIN="admin")

        def __getattr__(self, name):
            def decorator(*args, **kwargs):
                def register(fn):
                    fn.hook = (name, args, kwargs)
                    return fn

                return register

            return decorator

    modules = {
        "astrbot": {},
        "astrbot.api": {
            "AstrBotConfig": dict,
            "logger": SimpleNamespace(info=logs.append, warning=logs.append),
        },
        "astrbot.api.event": {
            "AstrMessageEvent": object,
            "MessageChain": Result,
            "filter": Filter(),
        },
        "astrbot.api.provider": {"ProviderRequest": object},
        "astrbot.api.star": {
            "Context": Context,
            "Star": Star,
            "StarTools": SimpleNamespace(get_data_dir=lambda _: tmp_path),
        },
        "astrbot.api.web": {
            "request": SimpleNamespace(username="test-admin"),
            "json_response": lambda data, **kwargs: data,
            "error_response": lambda text, **kwargs: {"error": text, **kwargs},
        },
    }
    for name, attrs in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    root = Path(__file__).parents[1]
    name = "iris_test_plugin"
    spec = importlib.util.spec_from_file_location(
        name, root / "__init__.py", submodule_search_locations=[str(root)]
    )
    package = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, package)
    spec.loader.exec_module(package)
    yield importlib.import_module(name + ".main"), logs
    for key in list(sys.modules):
        if key.startswith(name + "."):
            sys.modules.pop(key, None)


@pytest.fixture
def fake_iris():
    """Actual loopback HTTP service, with state and contract-like auth/scope checks."""
    state = SimpleNamespace(
        messages={},
        requests=[],
        media=[],
        goals={},
        receipts={},
        feedback=[],
        notices=[],
        fail=None,
        prepared=prepared(),
    )

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # Never write headers / request bodies / credentials to test logs.

        def handle_request(self):
            url = urlsplit(self.path)
            path = unquote(url.path)
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            state.requests.append((self.command, path, body, parse_qs(url.query)))
            if self.headers.get("Authorization") != "Bearer iris_test_token":
                return self.reply(401, {"error": {"code": "unauthorized"}})
            if state.fail:
                return self.reply(
                    state.fail, {"error": {"code": "unavailable", "message": "iris_test_token"}}
                )
            if path == "/api/v1/status":
                return self.reply(
                    200,
                    {
                        "service": "ready",
                        "model_health": {
                            "chat": {
                                "state": "configuration_error",
                                "last_error": "iris_test_token",
                            }
                        },
                        "entries": [{"pending_count": len(v)} for v in state.messages.values()],
                        "usage": {"today": {"calls": 2, "tokens": 42, "calls_without_usage": 1}},
                    },
                )
            if path.startswith("/api/v1/entries/"):
                entry, action = path.removeprefix("/api/v1/entries/").rsplit("/", 1)
                messages = state.messages.setdefault(entry, [])
                if action == "messages":
                    old = next((m for m in messages if m["dedupe_key"] == body["dedupe_key"]), None)
                    if old is None:
                        old = {**body, "id": len(messages) + 1}
                        messages.append(old)
                    return self.reply(
                        200, {"message_ids": [old["id"]], "pending_count": len(messages)}
                    )
                if action == "prepare":
                    data = {
                        **state.prepared,
                        "recent_messages": [
                            {**m, "sender_name": m["sender"], "unlearned": True} for m in messages
                        ],
                    }
                    if state.goals:
                        data["goals"] = [
                            g
                            for g in state.goals.values()
                            if g.get("entry_id") == entry and g["state"] == "open"
                        ]
                    return self.reply(200, data)
                if action == "learn":
                    return self.reply(
                        200, {"accepted": True, "pending_count": len(messages), "paused": True}
                    )
            if path == "/api/v1/media":
                state.media.append(body)
                return self.reply(201, {"id": "media-" + str(len(state.media))})
            if path == "/api/v1/memories/search":
                return self.reply(
                    200,
                    {"memories": state.prepared["memories"], "hints": [], "recall_id": "search-1"},
                )
            if path == "/api/v1/feedback":
                state.feedback.append(body)
                return self.reply(
                    200, {"accepted": body["memory_ids"], "strengthened": body["memory_ids"]}
                )
            if path == "/api/v1/goals":
                if self.command == "POST":
                    key = body["host_key"]
                    if key not in state.receipts:
                        goal = {**body, "id": len(state.goals) + 1, "revision": 1, "state": "open"}
                        state.goals[goal["id"]] = goal
                        state.receipts[key] = {
                            "goal": goal,
                            "submitted_id": goal["id"],
                            "dedup": {"status": "created"},
                        }
                    return self.reply(201, state.receipts[key])
                query = parse_qs(url.query)
                items = [g for g in state.goals.values() if g["state"] == "open"]
                if "entry_id" in query:
                    items = [g for g in items if g.get("entry_id") in {None, query["entry_id"][0]}]
                offset = int(query.get("offset", [0])[0])
                limit = int(query.get("limit", [100])[0])
                return self.reply(
                    200,
                    {
                        "items": items[offset : offset + limit],
                        "total": len(items),
                        "limit": limit,
                        "offset": offset,
                    },
                )
            if path.startswith("/api/v1/goals/"):
                goal = state.goals[int(path.rsplit("/", 1)[1])]
                if body.get("expected_revision") != goal["revision"]:
                    return self.reply(409, {"error": {"code": "revision_conflict"}})
                goal.update(body)
                goal["revision"] += 1
                return self.reply(200, goal)
            if path == "/api/v1/notifications":
                after = int(parse_qs(url.query).get("after", [0])[0])
                notices = [n for n in state.notices if n["id"] > after]
                return self.reply(
                    200,
                    {
                        "items": notices,
                        "next_cursor": max([after] + [n["id"] for n in notices]),
                        "has_more": False,
                    },
                )
            self.reply(404, {"error": {"code": "not_found"}})

        def reply(self, status, body):
            data = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        do_GET = do_POST = do_PATCH = handle_request

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state.url = f"http://127.0.0.1:{server.server_port}"
    yield state
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


@pytest.fixture
async def plugin_factory(plugin_module, fake_iris):
    module, logs = plugin_module
    plugins = []

    async def create(**config):
        plugin = module.IrisMemoryPlugin(
            Context(),
            {
                "base_url": fake_iris.url,
                "token": "iris_test_token",
                "reminders_enabled": False,
                **config,
            },
        )
        await plugin.initialize()
        plugins.append(plugin)
        return plugin

    yield create
    for plugin in plugins:
        await plugin.terminate()
    assert "iris_test_token" not in str(logs)
