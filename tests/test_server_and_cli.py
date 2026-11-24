import io
import json

import pytest

from agentmem_core.cli import main
from agentmem_core.client import MemoryClient


def test_sqlite_cli(tmp_path, capsys):
    args = ["--db-url", str(tmp_path / "service.db")]
    for command in (["init-db"], ["retain", "The Atlas service runs at dawn."],
                    ["recall", "When does the Atlas service run?"], ["reflect", "--force"], ["health"]):
        assert main([*args, *command]) == 0
        parsed = json.loads(capsys.readouterr().out)
        assert parsed if command[0] != "recall" else parsed["ranked"]


def test_fastapi_and_urllib_client(monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from agentmem_core.server import create_app
    import agentmem_core.client as transport

    headers = {"Authorization": "Bearer service-key"}
    with TestClient(create_app(db_url=":memory:", api_key="service-key")) as server:
        assert server.get("/health").status_code == 401

        def urlopen(request, timeout):
            response = server.request(request.get_method(), request.full_url.removeprefix("http://testserver"),
                                      content=request.data, headers=request.headers)
            response.raise_for_status()
            return io.BytesIO(response.content)

        monkeypatch.setattr(transport, "urlopen", urlopen)
        client = MemoryClient("http://testserver", api_key="service-key")
        scope = {"user_id": "alice", "session_id": "chat"}
        assert client.health()["backend"] == "sqlite"
        journal = client.retain(turns=[{"text": "The Atlas project uses v3.11."},
                                       {"text": "Atlas stores all data locally."}], **scope)
        assert len(journal["turns"]) == 2
        saved = client.retain("Atlas temporary freeze lifts next week.", epistemic_scope="experience",
                              timestamp=100, expires_at=500, **scope)
        recalled = client.recall("Atlas freeze", scopes=["experience"], timestamp=200, assemble=True, **scope)
        assert "[Agent历史动作]" in recalled["dynamic_memory"]
        assert not client.recall("Atlas freeze", scopes=["experience"], timestamp=500, **scope)["ranked"]
        assert not client.recall("Atlas freeze", scopes=["experience"], timestamp=200, user_id="bob")["ranked"]
        assert client.drilldown([saved["turn_id"]], **scope)
        assert not client.drilldown([saved["turn_id"]], **{**scope, "session_id": "other"})
        assert client.history(saved["id"], **scope)[0]["expires_at"] == 500
        assert client.reflect(user_id="alice", force=True)["consolidation"]["skipped"] is False
        assert client.forget(saved["id"], hard=True, user_id="alice")["deleted"]
        response = server.post("/v1/memories/recall", headers=headers, json={"query":"Atlas", **scope})
        assert response.status_code == 200 and float(response.headers["X-Retrieval-Ms"]) >= 0
        assert len(server.get("/v1/tools/schema", headers=headers).json()) == 5
        response = server.post("/v1/tools/dispatch", headers=headers,
                               json={"name": "memory_search", "arguments": {"query": "v3.11"}, **scope})
        assert response.status_code == 200 and response.json()
