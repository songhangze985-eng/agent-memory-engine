"""Use the running Compose project to verify recovery and explicit promotion."""

import argparse
import os
import subprocess
import time
from urllib.error import HTTPError, URLError
from uuid import uuid4

from agentmem_core.chrono import parse_session_clock
from agentmem_core.client import MemoryClient


def ready(client):
    deadline = time.monotonic() + 45
    while True:
        try:
            health = client.health()
            assert health["ok"] and health["backend"] == "postgresql" and health["vector"] == "pgvector", health
            return health
        except (URLError, ConnectionError, TimeoutError):
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    api_key = os.environ.get("AGENTMEM_CORE_API_KEY") or os.environ["MEMENGINE_API_KEY"]
    client = MemoryClient(args.url, api_key=api_key, timeout=4)
    owner = {"user_id": "workspace-smoke-" + uuid4().hex[:12], "agent_id": "verification", "session_id": "first"}
    shared_owner = {key: owner[key] for key in ("user_id", "agent_id")}

    def post(path, **body):
        return client._request(path, "POST", payload={**owner, **body})

    health = ready(client)
    print(f"Container database -> {health}", flush=True)
    try:
        MemoryClient(args.url, timeout=4).health()
    except HTTPError as exc:
        assert exc.code == 401
        print("Unauthenticated health -> HTTP 401", flush=True)
    else:
        raise AssertionError("health accepted a request without the service credential")

    content = "Atlas deployment checkpoint is BLUE-482."
    query = "What is the Atlas deployment checkpoint?"
    turn = post("/v1/turns/buffer", text=content)["turn_id"]
    task = post("/v1/tasks", goal="Verify the deployment checkpoint")["task_id"]
    path = f"/v1/tasks/{task}"
    item = post(f"{path}/items", content=content, item_type="provisional_claim", source_refs=[turn])["item_id"]
    checkpoint = {"phase": 2, "turn": turn, "item": item}
    post(f"{path}/checkpoint", state_dict=checkpoint)
    assert not client.recall(query, **shared_owner)["ranked"]
    print(f"Before restart -> buffered turn {turn}, active task {task}, LTM recall empty", flush=True)

    subprocess.run(["docker", "compose", "restart", "agentmem-core"], check=True)
    ready(client)
    workspace = post(f"{path}/workspace")
    assert workspace["status"] == "active" and workspace["checkpoint"] == checkpoint
    assert workspace["items"][0]["item_id"] == item
    assert workspace["items"][0]["verification_status"] == "unverified"
    answer = post("/v1/answer", query=query, task_id=task)
    assert "BLUE-482" in answer["response"] and answer["claims"][0]["evidence_ids"] == [f"turn:{turn}"]
    print(f"After restart -> checkpoint restored, answer cites turn:{turn}", flush=True)

    try:
        post(f"{path}/workspace", user_id=owner["user_id"] + "-foreign")
    except HTTPError as exc:
        assert exc.code == 422
    else:
        raise AssertionError("foreign user accessed the task")
    verify = post(f"{path}/items/{item}/verify", status="user_confirmed", verifier_ref=turn)
    assert verify["verification_status"] == "user_confirmed"
    assert not client.recall(query, **shared_owner)["ranked"]
    promoted = post(f"{path}/commit")["promoted"]
    assert set(promoted) == {item} and post(f"{path}/commit")["promoted"] == promoted
    fact = next(row for row in client.recall(query, **shared_owner)["ranked"] if row["id"] == promoted[item])
    assert fact["session_id"] == "first" and fact["source_turn_ids"] == [turn]
    assert not client.recall(query, **{**shared_owner, "user_id": owner["user_id"] + "-foreign"})["ranked"]
    print(f"Explicit commit -> {fact['id']} visible in cross-session Recall, original source preserved", flush=True)

    event = client.retain("I opened the Atlas lab on April 2, 2024.", speaker="Alice Chen",
                          mention_time="2024-04-05", **owner)
    client.retain("I calibrated the Atlas lab instruments on April 3, 2024 because our team had an audit.",
                  speaker="Alice Chen", mention_time="2024-04-06", **owner)
    recalled = client.recall("When did Alice Chen open the Atlas lab?", reference_time="2024-04-07",
                            mode="reconstructive", packing_mode="token_budget", token_budget=256, **shared_owner)
    evidence = next(row for row in recalled["ranked"] if row["id"] == event["operations"][0]["line_id"])
    assert evidence["speaker"] == "Alice Chen" and evidence["occurrence_start"] == parse_session_clock("2024-04-02")
    print(f"PostgreSQL chronicle Recall -> {evidence['id']}, occurrence 2024-04-02", flush=True)


if __name__ == "__main__":
    main()
