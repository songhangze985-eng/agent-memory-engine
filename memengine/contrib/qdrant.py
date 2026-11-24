"""Optional external Qdrant L2-only vector index adapter (non-core contrib)."""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
import uuid


def _point_uuid(node_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"memengine:vfs:{node_id}"))


class QdrantL2Index:
    """Zero-dependency HTTP/JSON Qdrant adapter for L2-only curated Markdown embeddings.

    Moved out of core into ``memengine.contrib``; core storage uses SQLite+LanceDB
    (default local) and PostgreSQL (sole horizontal scale tier).
    """

    engine_name = "qdrant-l2"

    def __init__(
        self,
        url: str = "http://127.0.0.1:6333",
        *,
        collection: str = "memengine_l2_nodes",
        vector_dim: int = 512,
        api_key: str | None = None,
        timeout: float = 5.0,
        transport=None,
    ):
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"invalid Qdrant URL: {url!r}")
        self.base_url = url.rstrip("/")
        self.collection = collection
        self.vector_dim = int(vector_dim)
        self.api_key = api_key
        self.timeout = float(timeout)
        self._transport = transport
        self._ensured = False

    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        url = f"{self.base_url}{path}"
        if self._transport is not None:
            return self._transport(method, url, body)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["api-key"] = self.api_key
        data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw.strip() else {}

    def ensure_collection(self) -> None:
        if self._ensured:
            return
        try:
            self._request("GET", f"/collections/{self.collection}")
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise
            self._request(
                "PUT",
                f"/collections/{self.collection}",
                {"vectors": {"size": self.vector_dim, "distance": "Cosine"}},
            )
        self._ensured = True

    def upsert_l2(self, node: dict, vector: list[float]) -> None:
        if node.get("layer") != "L2" or node.get("is_deleted"):
            self.delete(node["node_id"])
            return
        self.ensure_collection()
        payload = {
            "node_id": node["node_id"],
            "user_id": node["user_id"],
            "agent_id": node["agent_id"],
            "parent_path": node["parent_path"],
            "path": node["path"],
            "title": node["title"],
            "layer": "L2",
            "verification_status": node["verification_status"],
            "salience_tag": node["salience_tag"],
            "confidence_score": float(node["confidence_score"]),
        }
        self._request(
            "PUT",
            f"/collections/{self.collection}/points?wait=true",
            {"points": [{"id": _point_uuid(node["node_id"]), "vector": [float(x) for x in vector], "payload": payload}]},
        )

    def delete(self, node_id: str) -> None:
        if not self._ensured:
            try:
                self.ensure_collection()
            except Exception:
                return
        self._request(
            "POST",
            f"/collections/{self.collection}/points/delete?wait=true",
            {"points": [_point_uuid(node_id)]},
        )

    def search_l2(
        self,
        vector: list[float],
        *,
        user_id: str,
        agent_id: str,
        path_prefix: str | None = None,
        verified_only: bool = False,
        salience_tags: tuple[str, ...] | None = None,
        top_k: int = 10,
    ) -> list[dict]:
        if top_k <= 0:
            return []
        self.ensure_collection()
        must = [
            {"key": "user_id", "match": {"value": user_id}},
            {"key": "agent_id", "match": {"value": agent_id}},
            {"key": "layer", "match": {"value": "L2"}},
        ]
        if verified_only:
            must.append({"key": "verification_status", "match": {"value": "verified"}})
        if salience_tags:
            must.append({"key": "salience_tag", "match": {"any": list(salience_tags)}})
        fetch_k = top_k * 3 if path_prefix else top_k
        resp = self._request(
            "POST",
            f"/collections/{self.collection}/points/search",
            {
                "vector": [float(x) for x in vector],
                "limit": fetch_k,
                "with_payload": True,
                "filter": {"must": must},
            },
        )
        hits = []
        for point in resp.get("result", []):
            payload = point.get("payload") or {}
            path = payload.get("path", "")
            if path_prefix and not path.startswith(path_prefix):
                continue
            score = float(point.get("score") or 0.0)
            if score > 0:
                hits.append({"node_id": payload["node_id"], "vector_score": score})
            if len(hits) >= top_k:
                break
        return hits
