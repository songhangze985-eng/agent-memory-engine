"""Optional external L2-only vector index adapters (Qdrant REST API & embedded LanceDB Arrow/RaBitQ)."""
from __future__ import annotations

import json
import math
import threading
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Protocol


class L2VectorIndex(Protocol):
    """Auxiliary vector index restricted strictly to curated L2 Markdown notes."""

    def upsert_l2(self, node: dict, vector: list[float]) -> None: ...

    def delete(self, node_id: str) -> None: ...

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
    ) -> list[dict]: ...


def _point_uuid(node_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"memengine:vfs:{node_id}"))


class QdrantL2Index:
    """Zero-dependency HTTP/JSON Qdrant adapter for L2-only curated Markdown embeddings.

    Strictly enforces that only ``layer == "L2"`` nodes are indexed; L0 transcripts and
    L1/L3 nodes are rejected at the boundary so noisy raw logs never pollute semantic search.
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


def _sql_quote(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


class LanceL2Index:
    """Embedded Rust + Apache Arrow / LanceDB L2-only vector index with RaBitQ (IVF_RQ) ANN.

    Zero external daemon required; keeps L2 embeddings memory-mapped on disk in Lance columnar format
    and buffers writes into clean Arrow batches so large local VFS knowledge bases stay fast and cool.
    """

    engine_name = "lancedb-l2"
    ANN_MIN_ROWS = 256

    def __init__(
        self,
        uri: str | Path,
        *,
        table_name: str = "vfs_l2_vectors",
        batch_size: int = 128,
        index_type: str = "IVF_RQ",
    ):
        try:
            import lancedb
            import pyarrow as pa
        except ImportError as exc:
            raise RuntimeError("LanceL2Index requires pip install 'memengine[lance]'") from exc
        self._lancedb = lancedb
        self._pa = pa
        self.uri = str(Path(uri).expanduser().resolve())
        Path(self.uri).mkdir(parents=True, exist_ok=True)
        self.table_name = table_name
        self.batch_size = max(1, int(batch_size))
        self.index_type = index_type
        self._lock = threading.RLock()
        self._db = lancedb.connect(self.uri)
        self._table = None
        self._dim: int | None = None
        self._pending_upserts: dict[str, dict] = {}
        self._pending_deletes: set[str] = set()
        self._indexed = False
        self._open_existing()

    def _open_existing(self) -> None:
        if self.table_name in self._list_tables():
            self._table = self._db.open_table(self.table_name)
            try:
                field = self._table.schema.field("vector")
                self._dim = int(field.type.list_size)
            except Exception:
                self._dim = None

    def _list_tables(self) -> set[str]:
        res = self._db.list_tables()
        if hasattr(res, "tables"):
            return set(res.tables)
        return set(res)

    def _ensure_table(self, dim: int) -> None:
        if self._table is not None:
            if self._dim is not None and self._dim != dim:
                self._db.drop_table(self.table_name)
                self._table = None
                self._indexed = False
            else:
                self._dim = dim
                return
        pa = self._pa
        schema = pa.schema([
            pa.field("node_id", pa.string()),
            pa.field("user_id", pa.string()),
            pa.field("agent_id", pa.string()),
            pa.field("parent_path", pa.string()),
            pa.field("path", pa.string()),
            pa.field("verification_status", pa.string()),
            pa.field("salience_tag", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), int(dim))),
        ])
        self._table = self._db.create_table(self.table_name, schema=schema, mode="overwrite")
        self._dim = int(dim)

    def upsert_l2(self, node: dict, vector: list[float]) -> None:
        with self._lock:
            nid = str(node["node_id"])
            if node.get("layer") != "L2" or node.get("is_deleted"):
                self.delete(nid)
                return
            self._pending_deletes.discard(nid)
            self._pending_upserts[nid] = {
                "node_id": nid,
                "user_id": str(node["user_id"]),
                "agent_id": str(node["agent_id"]),
                "parent_path": str(node["parent_path"]),
                "path": str(node["path"]),
                "verification_status": str(node["verification_status"]),
                "salience_tag": str(node["salience_tag"]),
                "vector": [float(x) for x in vector],
            }
            if len(self._pending_upserts) >= self.batch_size:
                self.flush()

    def delete(self, node_id: str) -> None:
        with self._lock:
            nid = str(node_id)
            self._pending_upserts.pop(nid, None)
            self._pending_deletes.add(nid)
            if len(self._pending_deletes) >= self.batch_size:
                self.flush()

    def flush(self) -> None:
        with self._lock:
            if not self._pending_upserts and not self._pending_deletes:
                return
            if self._pending_upserts:
                first_vec = next(iter(self._pending_upserts.values()))["vector"]
                self._ensure_table(len(first_vec))
            if self._table is None:
                self._pending_deletes.clear()
                return
            to_delete = self._pending_deletes | set(self._pending_upserts)
            if to_delete:
                id_list = ",".join(_sql_quote(nid) for nid in sorted(to_delete))
                self._table.delete(f"node_id IN ({id_list})")
            if self._pending_upserts:
                self._table.add(list(self._pending_upserts.values()))
            self._pending_upserts.clear()
            self._pending_deletes.clear()

    def optimize_index(self) -> dict:
        with self._lock:
            self.flush()
            if self._table is None:
                return {"indexed": False, "rows": 0, "index_type": "none"}
            rows = int(self._table.count_rows())
            if rows < self.ANN_MIN_ROWS:
                return {"indexed": False, "rows": rows, "index_type": "flat-arrow"}
            partitions = max(4, min(128, int(math.sqrt(rows))))
            try:
                self._table.create_index(
                    metric="cosine",
                    num_partitions=partitions,
                    vector_column_name="vector",
                    index_type=self.index_type,
                    replace=True,
                )
                self._indexed = True
                return {"indexed": True, "rows": rows, "index_type": self.index_type, "partitions": partitions}
            except Exception:
                self._table.create_index(
                    metric="cosine",
                    num_partitions=partitions,
                    vector_column_name="vector",
                    index_type="IVF_FLAT",
                    replace=True,
                )
                self._indexed = True
                return {"indexed": True, "rows": rows, "index_type": "IVF_FLAT", "partitions": partitions}

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
        with self._lock:
            self.flush()
            if self._table is None or top_k <= 0 or (self._dim is not None and len(vector) != self._dim):
                return []
            clauses = [
                f"user_id = {_sql_quote(user_id)}",
                f"agent_id = {_sql_quote(agent_id)}",
            ]
            if verified_only:
                clauses.append("verification_status = 'verified'")
            if salience_tags:
                tag_list = ",".join(_sql_quote(t) for t in salience_tags)
                clauses.append(f"salience_tag IN ({tag_list})")
            fetch_k = top_k * 3 if (path_prefix and path_prefix != "/") else top_k
            rows = (
                self._table.search([float(x) for x in vector])
                .metric("cosine")
                .where(" AND ".join(clauses), prefilter=True)
                .select(["node_id", "path", "_distance"])
                .limit(int(fetch_k))
                .to_list()
            )
            hits = []
            for row in rows:
                path = row.get("path", "")
                if path_prefix and path_prefix != "/" and not path.startswith(path_prefix):
                    continue
                dist = float(row.get("_distance", 1.0))
                score = max(0.0, 1.0 - dist)
                if score > 0:
                    hits.append({"node_id": row["node_id"], "vector_score": score})
                if len(hits) >= top_k:
                    break
            return hits

    def close(self) -> None:
        with self._lock:
            self.flush()
