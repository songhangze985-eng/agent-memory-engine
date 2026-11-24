"""Optional external adapters moved out of core (core uses SQLite+LanceDB and PostgreSQL only)."""
from agentmem_core.contrib.qdrant import QdrantL2Index
from agentmem_core.contrib.s3_transcripts import S3TranscriptStore

__all__ = ["QdrantL2Index", "S3TranscriptStore"]
