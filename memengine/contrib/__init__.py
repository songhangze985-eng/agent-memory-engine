"""Optional external adapters moved out of core (core uses SQLite+LanceDB and PostgreSQL only)."""
from memengine.contrib.qdrant import QdrantL2Index
from memengine.contrib.s3_transcripts import S3TranscriptStore

__all__ = ["QdrantL2Index", "S3TranscriptStore"]
