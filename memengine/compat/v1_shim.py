"""Deprecated /v1 HTTP compatibility shim forwarding cleanly to the unified memory core."""
from __future__ import annotations

import warnings

V1_SUNSET_DATE = "2027-06-01"
V1_DEPRECATION_MSG = "The /v1/* HTTP routes are deprecated; migrate to unified /v2/memory/* routes."


def attach_v1_deprecation_middleware(app) -> None:
    """Mark all ``/v1/*`` responses with standard RFC 8594 Deprecation + Sunset headers."""

    @app.middleware("http")
    async def _v1_deprecation_shim(request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/v1/"):
            warnings.warn(V1_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
            response.headers["Deprecation"] = "true"
            response.headers["Sunset"] = V1_SUNSET_DATE
            response.headers["X-Deprecated-API"] = V1_DEPRECATION_MSG
        return response
