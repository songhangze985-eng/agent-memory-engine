"""Standard-library client for the memory HTTP service."""
import json
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


class MemoryClient:
    def __init__(self, url="http://127.0.0.1:8000", *, api_key=None, timeout=30):
        self.url = url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _request(self, path, method="GET", *, payload=None, params=None):
        suffix = "?" + urlencode({k: v for k, v in params.items() if v is not None}) if params else ""
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if payload is not None:
            headers["Content-Type"] = "application/json"
        data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode()
        request = Request(self.url + path + suffix, data, headers, method=method)
        with urlopen(request, timeout=self.timeout) as response:
            return json.load(response)

    def retain(self, text=None, **kwargs):
        return self._request("/v1/retain", "POST", payload={"text": text, **kwargs})

    def recall(self, query, **kwargs):
        return self._request("/v1/recall", "POST", payload={"query": query, **kwargs})

    def reflect(self, **kwargs):
        path = "/v1/reflect" if kwargs.get("query") is not None else "/v1/consolidate"
        return self._request(path, "POST", payload=kwargs)

    def drilldown(self, turn_ids, **kwargs):
        return self._request("/v1/drilldown", "POST", payload={"turn_ids": turn_ids, **kwargs})

    def history(self, target, **kwargs):
        return self._request("/v1/history/" + quote(target, safe=""), params=kwargs)

    def forget(self, memory_id, **kwargs):
        return self._request("/v1/memories/" + quote(memory_id, safe=""), "DELETE", params=kwargs)

    def health(self):
        return self._request("/health")
