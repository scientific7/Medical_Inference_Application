"""Dashboard's authenticated backend transport."""

import os
from urllib.parse import quote

import httpx


def model_path(model: dict) -> str:
    return "/".join(quote(model[key], safe="") for key in ("model_id", "version"))


class APIError(Exception):
    pass


class API:
    def __init__(self, token: str):
        self.base = os.getenv("INFERENCE_API_URL", "http://127.0.0.1:8000").rstrip("/")
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}

    def request(self, path: str, method: str = "GET", **kwargs):
        try:
            response = httpx.request(method, self.base + path, headers=self.headers, timeout=120, **kwargs)
        except httpx.RequestError as exc:
            raise APIError("Backend connection failed.") from exc
        if not response.is_success:
            request_id = response.headers.get("x-request-id")
            detail = ""
            try:
                message = response.json().get("detail")
                if isinstance(message, str):
                    detail = f" {message}"
            except (ValueError, AttributeError):
                pass
            suffix = f" Request {request_id}." if request_id else ""
            raise APIError(f"Backend returned HTTP {response.status_code}.{detail}{suffix}")
        return response.json()
