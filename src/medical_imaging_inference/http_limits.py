"""Reject oversized multipart requests before parsing upload parts."""

import asyncio
from collections import deque

from starlette.datastructures import Headers
from starlette.responses import JSONResponse


class UploadLimits:
    def __init__(self, app, image_bytes, batch_bytes):
        self.app = app
        self.image_bytes = image_bytes
        self.batch_bytes = batch_bytes
        self.slots = asyncio.Semaphore(8)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        path = scope["path"]
        if path == "/api/v1/inference/batch":
            limit = self.batch_bytes
        elif path.startswith("/api/v1/predictions/"):
            limit = self.image_bytes
        else:
            return await self.app(scope, receive, send)
        # Allow multipart headers and framing above the payload allowance.
        limit += 256 * 1024
        headers = Headers(scope=scope)
        values = headers.getlist("content-length")
        if not values:
            return await JSONResponse({"detail": "Content-Length is required for uploads"}, 411)(
                scope, receive, send
            )
        if len(values) != 1 or not values[0].isascii() or not values[0].isdigit() or len(values[0]) > 12:
            return await JSONResponse({"detail": "Invalid Content-Length"}, 400)(scope, receive, send)
        declared = int(values[0])
        if declared > limit:
            return await JSONResponse({"detail": "Request exceeds the upload size limit"}, 413)(
                scope, receive, send
            )

        if self.slots.locked():
            return await JSONResponse({"detail": "Upload capacity is busy"}, 429)(scope, receive, send)
        async with self.slots:
            return await self._upload(scope, receive, send, declared, limit)

    async def _upload(self, scope, receive, send, declared, limit):
        # Buffer at most the declared body before passing it to the multipart
        # parser. This also rejects mismatched lengths on non-network ASGI ingress.
        pending = deque()
        size = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            data = message.get("body", b"")
            size += len(data)
            if size > declared:
                status = 413 if size > limit else 400
                return await JSONResponse({"detail": "Upload length exceeds the declared limit"}, status)(
                    scope, receive, send
                )
            if data:
                pending.append(data)
            if not message.get("more_body", False):
                break
        if size != declared:
            return await JSONResponse({"detail": "Upload length differs from Content-Length"}, 400)(
                scope, receive, send
            )

        async def receive_upload():
            if pending:
                data = pending.popleft()
                return {"type": "http.request", "body": data, "more_body": bool(pending)}
            if not declared:
                return {"type": "http.request", "body": b"", "more_body": False}
            return await receive()

        return await self.app(scope, receive_upload, send)
