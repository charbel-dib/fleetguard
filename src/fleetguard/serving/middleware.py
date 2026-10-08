"""Bound streamed bodies, reject ambiguous JSON, and tag each response."""

import json
import logging
import uuid

from starlette.responses import JSONResponse

logger = logging.getLogger("fleetguard.api")


def error_response(code, message, *, status, request_id, details=None):
    return JSONResponse(
        {
            "error": {
                "code": code,
                "message": message,
                "details": details or [],
                "request_id": request_id,
            }
        },
        status_code=status,
    )


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key.")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("Nonfinite JSON number.")


class RequestEnvelope:
    def __init__(self, app, *, max_bytes):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        request_id = uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id

        response_started = False

        async def tagged_send(message):
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
                message["headers"] = [
                    (k, v) for k, v in message.get("headers", []) if k.lower() != b"x-request-id"
                ] + [(b"x-request-id", request_id.encode())]
            await send(message)

        async def reject(code, message, status):
            await error_response(code, message, status=status, request_id=request_id)(
                scope, receive, tagged_send
            )

        headers = scope.get("headers", [])
        lengths = [v for k, v in headers if k.lower() == b"content-length"]
        if lengths:
            if len(lengths) != 1 or not lengths[0].isdigit() or len(lengths[0]) > 20:
                return await reject("invalid_request", "Invalid Content-Length header.", 400)
            if int(lengths[0]) > self.max_bytes:
                return await reject("body_too_large", "Request body limit exceeded.", 413)
        inference = (
            scope["path"] in {"/v1/predict", "/v1/predict-batch"} and scope["method"] == "POST"
        )
        if inference:
            content_types = [v for k, v in headers if k.lower() == b"content-type"]
            encodings = [v for k, v in headers if k.lower() == b"content-encoding"]
            if (
                len(content_types) != 1
                or content_types[0].split(b";", 1)[0].strip().lower() != b"application/json"
            ):
                return await reject("unsupported_media_type", "Use application/json.", 415)
            if encodings and (len(encodings) != 1 or encodings[0].lower() != b"identity"):
                return await reject(
                    "unsupported_media_type", "Compressed request bodies are not supported.", 415
                )
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            if len(body) + len(chunk) > self.max_bytes:
                return await reject("body_too_large", "Request body limit exceeded.", 413)
            body.extend(chunk)
            if not message.get("more_body", False):
                break
        if inference and body:
            try:
                json.loads(
                    bytes(body), object_pairs_hook=_unique_pairs, parse_constant=_reject_constant
                )
            except (ValueError, UnicodeError, RecursionError):
                return await reject(
                    "invalid_json", "Use valid JSON with unique keys and finite numbers.", 400
                )
        delivered = False

        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        try:
            await self.app(scope, replay, tagged_send)
        except Exception as exc:
            logger.error("Request failure request=%s type=%s", request_id, type(exc).__name__)
            if response_started:
                raise
            await reject("internal_error", "Prediction failed.", 500)
