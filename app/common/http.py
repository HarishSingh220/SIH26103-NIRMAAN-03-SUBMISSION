"""Shared HTTP plumbing used by both services (anomaly detection and
cost/time overrun regression): a single error-response envelope, request-id
logging, capped CSV-upload reading, and the CPU-bound-work-to-threadpool
helper. Registered once on the combined app in `app/main.py` so both
services' routers get identical error handling instead of two independent
(and inevitably drifting) copies.
"""
from __future__ import annotations

import io
import logging
import os
import time
import uuid
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logger = logging.getLogger("paimana.api")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    logger.addHandler(_handler)
    logger.propagate = False
logger.setLevel(LOG_LEVEL)

MAX_CSV_BYTES = int(os.getenv("MAX_CSV_UPLOAD_BYTES", str(25 * 1024 * 1024)))  # 25 MB default


class ErrorResponse(BaseModel):
    detail: Any
    request_id: str | None = None


def register_error_handling(app: FastAPI) -> None:
    """Attach the shared exception handlers + request-id logging middleware
    to the combined app. Call this once, from app/main.py."""

    @app.exception_handler(StarletteHTTPException)
    async def _http_exception_handler(request: Request, exc: StarletteHTTPException):
        request_id = getattr(request.state, "request_id", None)
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(detail=exc.detail, request_id=request_id).model_dump(),
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_exception_handler(request: Request, exc: RequestValidationError):
        request_id = getattr(request.state, "request_id", None)
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                detail=jsonable_encoder(exc.errors()), request_id=request_id
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def _unhandled_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled exception in %s %s", request.method, request.url.path)
        request_id = getattr(request.state, "request_id", None)
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                detail="Internal server error.", request_id=request_id
            ).model_dump(),
        )

    @app.middleware("http")
    async def _request_logging_middleware(request: Request, call_next):
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        request.state.request_id = request_id
        start = time.perf_counter()
        response = await call_next(request)
        latency_ms = (time.perf_counter() - start) * 1000
        response.headers["x-request-id"] = request_id
        logger.info(
            "request_id=%s method=%s path=%s status=%d latency_ms=%.1f",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            latency_ms,
        )
        return response


async def read_upload_capped(file: UploadFile, max_bytes: int = MAX_CSV_BYTES) -> bytes:
    """Read an upload into memory, aborting as soon as it exceeds ``max_bytes``.

    Reads in fixed-size chunks and bails out the moment the running total
    crosses the limit, so memory usage is bounded by ``max_bytes`` (plus one
    chunk) regardless of how large the client claims/attempts to send --
    unlike a plain ``await file.read()`` (no size argument), which buffers
    the entire body before any size check can run.
    """
    chunk_size = 1024 * 1024  # 1 MB
    chunks = []
    total = 0
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"CSV upload too large (exceeds {max_bytes} bytes); "
                    "stopped reading before buffering the full file."
                ),
            )
        chunks.append(chunk)
    return b"".join(chunks)


def parse_csv_bytes(content: bytes) -> pd.DataFrame:
    try:
        return pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Unable to read CSV: {exc}") from exc


async def run_offloaded(fn, *args, **kwargs):
    """Run a CPU-bound scoring/prediction call in the threadpool, translating
    the exceptions it's expected to raise into the right HTTP status codes.
    The single place that maps exceptions to HTTP responses -- endpoints
    never need their own try/except around a scoring call.
    """
    try:
        return await run_in_threadpool(fn, *args, **kwargs)
    except HTTPException:
        raise
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception:
        logger.exception("Unhandled error while scoring a request")
        raise HTTPException(
            status_code=500,
            detail="Internal error while scoring the request.",
        )
