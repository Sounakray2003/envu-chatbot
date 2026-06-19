"""
Embedding service locked to OpenAI embeddings.

This codebase now supports a single embedding provider only:
  - OpenAI Embeddings API

The active model is resolved from `OPENAI_EMBEDDING_MODEL` and defaults to
`text-embedding-3-large`. Dense embeddings are generated remotely via OpenAI.
Sparse embeddings remain disabled and are represented as empty vectors.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional

import requests
from requests.exceptions import ConnectionError, Timeout

logger = logging.getLogger(__name__)

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"
DEFAULT_OPENAI_EMBEDDING_DIMENSIONS = 1024
DEFAULT_OPENAI_EMBEDDING_TIMEOUT = 120
DEFAULT_OPENAI_EMBEDDING_MAX_RETRIES = 3
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
OPENAI_BASE_URL_ENV = "OPENAI_BASE_URL"
OPENAI_EMBEDDING_MODEL_ENV = "OPENAI_EMBEDDING_MODEL"
OPENAI_EMBEDDING_DIMENSIONS_ENV = "OPENAI_EMBEDDING_DIMENSIONS"
OPENAI_EMBEDDING_TIMEOUT_ENV = "OPENAI_EMBEDDING_TIMEOUT"
OPENAI_EMBEDDING_MAX_RETRIES_ENV = "OPENAI_EMBEDDING_MAX_RETRIES"


def normalize_model_name(model_name: Optional[str]) -> str:
    """Normalize a model name like `Text Embedding 3 Large` to a stable slug."""
    return str(model_name or "").strip().lower().replace("_", "-").replace(" ", "-")


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _resolve_openai_base_url() -> str:
    configured = str(os.getenv(OPENAI_BASE_URL_ENV, DEFAULT_OPENAI_BASE_URL)).strip()
    return (configured or DEFAULT_OPENAI_BASE_URL).rstrip("/")


def _resolve_openai_api_key() -> str:
    api_key = str(os.getenv(OPENAI_API_KEY_ENV, "")).strip()
    if not api_key:
        raise RuntimeError(
            f"{OPENAI_API_KEY_ENV} is required for OpenAI embeddings."
        )
    return api_key


def _resolve_openai_embedding_model() -> str:
    configured = str(
        os.getenv(OPENAI_EMBEDDING_MODEL_ENV, DEFAULT_OPENAI_EMBEDDING_MODEL)
    ).strip()
    return configured or DEFAULT_OPENAI_EMBEDDING_MODEL


def _resolve_openai_embedding_dimensions(requested_dimensions: Any = None) -> int:
    configured = os.getenv(OPENAI_EMBEDDING_DIMENSIONS_ENV)
    if requested_dimensions not in (None, ""):
        return _safe_int(requested_dimensions, DEFAULT_OPENAI_EMBEDDING_DIMENSIONS)
    if configured not in (None, ""):
        return _safe_int(configured, DEFAULT_OPENAI_EMBEDDING_DIMENSIONS)
    return DEFAULT_OPENAI_EMBEDDING_DIMENSIONS


def _resolve_openai_embedding_timeout() -> int:
    configured = os.getenv(OPENAI_EMBEDDING_TIMEOUT_ENV)
    if configured not in (None, ""):
        return max(10, _safe_int(configured, DEFAULT_OPENAI_EMBEDDING_TIMEOUT))
    return DEFAULT_OPENAI_EMBEDDING_TIMEOUT


def _resolve_openai_embedding_max_retries() -> int:
    configured = os.getenv(OPENAI_EMBEDDING_MAX_RETRIES_ENV)
    if configured not in (None, ""):
        return max(1, _safe_int(configured, DEFAULT_OPENAI_EMBEDDING_MAX_RETRIES))
    return DEFAULT_OPENAI_EMBEDDING_MAX_RETRIES


def _format_connection_error(exc: Exception, base_url: str) -> str:
    message = str(exc)
    lowered = message.lower()
    if "failed to resolve" in lowered or "nameresolutionerror" in lowered or "getaddrinfo failed" in lowered:
        return (
            "Could not resolve the OpenAI host for embeddings. "
            f"Check DNS/network access to {base_url}. Original error: {message}"
        )
    return (
        "Could not connect to the OpenAI embeddings endpoint. "
        f"Check network access to {base_url}. Original error: {message}"
    )


class EmbeddingModelError(Exception):
    """
    Raised when the embedding API returns an error.

    Preserves the exact status code and response body so the caller can surface
    the real model/API error.
    """

    def __init__(self, status_code: int, response_body: str, model_name: str):
        self.status_code = status_code
        self.response_body = response_body
        self.model_name = model_name
        super().__init__(
            f"Embedding model '{model_name}' returned HTTP {status_code}: {response_body}"
        )


class Q0EmbeddingService:
    """Generate embeddings using the OpenAI Embeddings API only."""

    def __init__(self, embedding_details: Optional[Dict[str, Any]] = None):
        details = embedding_details or {}

        self.embedded_model_id = details.get("embedded_model_id")
        self.requested_model_name = (
            details.get("embedding_model_name") or _resolve_openai_embedding_model()
        )
        self.model_name = _resolve_openai_embedding_model()
        requested_model_name = normalize_model_name(self.requested_model_name)
        resolved_model_name = normalize_model_name(self.model_name)
        if requested_model_name and requested_model_name != resolved_model_name:
            logger.info(
                "Requested embedding model '%s' is no longer supported; using %s.",
                self.requested_model_name,
                self.model_name,
            )

        self.requested_dimensions = _resolve_openai_embedding_dimensions(
            details.get("dimensions")
        )
        self.embedding_dim = self.requested_dimensions
        self.resolved_model_name = self.model_name
        self.session = requests.Session()
        self.base_url = _resolve_openai_base_url()
        self.api_key = _resolve_openai_api_key()
        self.request_timeout_seconds = _resolve_openai_embedding_timeout()
        self.max_retries = _resolve_openai_embedding_max_retries()
        self.endpoint_mode = "openai"
        self.sparse_endpoint_url = ""

        logger.info("Loading embedding model : %s", self.model_name)
        logger.info("  Endpoint Mode         : %s", self.endpoint_mode)
        logger.info("  Endpoint              : %s/embeddings", self.base_url)
        logger.info("  Timeout (seconds)     : %s", self.request_timeout_seconds)
        logger.info("  Max Retries           : %s", self.max_retries)
        logger.info("  Sparse Model          : disabled")
        if self.embedded_model_id is not None:
            logger.info("  Embedded Model ID     : %s", self.embedded_model_id)
        logger.info("  Requested Dimensions  : %s", self.requested_dimensions)

    def _build_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _request_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate dense embeddings remotely via OpenAI."""
        if not texts:
            return []

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "input": texts,
            "encoding_format": "float",
        }
        if self.requested_dimensions:
            payload["dimensions"] = self.requested_dimensions

        last_exception: Optional[Exception] = None
        response = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.post(
                    f"{self.base_url}/embeddings",
                    headers=self._build_headers(),
                    json=payload,
                    timeout=self.request_timeout_seconds,
                )

                if response.status_code == 200:
                    break

                response_body = response.text
                retryable_status = response.status_code in {408, 409, 429, 500, 502, 503, 504}
                if retryable_status and attempt < self.max_retries:
                    sleep_seconds = min(2 ** (attempt - 1), 4)
                    logger.warning(
                        "[EMBED RETRY] model=%s status=%s attempt=%s/%s sleeping=%ss body=%s",
                        self.model_name,
                        response.status_code,
                        attempt,
                        self.max_retries,
                        sleep_seconds,
                        response_body[:300],
                    )
                    time.sleep(sleep_seconds)
                    continue

                logger.error(
                    "[EMBED ERROR] model=%s status=%s body=%s",
                    self.model_name,
                    response.status_code,
                    response_body[:1000],
                )
                raise EmbeddingModelError(
                    status_code=response.status_code,
                    response_body=response_body,
                    model_name=self.model_name,
                )
            except Timeout as exc:
                last_exception = exc
                if attempt >= self.max_retries:
                    break
                sleep_seconds = min(2 ** (attempt - 1), 4)
                logger.warning(
                    "[EMBED RETRY] model=%s timeout after %ss attempt=%s/%s sleeping=%ss",
                    self.model_name,
                    self.request_timeout_seconds,
                    attempt,
                    self.max_retries,
                    sleep_seconds,
                )
                time.sleep(sleep_seconds)
            except ConnectionError as exc:
                last_exception = exc
                formatted_message = _format_connection_error(exc, self.base_url)
                if attempt >= self.max_retries:
                    logger.error("[EMBED ERROR] %s", formatted_message)
                    raise RuntimeError(formatted_message) from exc
                sleep_seconds = min(2 ** (attempt - 1), 4)
                logger.warning(
                    "[EMBED RETRY] %s attempt=%s/%s sleeping=%ss",
                    formatted_message,
                    attempt,
                    self.max_retries,
                    sleep_seconds,
                )
                time.sleep(sleep_seconds)
            except requests.RequestException as exc:
                last_exception = exc
                if attempt >= self.max_retries:
                    raise RuntimeError(
                        f"OpenAI embeddings request failed: {exc}"
                    ) from exc
                sleep_seconds = min(2 ** (attempt - 1), 4)
                logger.warning(
                    "[EMBED RETRY] model=%s request-error=%s attempt=%s/%s sleeping=%ss",
                    self.model_name,
                    exc,
                    attempt,
                    self.max_retries,
                    sleep_seconds,
                )
                time.sleep(sleep_seconds)

        if response is None or response.status_code != 200:
            if last_exception is not None:
                if isinstance(last_exception, Timeout):
                    raise RuntimeError(
                        "OpenAI embeddings request timed out after "
                        f"{self.max_retries} attempt(s) to {self.base_url}."
                    ) from last_exception
                raise RuntimeError(
                    f"OpenAI embeddings request failed after {self.max_retries} attempt(s): {last_exception}"
                ) from last_exception
            raise RuntimeError("OpenAI embeddings request failed before a valid response was received.")

        try:
            data = response.json().get("data", [])
        except ValueError as exc:
            raise ValueError(
                f"Unexpected OpenAI embeddings response: {response.text[:300]}"
            ) from exc

        embeddings: List[List[float]] = []
        for item in data:
            embedding = item.get("embedding")
            if not isinstance(embedding, list):
                raise ValueError(f"Unexpected embedding item format: {item!r}")
            embeddings.append(embedding)

        if len(embeddings) != len(texts):
            raise ValueError(
                f"Unexpected dense embedding batch size: expected {len(texts)}, got {len(embeddings)}"
            )

        if embeddings:
            self.embedding_dim = len(embeddings[0])

        return embeddings

    def _request_embedding(self, text: str) -> List[float]:
        embeddings = self._request_embeddings([text])
        return embeddings[0]

    def _request_sparse_embeddings(self, texts: List[str]) -> List[Dict[str, List]]:
        return [{"indices": [], "values": []} for _ in texts]

    def _request_sparse_embedding(self, text: str) -> Dict[str, List]:
        return {"indices": [], "values": []}

    def embed_query(self, text: str) -> List[float]:
        try:
            return self._request_embedding(text)
        except EmbeddingModelError:
            raise
        except Exception as exc:
            logger.error("Error generating query embedding: %s", exc)
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        try:
            start_time = time.time()
            embeddings = self._request_embeddings(texts)
            elapsed_time = time.time() - start_time
            throughput = len(texts) / elapsed_time if elapsed_time > 0 else 0

            logger.info(
                "Embedded %s documents in %.2fs (%.1f docs/sec)",
                len(texts),
                elapsed_time,
                throughput,
            )
            return embeddings
        except EmbeddingModelError:
            raise
        except Exception as exc:
            logger.error("Error embedding documents: %s", exc)
            raise

    def embed_with_sparse(self, text: str) -> Dict[str, Any]:
        dense = self._request_embedding(text)
        return {
            "dense": dense,
            "sparse": {"indices": [], "values": []},
        }

    def embed_documents_with_sparse(self, texts: List[str]) -> List[Dict[str, Any]]:
        if not texts:
            return []

        start_time = time.time()
        dense_embeddings = self._request_embeddings(texts)
        sparse_embeddings = [{"indices": [], "values": []} for _ in texts]

        embeddings = [
            {"dense": dense, "sparse": sparse}
            for dense, sparse in zip(dense_embeddings, sparse_embeddings)
        ]

        elapsed_time = time.time() - start_time
        throughput = len(texts) / elapsed_time if elapsed_time > 0 else 0
        logger.info(
            "Generated dual embeddings for %s documents in %.2fs (%.1f docs/sec)",
            len(texts),
            elapsed_time,
            throughput,
        )
        return embeddings

    def get_sparse_vector(self, text: str) -> Dict[str, List]:
        return {"indices": [], "values": []}

    def get_embedding_dimension(self) -> int:
        return self.embedding_dim

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "embedded_model_id": self.embedded_model_id,
            "requested_model_name": self.requested_model_name,
            "model_name": self.resolved_model_name,
            "provider": "openai",
            "embedding_dim": self.embedding_dim,
            "requested_dimensions": self.requested_dimensions,
            "base_url": f"{self.base_url}/embeddings",
            "sparse_endpoint_url": self.sparse_endpoint_url,
            "endpoint_mode": self.endpoint_mode,
        }


def create_embedding_service(
    embedding_details: Optional[Dict[str, Any]] = None,
) -> Q0EmbeddingService:
    """Create the embedding service from request metadata."""
    return Q0EmbeddingService(embedding_details)
