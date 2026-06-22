"""
Ingestion-only FastAPI service.

This module intentionally exposes only document ingestion management:
  - upload a file, or ingest a website URL
  - list stored uploaded/ingested sources
  - delete stored chunks by file_id

Question-answer retrieval, semantic cache, chat generation, and session history
logic have been removed from this entry point.
"""

import logging
import os
import shutil
import uuid
import asyncio
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

MAIN_MEMORY_COLLECTION = "main_memory"
USER_DEFINED_QDRANT_BACKEND_ID = 2
DEFAULT_KNOWLEDGE_BASE_ID = int(os.getenv("KNOWLEDGE_BASE_ID", "1") or "1")
DEFAULT_VECTOR_STORE_ID = int(os.getenv("VECTOR_STORE_ID", "2") or "2")
DEFAULT_OPENAI_EMBEDDING_MODEL = (
    str(os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")).strip()
    or "text-embedding-3-large"
)
DEFAULT_OPENAI_EMBEDDING_DIMENSIONS = int(
    str(os.getenv("OPENAI_EMBEDDING_DIMENSIONS", "1024")).strip() or "1024"
)
DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1024") or "1024")
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50") or "50")
DEFAULT_WEBSITE_MAX_PAGES = int(os.getenv("WEBSITE_MAX_PAGES", "500") or "500")
DEFAULT_WEBSITE_MAX_DEPTH = int(os.getenv("WEBSITE_MAX_DEPTH", "5") or "5")
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))
MAX_BATCH_FILES = int(os.getenv("MAX_BATCH_FILES", "10") or "10")
MAX_ZIP_FILES = int(os.getenv("MAX_ZIP_FILES", "100") or "100")
MAX_ZIP_UNCOMPRESSED_BYTES = int(
    os.getenv("MAX_ZIP_UNCOMPRESSED_BYTES", str(100 * 1024 * 1024))
)
MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", "300") or "300")
INGESTION_CONCURRENCY = max(1, int(os.getenv("INGESTION_CONCURRENCY", "1") or "1"))
INGESTION_QUEUE_TIMEOUT_SECONDS = float(
    os.getenv("INGESTION_QUEUE_TIMEOUT_SECONDS", "10") or "10"
)
MAX_INGESTION_SECONDS_PER_ITEM = float(
    os.getenv("MAX_INGESTION_SECONDS_PER_ITEM", "900") or "900"
)
ADMIN_OPERATION_TIMEOUT_SECONDS = float(
    os.getenv("ADMIN_OPERATION_TIMEOUT_SECONDS", "45") or "45"
)
SUPPORTED_UPLOAD_TYPES = {
    ".pdf", ".docx", ".doc", ".txt", ".md", ".markdown",
    ".json", ".csv", ".tsv", ".xlsx", ".xls", ".xlsm",
    ".html", ".htm", ".xml", ".zip",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff", ".tif",
}
_ingestion_semaphore = asyncio.Semaphore(INGESTION_CONCURRENCY)


def _normalize_optional_string(value: Any) -> Optional[str]:
    if value in (None, ""):
        return None
    normalized = str(value).strip()
    return normalized or None


def _coerce_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0

    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on", "active", "enabled"}:
        return True
    if normalized in {"0", "false", "no", "off", "inactive", "disabled"}:
        return False
    return default


def _sanitize_uploaded_filename(filename: Optional[str]) -> str:
    raw_name = Path(filename or "").name.strip()
    if not raw_name:
        return ""

    import re

    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_name).strip("._-")
    return safe_name or ""


def _validate_supported_extension(file_path: Path) -> None:
    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_UPLOAD_TYPES:
        raise ValueError(
            f"Unsupported file type '{suffix or '[no extension]'}'. "
            f"Supported types: {', '.join(sorted(SUPPORTED_UPLOAD_TYPES))}"
        )


def _get_pdf_page_count(file_path: Path) -> Optional[int]:
    if file_path.suffix.lower() != ".pdf":
        return None

    try:
        import fitz
    except ImportError:
        logger.warning("PyMuPDF not available; skipping PDF page-count preflight.")
        return None

    doc = None
    try:
        doc = fitz.open(str(file_path))
        return len(doc)
    except Exception as exc:
        raise ValueError(f"Could not read PDF page count: {exc}") from exc
    finally:
        if doc is not None:
            doc.close()


def _validate_pdf_upload(file_path: Path) -> None:
    page_count = _get_pdf_page_count(file_path)
    if page_count is None:
        return
    if page_count <= 0:
        raise ValueError("PDF has no readable pages.")
    if page_count > MAX_PDF_PAGES:
        raise ValueError(
            f"PDF has {page_count} pages. Max allowed is {MAX_PDF_PAGES} pages."
        )


def _normalize_ingestion_url(source_url: Optional[str]) -> Optional[str]:
    url = _normalize_optional_string(source_url)
    if not url:
        return None

    parsed = urlparse(url)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        raise ValueError("url must be a valid http or https URL.")
    return url


async def _save_upload_to_workspace(
    upload_file: Any,
    destination: Path,
    *,
    max_bytes: int = MAX_UPLOAD_BYTES,
) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    total_bytes = 0
    with destination.open("wb") as handle:
        while True:
            chunk = await upload_file.read(1024 * 1024)
            if not chunk:
                break
            total_bytes += len(chunk)
            if total_bytes > max_bytes:
                raise ValueError(
                    f"File is too large. Max upload size is {max_bytes // (1024 * 1024)} MB."
                )
            handle.write(chunk)
    return total_bytes


def _validate_zip_upload(zip_path: Path) -> None:
    if zip_path.suffix.lower() != ".zip":
        return

    try:
        with zipfile.ZipFile(zip_path) as archive:
            members = [member for member in archive.infolist() if not member.is_dir()]
            if len(members) > MAX_ZIP_FILES:
                raise ValueError(
                    f"ZIP has too many files. Max allowed is {MAX_ZIP_FILES} files."
                )

            total_uncompressed = 0
            for member in members:
                member_name = member.filename.replace("\\", "/")
                if member_name.startswith("/") or ".." in Path(member_name).parts:
                    raise ValueError("ZIP contains an unsafe file path.")
                suffix = Path(member_name).suffix.lower()
                if suffix not in SUPPORTED_UPLOAD_TYPES - {".zip"}:
                    raise ValueError(
                        f"ZIP contains unsupported file type '{suffix or '[no extension]'}' "
                        f"for member '{member_name}'."
                    )
                total_uncompressed += int(member.file_size or 0)

            if total_uncompressed > MAX_ZIP_UNCOMPRESSED_BYTES:
                raise ValueError(
                    "ZIP is too large after extraction. "
                    f"Max uncompressed size is {MAX_ZIP_UNCOMPRESSED_BYTES // (1024 * 1024)} MB."
                )
    except zipfile.BadZipFile as exc:
        raise ValueError("Uploaded ZIP file is invalid or corrupted.") from exc


def _validate_uploaded_file_before_ingestion(file_path: Path) -> None:
    _validate_supported_extension(file_path)
    _validate_pdf_upload(file_path)
    _validate_zip_upload(file_path)


def _build_base_ingestion_request(
    *,
    name: str,
    source_type_name: str,
    source_details: Dict[str, Any],
    is_active: bool,
) -> Dict[str, Any]:
    return {
        "knowledge_base_id": DEFAULT_KNOWLEDGE_BASE_ID,
        "name": name,
        "source_type_name": source_type_name,
        "chunking_details": {
            "chunking_type": "SEMANTIC",
            "chunkSize": DEFAULT_CHUNK_SIZE,
            "chunkOverlap": DEFAULT_CHUNK_OVERLAP,
        },
        "embedding_details": {
            "embedding_model_name": DEFAULT_OPENAI_EMBEDDING_MODEL,
            "dimensions": DEFAULT_OPENAI_EMBEDDING_DIMENSIONS,
        },
        "vector_store_details": {
            "vector_store_id": DEFAULT_VECTOR_STORE_ID,
            "collection_name": MAIN_MEMORY_COLLECTION,
            "QDRANT_COLLECTION_NAME": MAIN_MEMORY_COLLECTION,
            "qdrant_collection_name": MAIN_MEMORY_COLLECTION,
        },
        "source_details": source_details,
        "isActive": is_active,
    }


def _build_upload_request(
    *,
    file_path: Path,
    original_filename: str,
    is_active: bool,
) -> Dict[str, Any]:
    return _build_base_ingestion_request(
        name=f"File Ingestion - {Path(original_filename).stem or 'upload'}",
        source_type_name="File Upload",
        source_details={
            "file_path": str(file_path),
            "filename": original_filename,
            "source_type_name": "File Upload",
            "isActive": is_active,
        },
        is_active=is_active,
    )


def _build_website_ingestion_request(
    *,
    start_url: str,
    is_active: bool,
) -> Dict[str, Any]:
    return _build_base_ingestion_request(
        name=f"Website Ingestion - {urlparse(start_url).netloc or start_url}",
        source_type_name="Website",
        source_details={
            "start_url": start_url,
            "source_type_name": "Website",
            "max_pages": DEFAULT_WEBSITE_MAX_PAGES,
            "max_depth": DEFAULT_WEBSITE_MAX_DEPTH,
            "respect_robots_txt": _coerce_bool(
                os.getenv("WEBSITE_RESPECT_ROBOTS_TXT"),
                default=True,
            ),
            "discover_sitemaps": _coerce_bool(
                os.getenv("WEBSITE_DISCOVER_SITEMAPS"),
                default=True,
            ),
            "scope_to_start_path": _coerce_bool(
                os.getenv("WEBSITE_SCOPE_TO_START_PATH"),
                default=False,
            ),
            "isActive": is_active,
        },
        is_active=is_active,
    )


async def _resolve_ingestion_file_id(request_data: Dict[str, Any]) -> Optional[str]:
    source_details = request_data.get("source_details", {}) or {}
    source_type_name = str(request_data.get("source_type_name", "")).strip().lower()

    if "website" in source_type_name or any(
        key in source_details for key in ("start_url", "root_url", "website_url")
    ):
        from services.sources.website_source import WebsiteSource

        website_source = WebsiteSource(request_data)
        return _normalize_optional_string(website_source.generated_file_id)

    from services.sources.folder_source import FolderSource

    discovered_files = await FolderSource(request_data).discover()
    if not discovered_files:
        return None
    return _normalize_optional_string(discovered_files[0].get("file_id"))


async def _ensure_ingestion_file_id(request_data: Dict[str, Any]) -> str:
    source_details = request_data.setdefault("source_details", {})
    existing_file_id = _normalize_optional_string(source_details.get("file_id"))
    if existing_file_id:
        source_details["file_id"] = existing_file_id
        return existing_file_id

    resolved_file_id = await _resolve_ingestion_file_id(request_data)
    if not resolved_file_id:
        raise RuntimeError("file_id was not generated for this ingestion request.")

    source_details["file_id"] = resolved_file_id
    return resolved_file_id


async def _run_ingestion_request(request_data: Dict[str, Any]) -> Dict[str, Any]:
    from services.ingestion_service import IngestionService

    service = IngestionService(request_data)
    try:
        return await asyncio.wait_for(
            service.run_ingestion(),
            timeout=MAX_INGESTION_SECONDS_PER_ITEM,
        )
    except asyncio.TimeoutError as exc:
        raise TimeoutError(
            f"Ingestion exceeded {int(MAX_INGESTION_SECONDS_PER_ITEM)} seconds."
        ) from exc


def _get_vector_store(*, ensure_collection: bool = False):
    from services.vectore_store import get_vector_store

    request_data = {
        "vector_store_details": {
            "vector_store_id": DEFAULT_VECTOR_STORE_ID,
            "collection_name": MAIN_MEMORY_COLLECTION,
            "QDRANT_COLLECTION_NAME": MAIN_MEMORY_COLLECTION,
            "qdrant_collection_name": MAIN_MEMORY_COLLECTION,
        }
    }
    return get_vector_store(
        collection_name=MAIN_MEMORY_COLLECTION,
        vector_size=DEFAULT_OPENAI_EMBEDDING_DIMENSIONS,
        request_data=request_data,
        vector_type="dense",
        enable_sparse_vectors=False,
        ensure_collection=ensure_collection,
    )


def _payload_source_details(payload: Dict[str, Any]) -> Dict[str, Any]:
    source_details = payload.get("source_details")
    return source_details if isinstance(source_details, dict) else {}


def _uploaded_file_entry_from_payload(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    source_details = _payload_source_details(payload)
    file_id = _normalize_optional_string(
        payload.get("file_id") or source_details.get("file_id")
    )
    source_url = _normalize_optional_string(
        source_details.get("start_url")
        or source_details.get("root_url")
        or source_details.get("website_url")
        or source_details.get("url")
        or payload.get("source_url")
    )
    filename = _normalize_optional_string(
        source_details.get("filename")
        or payload.get("filename")
        or payload.get("file_name")
    )

    if not file_id and not source_url and not filename:
        return None

    ingestion_metadata = payload.get("ingestion_metadata")
    if not isinstance(ingestion_metadata, dict):
        ingestion_metadata = {}

    source_type_name = _normalize_optional_string(
        payload.get("source_type_name") or source_details.get("source_type_name")
    )
    if not source_type_name:
        source_type_name = "Website" if source_url else "File Upload"

    return {
        "file_id": file_id,
        "filename": filename or source_url or file_id or "unknown",
        "source_url": source_url,
        "source_mapping_id": _normalize_optional_string(
            payload.get("source_mapping_id")
            or payload.get("source_id")
            or source_details.get("source_mapping_id")
            or source_details.get("id")
        ),
        "knowledge_base_id": _normalize_optional_string(payload.get("knowledge_base_id")),
        "knowledge_base_name": _normalize_optional_string(payload.get("knowledge_base_name")),
        "source_type_name": source_type_name,
        "last_ingested_at": _normalize_optional_string(
            ingestion_metadata.get("ingestion_timestamp")
        ),
    }


def _list_uploaded_files_from_qdrant() -> List[Dict[str, Any]]:
    vector_store = _get_vector_store(ensure_collection=False)
    files_by_key: Dict[str, Dict[str, Any]] = {}
    offset: Optional[Any] = None

    while True:
        payload: Dict[str, Any] = {
            "with_payload": True,
            "with_vector": False,
            "limit": 256,
        }
        if offset is not None:
            payload["offset"] = offset

        response = vector_store._post(  # type: ignore[attr-defined]
            f"/collections/{MAIN_MEMORY_COLLECTION}/points/scroll",
            payload,
        )
        if response.status_code == 404:
            return []
        response.raise_for_status()

        result = response.json().get("result", {})
        points = result.get("points") or []
        for point in points:
            point_payload = point.get("payload") or {}
            entry = _uploaded_file_entry_from_payload(point_payload)
            if entry is None:
                continue

            dedupe_key = (
                entry.get("file_id")
                or entry.get("source_mapping_id")
                or entry.get("source_url")
                or entry.get("filename")
            )
            if not dedupe_key:
                continue

            existing = files_by_key.get(dedupe_key)
            if existing is None:
                files_by_key[dedupe_key] = {**entry, "total_chunks": 1}
                continue

            existing["total_chunks"] += 1
            if not existing.get("file_id") and entry.get("file_id"):
                existing["file_id"] = entry["file_id"]
            if entry.get("last_ingested_at") and (
                not existing.get("last_ingested_at")
                or entry["last_ingested_at"] > existing["last_ingested_at"]
            ):
                existing["last_ingested_at"] = entry["last_ingested_at"]

        offset = result.get("next_page_offset")
        if offset is None:
            break

    return sorted(
        files_by_key.values(),
        key=lambda item: (
            item.get("source_type_name") or "",
            item.get("filename") or "",
            item.get("file_id") or "",
        ),
    )


def _delete_uploaded_file_chunks_from_qdrant(file_id: str) -> Dict[str, Any]:
    normalized_file_id = _normalize_optional_string(file_id)
    if not normalized_file_id:
        raise ValueError("file_id is required.")

    vector_store = _get_vector_store(ensure_collection=False)
    matched_points = vector_store.count_by_file_id(normalized_file_id)
    if matched_points <= 0:
        raise FileNotFoundError(
            f"No chunks found for file_id '{normalized_file_id}' in collection '{MAIN_MEMORY_COLLECTION}'."
        )

    deleted_points = vector_store.delete_by_file_id(normalized_file_id)
    remaining_points = vector_store.count_by_file_id(normalized_file_id)

    return {
        "file_id": normalized_file_id,
        "collection": MAIN_MEMORY_COLLECTION,
        "matched_points": matched_points,
        "deleted_points": deleted_points,
        "remaining_points": remaining_points,
        "status": "ok",
        "operation_id": None,
    }


def create_app():
    try:
        from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import JSONResponse
        from pydantic import BaseModel, Field
    except ImportError as exc:
        raise ImportError(
            "FastAPI not installed. Run: pip install fastapi uvicorn pydantic python-multipart"
        ) from exc

    class ErrorResponse(BaseModel):
        detail: str
        error_type: Optional[str] = None

    class UploadedFileResult(BaseModel):
        file_id: Optional[str] = None
        filename: str
        source_url: Optional[str] = None
        source_mapping_id: Optional[str] = None
        knowledge_base_id: Optional[str] = None
        knowledge_base_name: Optional[str] = None
        source_type_name: str
        total_chunks: int
        last_ingested_at: Optional[str] = None

    class UploadedFilesResponse(BaseModel):
        count: int
        files: List[UploadedFileResult]

    class DeleteUploadedFileResponse(BaseModel):
        file_id: str
        collection: str
        matched_points: int
        deleted_points: int
        remaining_points: int
        status: str
        operation_id: Optional[str] = None

    class FileUploadIngestionResponse(BaseModel):
        status: str
        file_id: Optional[str] = None
        filename: Optional[str] = None
        source_url: Optional[str] = None
        knowledge_base_id: int
        collection_name: Optional[str] = None
        results: Dict[str, Any] = Field(default_factory=dict)

    class BatchFileUploadIngestionResponse(BaseModel):
        status: str
        total_items: int
        succeeded: int
        failed: int
        items: List[FileUploadIngestionResponse]

    app = FastAPI(
        title="RAG Ingestion API",
        description="Ingestion-only API for file upload, source listing, and deletion.",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(Exception)
    async def _global_exc_handler(request: Any, exc: Exception) -> JSONResponse:
        logger.error("Unhandled exception: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": str(exc), "error_type": type(exc).__name__},
        )

    @app.get("/health", tags=["System"])
    async def health_check() -> Dict[str, str]:
        return {
            "status": "ok",
            "service": "ingestion-api",
            "collection": MAIN_MEMORY_COLLECTION,
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }

    @app.post(
        "/ingest/file-upload",
        response_model=BatchFileUploadIngestionResponse,
        responses={500: {"model": ErrorResponse}},
        tags=["Ingestion"],
        summary="Upload one or more files, or provide one website URL and run ingestion",
    )
    async def ingest_uploaded_file(
        file: Optional[UploadFile] = File(None),
        files: Optional[List[UploadFile]] = File(None),
        url: Optional[str] = Form(None),
        is_active: bool = Form(True),
    ) -> BatchFileUploadIngestionResponse:
        upload_root = Path(__file__).resolve().parent / "uploaded_files"
        upload_dir = upload_root / str(uuid.uuid4())
        normalized_input_url = _normalize_ingestion_url(url)
        active_files: List[UploadFile] = []

        if file is not None:
            active_files.append(file)
        if files:
            active_files.extend(files)

        if not active_files and normalized_input_url is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Provide file/files or url.",
            )
        if active_files and normalized_input_url is not None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Provide only files or url, not both.",
            )
        if len(active_files) > MAX_BATCH_FILES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Too many files in one request. Max allowed is {MAX_BATCH_FILES}.",
            )

        try:
            try:
                await asyncio.wait_for(
                    _ingestion_semaphore.acquire(),
                    timeout=INGESTION_QUEUE_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError as exc:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Ingestion is busy. Try again shortly.",
                ) from exc

            items: List[FileUploadIngestionResponse] = []
            try:
                if normalized_input_url is not None:
                    request_data = _build_website_ingestion_request(
                        start_url=normalized_input_url,
                        is_active=is_active,
                    )
                    file_id = await _ensure_ingestion_file_id(request_data)
                    results = await _run_ingestion_request(request_data)
                    items.append(
                        FileUploadIngestionResponse(
                            status=str(results.get("status", "FAILED")),
                            file_id=file_id,
                            filename=None,
                            source_url=normalized_input_url,
                            knowledge_base_id=int(
                                request_data.get("knowledge_base_id", 1)
                            ),
                            collection_name=(
                                _normalize_optional_string(results.get("collection_name"))
                                or MAIN_MEMORY_COLLECTION
                            ),
                            results=results,
                        )
                    )
                else:
                    for active_file in active_files:
                        safe_filename = _sanitize_uploaded_filename(active_file.filename)
                        if not safe_filename:
                            raise HTTPException(
                                status_code=status.HTTP_400_BAD_REQUEST,
                                detail="A filename is required for every uploaded file.",
                            )

                        saved_file_path = upload_dir / f"{uuid.uuid4()}_{safe_filename}"
                        try:
                            saved_bytes = await _save_upload_to_workspace(
                                active_file,
                                saved_file_path,
                            )
                            _validate_uploaded_file_before_ingestion(saved_file_path)
                            request_data = _build_upload_request(
                                file_path=saved_file_path,
                                original_filename=safe_filename,
                                is_active=is_active,
                            )
                            file_id = await _ensure_ingestion_file_id(request_data)
                            results = await _run_ingestion_request(request_data)
                            results.setdefault("uploaded_size_bytes", saved_bytes)
                            items.append(
                                FileUploadIngestionResponse(
                                    status=str(results.get("status", "FAILED")),
                                    file_id=file_id,
                                    filename=safe_filename,
                                    source_url=None,
                                    knowledge_base_id=int(
                                        request_data.get("knowledge_base_id", 1)
                                    ),
                                    collection_name=(
                                        _normalize_optional_string(
                                            results.get("collection_name")
                                        )
                                        or MAIN_MEMORY_COLLECTION
                                    ),
                                    results=results,
                                )
                            )
                        except Exception as item_exc:
                            logger.error(
                                "Upload ingestion failed for %s: %s",
                                safe_filename,
                                item_exc,
                                exc_info=True,
                            )
                            items.append(
                                FileUploadIngestionResponse(
                                    status="FAILED",
                                    file_id=None,
                                    filename=safe_filename,
                                    source_url=None,
                                    knowledge_base_id=DEFAULT_KNOWLEDGE_BASE_ID,
                                    collection_name=MAIN_MEMORY_COLLECTION,
                                    results={"errors": [str(item_exc)]},
                                )
                            )

                failed = sum(1 for item in items if item.status.upper() in {"FAILED", "ERROR"})
                succeeded = len(items) - failed
                return BatchFileUploadIngestionResponse(
                    status=(
                        "success"
                        if failed == 0
                        else "partial_success"
                        if succeeded > 0
                        else "FAILED"
                    ),
                    total_items=len(items),
                    succeeded=succeeded,
                    failed=failed,
                    items=items,
                )
            finally:
                _ingestion_semaphore.release()
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("Upload ingestion failed: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Upload ingestion failed: {exc}",
            ) from exc
        finally:
            for active_file in active_files:
                try:
                    await active_file.close()
                except Exception:
                    pass
            shutil.rmtree(upload_dir, ignore_errors=True)

    @app.get(
        "/ingest/file-uploads",
        response_model=UploadedFilesResponse,
        responses={500: {"model": ErrorResponse}},
        tags=["Ingestion"],
        summary="List uploaded files and website ingestions",
    )
    async def files_uploaded() -> UploadedFilesResponse:
        try:
            files = await asyncio.wait_for(
                asyncio.to_thread(_list_uploaded_files_from_qdrant),
                timeout=ADMIN_OPERATION_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError as exc:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail=(
                    "Listing uploaded files timed out while scanning Qdrant. "
                    "Try again later or reduce main_memory collection size."
                ),
            ) from exc
        except Exception as exc:
            logger.error("Failed to list uploaded files: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to list uploaded files: {exc}",
            ) from exc

        return UploadedFilesResponse(count=len(files), files=files)

    @app.delete(
        "/ingest/file-uploads/{file_id}",
        response_model=DeleteUploadedFileResponse,
        responses={500: {"model": ErrorResponse}},
        tags=["Ingestion"],
        summary="Delete stored chunks for a file_id",
    )
    async def delete_uploaded_file(file_id: str) -> DeleteUploadedFileResponse:
        try:
            deleted = await asyncio.wait_for(
                asyncio.to_thread(_delete_uploaded_file_chunks_from_qdrant, file_id),
                timeout=ADMIN_OPERATION_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError as exc:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Delete timed out while waiting for Qdrant.",
            ) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        except Exception as exc:
            logger.error(
                "Failed to delete uploaded file chunks for file_id=%s: %s",
                file_id,
                exc,
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to delete uploaded file chunks: {exc}",
            ) from exc

        return DeleteUploadedFileResponse(**deleted)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8094")))
