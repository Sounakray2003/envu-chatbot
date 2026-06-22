"""
Admin API for live document ingestion and deletion.

This service wraps the existing IngestionService so admins can add, list,
check, and delete files without restarting the retrieval API or recreating the
Qdrant collection.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

MAIN_MEMORY_COLLECTION = "main_memory"
DEFAULT_KNOWLEDGE_BASE_ID = int(os.getenv("KNOWLEDGE_BASE_ID", "1") or "1")
DEFAULT_VECTOR_STORE_ID = int(os.getenv("VECTOR_STORE_ID", "2") or "2")
DEFAULT_OPENAI_EMBEDDING_MODEL = (
    str(os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")).strip()
    or "text-embedding-3-large"
)
DEFAULT_OPENAI_EMBEDDING_DIMENSIONS = int(
    str(os.getenv("OPENAI_EMBEDDING_DIMENSIONS", "1024")).strip() or "1024"
)
DEFAULT_WEBSITE_MAX_PAGES = int(os.getenv("WEBSITE_MAX_PAGES", "500") or "500")
DEFAULT_WEBSITE_MAX_DEPTH = int(os.getenv("WEBSITE_MAX_DEPTH", "5") or "5")
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))
MAX_ZIP_FILES = int(os.getenv("MAX_ZIP_FILES", "100") or "100")
MAX_ZIP_UNCOMPRESSED_BYTES = int(
    os.getenv("MAX_ZIP_UNCOMPRESSED_BYTES", str(100 * 1024 * 1024))
)
MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", "300") or "300")
SUPPORTED_UPLOAD_TYPES = {
    ".pdf", ".docx", ".doc", ".txt", ".md", ".markdown",
    ".json", ".csv", ".tsv", ".xlsx", ".xls", ".xlsm",
    ".html", ".htm", ".xml", ".zip",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff", ".tif",
}

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "/app/uploads"))
REGISTRY_PATH = Path(os.getenv("FILE_REGISTRY_PATH", "/app/file_registry/files.json"))

_registry_lock = Lock()


class DeleteResponse(BaseModel):
    file_id: str
    status: str
    deleted_points: int


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _normalize_optional_string(value: Any) -> Optional[str]:
    if value in (None, ""):
        return None
    normalized = str(value).strip()
    return normalized or None


def _safe_filename(filename: str) -> str:
    name = Path(filename or "upload.pdf").name
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")
    return safe or "upload.pdf"


def _coerce_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on", "active", "enabled"}:
        return True
    if normalized in {"0", "false", "no", "off", "inactive", "disabled"}:
        return False
    return default


def _normalize_url(source_url: Optional[str]) -> Optional[str]:
    url = str(source_url or "").strip()
    if not url:
        return None
    parsed = urlparse(url)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        raise HTTPException(status_code=400, detail="url must be a valid http or https URL")
    return url


def _validate_supported_extension(file_path: Path) -> None:
    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_UPLOAD_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{suffix or '[no extension]'}'. "
                f"Supported types: {', '.join(sorted(SUPPORTED_UPLOAD_TYPES))}"
            ),
        )


def _validate_pdf_upload(file_path: Path) -> None:
    if file_path.suffix.lower() != ".pdf":
        return
    try:
        import fitz
    except ImportError:
        logger.warning("PyMuPDF not available; skipping PDF page-count preflight.")
        return

    doc = None
    try:
        doc = fitz.open(str(file_path))
        page_count = len(doc)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read PDF page count: {exc}") from exc
    finally:
        if doc is not None:
            doc.close()

    if page_count <= 0:
        raise HTTPException(status_code=400, detail="PDF has no readable pages.")
    if page_count > MAX_PDF_PAGES:
        raise HTTPException(
            status_code=400,
            detail=f"PDF has {page_count} pages. Max allowed is {MAX_PDF_PAGES} pages.",
        )


def _validate_zip_upload(zip_path: Path) -> None:
    if zip_path.suffix.lower() != ".zip":
        return
    try:
        with zipfile.ZipFile(zip_path) as archive:
            members = [member for member in archive.infolist() if not member.is_dir()]
            if len(members) > MAX_ZIP_FILES:
                raise HTTPException(
                    status_code=400,
                    detail=f"ZIP has too many files. Max allowed is {MAX_ZIP_FILES} files.",
                )

            total_uncompressed = 0
            supported_members = 0
            unsupported_members = 0
            for member in members:
                member_name = member.filename.replace("\\", "/")
                if member_name.startswith("/") or ".." in Path(member_name).parts:
                    raise HTTPException(status_code=400, detail="ZIP contains an unsafe file path.")
                suffix = Path(member_name).suffix.lower()
                if suffix not in SUPPORTED_UPLOAD_TYPES - {".zip"}:
                    unsupported_members += 1
                    logger.info(
                        "ZIP preflight skipping unsupported member: %s (%s)",
                        member_name,
                        suffix or "[no extension]",
                    )
                    continue
                supported_members += 1
                total_uncompressed += int(member.file_size or 0)

            if supported_members <= 0:
                raise HTTPException(
                    status_code=400,
                    detail="ZIP contains no supported files to ingest.",
                )
            if unsupported_members > 0:
                logger.info(
                    "ZIP preflight will skip %d unsupported member(s).",
                    unsupported_members,
                )

            if total_uncompressed > MAX_ZIP_UNCOMPRESSED_BYTES:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "ZIP is too large after extraction. "
                        f"Max uncompressed size is {MAX_ZIP_UNCOMPRESSED_BYTES // (1024 * 1024)} MB."
                    ),
                )
    except zipfile.BadZipFile as exc:
        raise HTTPException(status_code=400, detail="Uploaded ZIP file is invalid or corrupted.") from exc


def _validate_uploaded_file(file_path: Path) -> None:
    _validate_supported_extension(file_path)
    _validate_pdf_upload(file_path)
    _validate_zip_upload(file_path)


async def _save_upload_stream(upload_file: UploadFile, destination: Path) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    total_bytes = 0
    with destination.open("wb") as handle:
        while True:
            chunk = await upload_file.read(1024 * 1024)
            if not chunk:
                break
            total_bytes += len(chunk)
            if total_bytes > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"File is too large. Max upload size is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
                )
            handle.write(chunk)
    if total_bytes <= 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    return total_bytes


def _load_registry() -> Dict[str, Dict[str, Any]]:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not REGISTRY_PATH.exists():
        return {}
    try:
        with REGISTRY_PATH.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError:
        logger.warning("Registry file is invalid JSON; starting with empty registry")
        return {}
    return data if isinstance(data, dict) else {}


def _save_registry(data: Dict[str, Dict[str, Any]]) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = REGISTRY_PATH.with_suffix(".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
    os.replace(temp_path, REGISTRY_PATH)


def _update_file_record(file_id: str, **updates: Any) -> Dict[str, Any]:
    with _registry_lock:
        registry = _load_registry()
        current = registry.get(file_id, {})
        current.update(updates)
        current["file_id"] = file_id
        current["updated_at"] = _utc_now()
        registry[file_id] = current
        _save_registry(registry)
        return dict(current)


def _get_file_record(file_id: str) -> Optional[Dict[str, Any]]:
    with _registry_lock:
        return _load_registry().get(file_id)


def _build_request_data(
    *,
    file_id: str,
    filename: Optional[str],
    knowledge_base_id: int,
    member_id: Optional[str],
    org_id: Optional[str],
    source_mapping_id: Optional[str],
    saved_path: Optional[Path] = None,
    url: Optional[str] = None,
) -> Dict[str, Any]:
    if url:
        source_type_name = "Website"
        source_details: Dict[str, Any] = {
            "start_url": url,
            "source_type_name": "Website",
            "file_id": file_id,
            "max_pages": DEFAULT_WEBSITE_MAX_PAGES,
            "max_depth": DEFAULT_WEBSITE_MAX_DEPTH,
            "respect_robots_txt": _coerce_bool(os.getenv("WEBSITE_RESPECT_ROBOTS_TXT"), True),
            "discover_sitemaps": _coerce_bool(os.getenv("WEBSITE_DISCOVER_SITEMAPS"), True),
            "scope_to_start_path": _coerce_bool(os.getenv("WEBSITE_SCOPE_TO_START_PATH"), False),
            "isActive": False,
        }
        request_name = f"Admin Website - {urlparse(url).netloc or url}"
    else:
        if saved_path is None or filename is None:
            raise ValueError("saved_path and filename are required for file upload ingestion")
        source_type_name = "File Upload"
        source_details = {
            "file_path": str(saved_path),
            "filename": filename,
            "source_type_name": "File Upload",
            "file_id": file_id,
            "isActive": False,
        }
        request_name = f"Admin Upload - {filename}"

    if source_mapping_id:
        source_details["source_mapping_id"] = source_mapping_id

    request_data: Dict[str, Any] = {
        "knowledge_base_id": knowledge_base_id,
        "name": request_name,
        "source_type_name": source_type_name,
        "isActive": False,
        "chunking_details": {
            "chunking_type": "SEMANTIC",
            "chunkSize": int(os.getenv("CHUNK_SIZE", "1024") or "1024"),
            "chunkOverlap": int(os.getenv("CHUNK_OVERLAP", "50") or "50"),
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
    }
    if member_id:
        request_data["member_id"] = member_id
    if org_id:
        request_data["org_id"] = org_id
    return request_data


def _get_admin_vector_store():
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
        ensure_collection=False,
    )


def _payload_source_details(payload: Dict[str, Any]) -> Dict[str, Any]:
    source_details = payload.get("source_details")
    return source_details if isinstance(source_details, dict) else {}


def _file_record_from_qdrant_payload(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
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

    is_active = _coerce_bool(payload.get("isActive"), default=True)
    source_type_name = _normalize_optional_string(
        payload.get("source_type_name") or source_details.get("source_type_name")
    )
    if not source_type_name:
        source_type_name = "Website" if source_url else "File Upload"

    return {
        "file_id": file_id,
        "filename": filename or source_url or file_id or "unknown",
        "source_url": source_url,
        "status": "active" if is_active else "inactive",
        "collection_name": MAIN_MEMORY_COLLECTION,
        "knowledge_base_id": payload.get("knowledge_base_id"),
        "knowledge_base_name": _normalize_optional_string(payload.get("knowledge_base_name")),
        "member_id": _normalize_optional_string(payload.get("member_id")),
        "org_id": payload.get("org_id"),
        "source_mapping_id": _normalize_optional_string(
            payload.get("source_mapping_id")
            or payload.get("source_id")
            or source_details.get("source_mapping_id")
            or source_details.get("id")
        ),
        "source_type_name": source_type_name,
        "uploaded_at": _normalize_optional_string(
            ingestion_metadata.get("ingestion_timestamp")
        ),
        "completed_at": _normalize_optional_string(
            ingestion_metadata.get("ingestion_timestamp")
        ),
    }


def _merge_qdrant_file_record(
    files_by_key: Dict[str, Dict[str, Any]],
    entry: Dict[str, Any],
) -> None:
    dedupe_key = (
        entry.get("file_id")
        or entry.get("source_mapping_id")
        or entry.get("source_url")
        or entry.get("filename")
    )
    if not dedupe_key:
        return

    existing = files_by_key.get(str(dedupe_key))
    if existing is None:
        files_by_key[str(dedupe_key)] = {
            **entry,
            "chunk_count": 1,
            "vector_count": 1,
            "results": {
                "status": "success",
                "total_files_processed": 1,
                "total_chunks_created": 1,
                "total_vectors_stored": 1,
                "errors": [],
                "warnings": [],
            },
        }
        return

    existing["chunk_count"] = int(existing.get("chunk_count") or 0) + 1
    existing["vector_count"] = int(existing.get("vector_count") or 0) + 1
    if isinstance(existing.get("results"), dict):
        existing["results"]["total_chunks_created"] = existing["chunk_count"]
        existing["results"]["total_vectors_stored"] = existing["vector_count"]

    if not existing.get("file_id") and entry.get("file_id"):
        existing["file_id"] = entry["file_id"]
    if entry.get("completed_at") and (
        not existing.get("completed_at")
        or str(entry["completed_at"]) > str(existing["completed_at"])
    ):
        existing["completed_at"] = entry["completed_at"]
        existing["uploaded_at"] = entry.get("uploaded_at") or existing.get("uploaded_at")


def _list_file_records_from_qdrant() -> List[Dict[str, Any]]:
    vector_store = _get_admin_vector_store()
    files_by_key: Dict[str, Dict[str, Any]] = {}
    offset: Optional[Any] = None

    while True:
        scroll_payload: Dict[str, Any] = {
            "with_payload": True,
            "with_vector": False,
            "limit": 256,
        }
        if offset is not None:
            scroll_payload["offset"] = offset

        response = vector_store._post(  # type: ignore[attr-defined]
            f"/collections/{MAIN_MEMORY_COLLECTION}/points/scroll",
            scroll_payload,
        )
        if response.status_code == 404:
            return []
        response.raise_for_status()

        result = response.json().get("result", {})
        for point in result.get("points") or []:
            point_payload = point.get("payload") or {}
            entry = _file_record_from_qdrant_payload(point_payload)
            if entry is None or entry.get("status") != "active":
                continue
            _merge_qdrant_file_record(files_by_key, entry)

        offset = result.get("next_page_offset")
        if offset is None:
            break

    return sorted(
        files_by_key.values(),
        key=lambda item: str(item.get("completed_at") or item.get("uploaded_at") or ""),
        reverse=True,
    )


def _get_file_record_from_qdrant(file_id: str) -> Optional[Dict[str, Any]]:
    normalized_file_id = str(file_id or "").strip()
    if not normalized_file_id:
        return None

    vector_store = _get_admin_vector_store()
    scroll_payload: Dict[str, Any] = {
        "with_payload": True,
        "with_vector": False,
        "limit": 256,
        "filter": {
            "must": [
                {
                    "key": "file_id",
                    "match": {"value": normalized_file_id},
                }
            ]
        },
    }
    files_by_key: Dict[str, Dict[str, Any]] = {}
    offset: Optional[Any] = None

    while True:
        if offset is not None:
            scroll_payload["offset"] = offset
        elif "offset" in scroll_payload:
            scroll_payload.pop("offset", None)

        response = vector_store._post(  # type: ignore[attr-defined]
            f"/collections/{MAIN_MEMORY_COLLECTION}/points/scroll",
            scroll_payload,
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()

        result = response.json().get("result", {})
        points = result.get("points") or []
        for point in points:
            point_payload = point.get("payload") or {}
            entry = _file_record_from_qdrant_payload(point_payload)
            if entry is not None:
                _merge_qdrant_file_record(files_by_key, entry)

        offset = result.get("next_page_offset")
        if offset is None:
            break

    return next(iter(files_by_key.values()), None)


async def _run_ingestion_job(
    *,
    file_id: str,
    filename: Optional[str],
    knowledge_base_id: int,
    member_id: Optional[str],
    org_id: Optional[str],
    source_mapping_id: Optional[str],
    saved_path: Optional[Path] = None,
    url: Optional[str] = None,
) -> None:
    from services.ingestion_service import IngestionService

    started_at = _utc_now()
    _update_file_record(file_id, status="processing", started_at=started_at)

    try:
        request_data = _build_request_data(
            file_id=file_id,
            filename=filename,
            knowledge_base_id=knowledge_base_id,
            member_id=member_id,
            org_id=org_id,
            source_mapping_id=source_mapping_id,
            saved_path=saved_path,
            url=url,
        )
        service = IngestionService(request_data)
        results = await service.run_ingestion()

        if results.get("status") == "FAILED" or results.get("total_vectors_stored", 0) <= 0:
            _update_file_record(
                file_id,
                status="failed",
                results=results,
                error="Ingestion failed or stored zero vectors",
            )
            return

        vector_store = _get_admin_vector_store()
        activated_points = vector_store.set_file_active(file_id, True)

        _update_file_record(
            file_id,
            status="active",
            collection_name=MAIN_MEMORY_COLLECTION,
            chunk_count=results.get("total_chunks_created", 0),
            vector_count=results.get("total_vectors_stored", 0),
            activated_points=activated_points,
            results=results,
            completed_at=_utc_now(),
        )
    except Exception as exc:
        logger.error("Ingestion job failed for file_id=%s: %s", file_id, exc, exc_info=True)
        _update_file_record(
            file_id,
            status="failed",
            error=str(exc),
            completed_at=_utc_now(),
        )


def _run_ingestion_job_background(**kwargs: Any) -> None:
    asyncio.run(_run_ingestion_job(**kwargs))


def _summarize_file_record(record: Dict[str, Any]) -> Dict[str, Any]:
    summary = dict(record)
    results = summary.get("results")
    if isinstance(results, dict):
        summary["results"] = {
            "status": results.get("status"),
            "total_files_processed": results.get("total_files_processed"),
            "total_chunks_created": results.get("total_chunks_created"),
            "total_vectors_stored": results.get("total_vectors_stored"),
            "errors": results.get("errors", [])[:5] if isinstance(results.get("errors"), list) else results.get("errors"),
            "warnings": results.get("warnings", [])[:5] if isinstance(results.get("warnings"), list) else results.get("warnings"),
        }
    return summary


app = FastAPI(title="RAG Admin Ingestion API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "collection": MAIN_MEMORY_COLLECTION}


@app.post("/ingest/file-upload", include_in_schema=False)
@app.post("/admin/files/upload")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(default=None),
    url: Optional[str] = Form(default=None),
    file_id: Optional[str] = Form(default=None),
    knowledge_base_id: int = Form(default=DEFAULT_KNOWLEDGE_BASE_ID),
    member_id: Optional[str] = Form(default=None),
    org_id: Optional[str] = Form(default=None),
    source_mapping_id: Optional[str] = Form(default=None),
) -> Dict[str, Any]:
    normalized_url = _normalize_url(url)
    if file is None and normalized_url is None:
        raise HTTPException(status_code=400, detail="Provide file or url.")
    if file is not None and normalized_url is not None:
        raise HTTPException(status_code=400, detail="Provide only file or url, not both.")

    resolved_file_id = str(file_id or uuid.uuid4()).strip()
    if not resolved_file_id:
        raise HTTPException(status_code=400, detail="file_id cannot be empty")

    filename: Optional[str] = None
    saved_path: Optional[Path] = None
    uploaded_size_bytes: Optional[int] = None
    source_url: Optional[str] = None

    if file is not None:
        filename = _safe_filename(file.filename or "upload")
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        saved_path = UPLOAD_DIR / f"{resolved_file_id}_{filename}"
        try:
            uploaded_size_bytes = await _save_upload_stream(file, saved_path)
            _validate_uploaded_file(saved_path)
        finally:
            await file.close()
    else:
        source_url = normalized_url

    now = _utc_now()
    _update_file_record(
        resolved_file_id,
        filename=filename or source_url or resolved_file_id,
        source_url=source_url,
        storage_path=str(saved_path) if saved_path else None,
        status="queued",
        collection_name=MAIN_MEMORY_COLLECTION,
        knowledge_base_id=knowledge_base_id,
        member_id=member_id,
        org_id=org_id,
        source_mapping_id=source_mapping_id,
        uploaded_size_bytes=uploaded_size_bytes,
        created_at=now,
        uploaded_at=now,
    )

    background_tasks.add_task(
        _run_ingestion_job_background,
        file_id=resolved_file_id,
        saved_path=saved_path,
        filename=filename,
        knowledge_base_id=knowledge_base_id,
        member_id=member_id,
        org_id=org_id,
        source_mapping_id=source_mapping_id,
        url=source_url,
    )

    return {
        "file_id": resolved_file_id,
        "filename": filename,
        "source_url": source_url,
        "status": "queued",
        "collection_name": MAIN_MEMORY_COLLECTION,
    }


@app.get("/ingest/file-uploads", include_in_schema=False)
@app.get("/admin/files")
def list_files() -> Dict[str, Any]:
    files_by_key: Dict[str, Dict[str, Any]] = {}
    for item in _list_file_records_from_qdrant():
        dedupe_key = (
            item.get("file_id")
            or item.get("source_mapping_id")
            or item.get("source_url")
            or item.get("filename")
        )
        if dedupe_key:
            files_by_key[str(dedupe_key)] = item

    with _registry_lock:
        registry = _load_registry()
    for item in registry.values():
        status = str(item.get("status") or "").lower()
        if status in {"deleted", "active"}:
            continue
        dedupe_key = (
            item.get("file_id")
            or item.get("source_mapping_id")
            or item.get("source_url")
            or item.get("filename")
        )
        if dedupe_key:
            files_by_key[str(dedupe_key)] = _summarize_file_record(item)

    files = sorted(
        files_by_key.values(),
        key=lambda item: str(
            item.get("created_at")
            or item.get("completed_at")
            or item.get("uploaded_at")
            or item.get("updated_at")
            or ""
        ),
        reverse=True,
    )
    return {"files": files, "total": len(files)}


@app.get("/admin/files/{file_id}")
def get_file_status(file_id: str) -> Dict[str, Any]:
    record = _get_file_record(file_id)
    if record and str(record.get("status") or "").lower() in {"queued", "processing", "failed", "delete_failed"}:
        return record

    qdrant_record = _get_file_record_from_qdrant(file_id)
    if qdrant_record:
        if record:
            merged = {**record, **qdrant_record}
            return _summarize_file_record(merged)
        return qdrant_record

    if not record:
        raise HTTPException(status_code=404, detail="file_id not found")
    return record


@app.delete("/ingest/file-uploads/{file_id}", response_model=DeleteResponse, include_in_schema=False)
@app.delete("/admin/files/{file_id}", response_model=DeleteResponse)
async def delete_file(file_id: str) -> DeleteResponse:
    record = _get_file_record(file_id)
    qdrant_record = _get_file_record_from_qdrant(file_id)
    if not record and not qdrant_record:
        raise HTTPException(status_code=404, detail="file_id not found")

    _update_file_record(file_id, status="deleting")

    try:
        vector_store = _get_admin_vector_store()
        vector_store.set_file_active(file_id, False)
        deleted_points = vector_store.delete_by_file_id(file_id)
    except Exception as exc:
        _update_file_record(file_id, status="delete_failed", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    _update_file_record(
        file_id,
        status="deleted",
        deleted_points=deleted_points,
        deleted_at=_utc_now(),
    )
    return DeleteResponse(
        file_id=file_id,
        status="deleted",
        deleted_points=deleted_points,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("ADMIN_API_PORT", "8094")))
