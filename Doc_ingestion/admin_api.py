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
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

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
MAX_UPLOAD_SIZE_BYTES = int(os.getenv("MAX_UPLOAD_SIZE_MB", "25") or "25") * 1024 * 1024

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "/app/uploads"))
REGISTRY_PATH = Path(os.getenv("FILE_REGISTRY_PATH", "/app/file_registry/files.json"))

_registry_lock = Lock()


class DeleteResponse(BaseModel):
    file_id: str
    status: str
    deleted_points: int


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_filename(filename: str) -> str:
    name = Path(filename or "upload.pdf").name
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")
    return safe or "upload.pdf"


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

def _list_files_from_collection() -> Dict[str, Dict[str, Any]]:
    """Best-effort scan of Qdrant payloads to discover files present in the collection."""
    from services.vectore_store import get_vector_store

    request_data = {
        "vector_store_details": {
            "vector_store_id": DEFAULT_VECTOR_STORE_ID,
            "collection_name": MAIN_MEMORY_COLLECTION,
            "QDRANT_COLLECTION_NAME": MAIN_MEMORY_COLLECTION,
            "qdrant_collection_name": MAIN_MEMORY_COLLECTION,
        }
    }
    vector_store = get_vector_store(
        collection_name=MAIN_MEMORY_COLLECTION,
        vector_size=DEFAULT_OPENAI_EMBEDDING_DIMENSIONS,
        request_data=request_data,
        vector_type="dense",
        enable_sparse_vectors=False,
        ensure_collection=False,
    )

    backend = getattr(vector_store, "_backend", vector_store)
    if not hasattr(backend, "_scroll_collection_batch"):
        return {}

    files: Dict[str, Dict[str, Any]] = {}
    offset = None
    while True:
        points, next_offset = backend._scroll_collection_batch(
            MAIN_MEMORY_COLLECTION,
            limit=256,
            offset=offset,
        )
        if not points:
            break

        for point in points:
            payload = point.get("payload") or {}
            source_details = payload.get("source_details") or {}
            metadata = payload.get("metadata") or {}
            file_id = str(
                payload.get("file_id")
                or source_details.get("file_id")
                or metadata.get("file_id")
                or ""
            ).strip()
            if not file_id:
                continue

            existing = files.setdefault(
                file_id,
                {
                    "file_id": file_id,
                    "filename": source_details.get("filename")
                    or metadata.get("filename")
                    or metadata.get("source_file")
                    or payload.get("source_name")
                    or payload.get("storage_path")
                    or source_details.get("storage_path")
                    or file_id,
                    "collection_name": MAIN_MEMORY_COLLECTION,
                    "chunk_count": 0,
                },
            )
            existing["chunk_count"] += 1

        if next_offset is None:
            break
        offset = next_offset

    return files

def _get_file_from_collection(file_id: str) -> Optional[Dict[str, Any]]:
    file_map = _list_files_from_collection()
    return file_map.get(str(file_id or "").strip())


def _build_url_request_data(
    *,
    file_id: str,
    source_url: str,
    knowledge_base_id: int,
    member_id: Optional[str],
    org_id: Optional[str],
    source_mapping_id: Optional[str],
) -> Dict[str, Any]:
    source_details: Dict[str, Any] = {
        "start_url": source_url,
        "file_id": file_id,
        "isActive": False,
    }
    if source_mapping_id:
        source_details["source_mapping_id"] = source_mapping_id

    request_data: Dict[str, Any] = {
        "knowledge_base_id": knowledge_base_id,
        "name": f"Admin URL Ingestion - {source_url}",
        "source_type_name": "Website",
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


def _run_url_ingestion_job(
    *,
    file_id: str,
    source_url: str,
    knowledge_base_id: int,
    member_id: Optional[str],
    org_id: Optional[str],
    source_mapping_id: Optional[str],
) -> None:
    from services.ingestion_service import IngestionService

    _update_file_record(file_id, status="processing")

    try:
        request_data = _build_url_request_data(
            file_id=file_id,
            source_url=source_url,
            knowledge_base_id=DEFAULT_KNOWLEDGE_BASE_ID,
            member_id=None,
            org_id=None,
            source_mapping_id=None,
        )
        service = IngestionService(request_data)
        results = asyncio.run(service.run_ingestion())

        if results.get("status") == "FAILED" or results.get("total_vectors_stored", 0) <= 0:
            _update_file_record(
                file_id,
                status="failed",
                results=results,
                error="Ingestion failed or stored zero vectors",
                completed_at=_utc_now(),
            )
            return

        _update_file_record(
            file_id,
            status="active",
            collection_name=MAIN_MEMORY_COLLECTION,
            chunk_count=results.get("total_chunks_created", 0),
            vector_count=results.get("total_vectors_stored", 0),
            results=results,
            completed_at=_utc_now(),
        )
    except Exception as exc:
        logger.error("URL ingestion job failed for file_id=%s: %s", file_id, exc, exc_info=True)
        _update_file_record(
            file_id,
            status="failed",
            error=str(exc),
            completed_at=_utc_now(),
        )

def _build_request_data(
    *,
    file_id: str,
    saved_path: Path,
    filename: str,
    knowledge_base_id: int,
    member_id: Optional[str],
    org_id: Optional[str],
    source_mapping_id: Optional[str],
) -> Dict[str, Any]:
    source_details: Dict[str, Any] = {
        "file_path": str(saved_path),
        "filename": filename,
        "file_id": file_id,
        "isActive": False,
    }
    if source_mapping_id:
        source_details["source_mapping_id"] = source_mapping_id

    request_data: Dict[str, Any] = {
        "knowledge_base_id": knowledge_base_id,
        "name": f"Admin Upload - {filename}",
        "source_type_name": "File Upload",
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


async def _run_ingestion_job(
    *,
    file_id: str,
    saved_path: Path,
    filename: str,
    knowledge_base_id: int,
    member_id: Optional[str],
    org_id: Optional[str],
    source_mapping_id: Optional[str],
) -> None:
    from services.ingestion_service import IngestionService

    started_at = _utc_now()
    _update_file_record(file_id, status="processing", started_at=started_at)

    try:
        request_data = _build_request_data(
            file_id=file_id,
            saved_path=saved_path,
            filename=filename,
            knowledge_base_id=DEFAULT_KNOWLEDGE_BASE_ID,
            member_id=None,
            org_id=None,
            source_mapping_id=None,
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


@app.post("/admin/files/upload")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(default=None),
    url: Optional[str] = Form(default=None),
) -> Dict[str, Any]:
    resolved_file_id = str(uuid.uuid4()).strip()
    if not resolved_file_id:
        raise HTTPException(status_code=400, detail="file_id cannot be empty")

    source_url = str(url or "").strip()
    if file is None and not source_url:
        raise HTTPException(status_code=400, detail="Provide either a file or a url")
    if file is not None and source_url:
        raise HTTPException(status_code=400, detail="Provide either a file or a url, not both")

    now = _utc_now()

    if source_url:
        if not source_url.lower().startswith(("http://", "https://")):
            raise HTTPException(status_code=400, detail="url must start with http:// or https://")

        _update_file_record(
            resolved_file_id,
            filename=source_url,
            status="queued",
            collection_name=MAIN_MEMORY_COLLECTION,
            knowledge_base_id=DEFAULT_KNOWLEDGE_BASE_ID,
            member_id=None,
            org_id=None,
            source_mapping_id=None,
            source_url=source_url,
            created_at=now,
            uploaded_at=now,
        )

        background_tasks.add_task(
            _run_url_ingestion_job,
            file_id=resolved_file_id,
            source_url=source_url,
            knowledge_base_id=DEFAULT_KNOWLEDGE_BASE_ID,
            member_id=None,
            org_id=None,
            source_mapping_id=None,
        )

        return {
            "file_id": resolved_file_id,
            "filename": source_url,
            "status": "queued",
            "collection_name": MAIN_MEMORY_COLLECTION,
        }

    filename = _safe_filename(file.filename or "upload.pdf")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    saved_path = UPLOAD_DIR / f"{resolved_file_id}_{filename}"

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(content) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(status_code=400, detail=f"Uploaded file size cannot exceed {int(MAX_UPLOAD_SIZE_BYTES / (1024 * 1024))} MB")
    saved_path.write_bytes(content)

    _update_file_record(
        resolved_file_id,
        filename=filename,
        storage_path=str(saved_path),
        status="queued",
        collection_name=MAIN_MEMORY_COLLECTION,
        knowledge_base_id=DEFAULT_KNOWLEDGE_BASE_ID,
        member_id=None,
        org_id=None,
        source_mapping_id=None,
        created_at=now,
        uploaded_at=now,
    )

    background_tasks.add_task(
        _run_ingestion_job,
        file_id=resolved_file_id,
        saved_path=saved_path,
        filename=filename,
        knowledge_base_id=DEFAULT_KNOWLEDGE_BASE_ID,
        member_id=None,
        org_id=None,
        source_mapping_id=None,
    )

    return {
        "file_id": resolved_file_id,
        "filename": filename,
        "status": "queued",
        "collection_name": MAIN_MEMORY_COLLECTION,
    }


@app.get("/admin/files")
def list_files() -> Dict[str, Any]:
    with _registry_lock:
        registry = _load_registry()

    discovered = _list_files_from_collection()
    files = []
    for file_id, item in discovered.items():
        registry_item = registry.get(file_id, {})
        files.append({
            "file_id": file_id,
            "filename": registry_item.get("filename") or item.get("filename"),
            "collection_name": MAIN_MEMORY_COLLECTION,
            "chunk_count": item.get("chunk_count", 0),
            "status": registry_item.get("status") or "active",
            "created_at": registry_item.get("created_at"),
            "completed_at": registry_item.get("completed_at"),
            "knowledge_base_id": registry_item.get("knowledge_base_id", DEFAULT_KNOWLEDGE_BASE_ID),
        })

    files.sort(
        key=lambda item: str(item.get("created_at") or item.get("completed_at") or ""),
        reverse=True,
    )
    return { "total_files": len(files),"files": files}


@app.get("/admin/files/{file_id}")
def get_file_status(file_id: str) -> Dict[str, Any]:
    collection_item = _get_file_from_collection(file_id)
    if not collection_item:
        raise HTTPException(status_code=404, detail="file_id not found")

    registry_item = _get_file_record(file_id) or {}
    return {
        "file_id": file_id,
        "filename": registry_item.get("filename") or collection_item.get("filename"),
        "collection_name": MAIN_MEMORY_COLLECTION,
        "chunk_count": collection_item.get("chunk_count", 0),
        "status": registry_item.get("status") or "active",
        "created_at": registry_item.get("created_at"),
        "completed_at": registry_item.get("completed_at"),
        "knowledge_base_id": registry_item.get("knowledge_base_id", DEFAULT_KNOWLEDGE_BASE_ID),
        "storage_path": registry_item.get("storage_path"),
        "source_mapping_id": registry_item.get("source_mapping_id"),
    }


@app.delete("/admin/files/{file_id}", response_model=DeleteResponse)
async def delete_file(file_id: str) -> DeleteResponse:
    record = _get_file_record(file_id)
    if not record:
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
