#!/usr/bin/env python3
"""
RAG System - Data Ingestion Pipeline (Redesigned)
Entry point that accepts REQUEST_JSON from environment variable.

Supported inputs:
  - Full JSON request object
  - Plain website URL (for website ingestion)
  - Plain local file / zip filename (resolved from the workspace)

Usage:
    docker run -e REQUEST_JSON='{"knowledge_base_id": 13, ...}' rag-ingestion
    docker run -e REQUEST_JSON='https://www.in.envu.com/' rag-ingestion
    docker run -e REQUEST_JSON='sample_zip.zip' rag-ingestion
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from urllib.parse import urlparse

from dotenv import load_dotenv
load_dotenv()

# Setup logging
def setup_logging() -> logging.Logger:
    """Configure logging"""
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"rag_ingestion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


logger = setup_logging()

INGESTION_COLLECTION_NAME = "main_memory"
DEFAULT_KNOWLEDGE_BASE_ID = 1
DEFAULT_VECTOR_STORE_ID = 1
DEFAULT_WEBSITE_MAX_PAGES = 500
DEFAULT_WEBSITE_MAX_DEPTH = 5
DEFAULT_OPENAI_EMBEDDING_MODEL = (
    str(os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")).strip()
    or "text-embedding-3-large"
)
DEFAULT_OPENAI_EMBEDDING_DIMENSIONS = int(
    str(os.getenv("OPENAI_EMBEDDING_DIMENSIONS", "1024")).strip() or "1024"
)


def print_banner():
    """Print startup banner"""
    print("\n" + "=" * 70)
    print("  RAG SYSTEM — DATA INGESTION PIPELINE (REDESIGNED)")
    print("=" * 70)
    print(f"  Started: {datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')}")
    print("=" * 70 + "\n")


def parse_request_json() -> Dict[str, Any]:
    """Parse REQUEST_JSON/INGEST_SOURCE into a full ingestion request."""
    request_input, source_env_name = _get_request_input()
    normalized_request_json = _normalize_request_json_string(request_input)

    for candidate, was_normalized in (
        (request_input, False),
        (normalized_request_json, True),
    ):
        if was_normalized and candidate == request_input:
            continue
        try:
            parsed_payload = json.loads(candidate)
            request_data = _coerce_request_payload(parsed_payload)
            print(json.dumps(request_data, ensure_ascii=True))
            if was_normalized:
                logger.warning(
                    "%s used backtick-delimited JSON-ish content. Normalized it before parsing.",
                    source_env_name,
                )
            logger.info("Parsed %s successfully", source_env_name)
            return request_data
        except json.JSONDecodeError:
            continue

    request_data = _build_request_from_shorthand(request_input)
    print(json.dumps(request_data, ensure_ascii=True))
    logger.info(
        "Built ingestion request from %s shortcut input",
        source_env_name,
    )
    return request_data


def _get_request_input() -> Tuple[str, str]:
    """Return the raw request input and the env var it came from."""
    request_json_str = os.getenv("REQUEST_JSON")
    if request_json_str not in (None, ""):
        return request_json_str, "REQUEST_JSON"

    ingest_source = os.getenv("INGEST_SOURCE")
    if ingest_source not in (None, ""):
        return ingest_source, "INGEST_SOURCE"

    logger.error("Neither REQUEST_JSON nor INGEST_SOURCE environment variable is set")
    raise ValueError("REQUEST_JSON or INGEST_SOURCE environment variable is required")


def _coerce_request_payload(payload: Any) -> Dict[str, Any]:
    """Accept a JSON object/list directly or expand a shortcut input."""
    if isinstance(payload, dict):
        return payload

    if isinstance(payload, list):
        return _build_request_from_list_payload(payload)

    if isinstance(payload, str):
        return _build_request_from_shorthand(payload)

    raise ValueError(
        "Input must be a JSON object, a JSON list of sources, a website URL, "
        "or a local file/zip path."
    )


def _build_request_from_shorthand(raw_value: str) -> Dict[str, Any]:
    """Expand plain shortcut input into a full ingestion request."""
    shorthand = str(raw_value or "").strip()
    if not shorthand:
        raise ValueError("Shortcut ingestion input cannot be empty.")

    multi_source_values = _parse_multi_source_shorthand_values(shorthand)
    if multi_source_values is not None:
        return _build_multi_source_request_from_values(multi_source_values)

    source_type_name, source_details, source_label = _build_source_config_from_shorthand(
        shorthand
    )
    if source_type_name == "Website":
        return _build_base_request(
            name=f"Website Ingestion - {source_label}",
            source_type_name=source_type_name,
            source_details=source_details,
        )

    return _build_base_request(
        name=f"File Ingestion - {source_label}",
        source_type_name=source_type_name,
        source_details=source_details,
    )


def _is_probable_website_url(value: str) -> bool:
    """Return True when a value looks like an http/https website URL."""
    parsed = urlparse(str(value or "").strip())
    return parsed.scheme.lower() in {"http", "https"} and bool(parsed.netloc)


def _build_request_from_list_payload(payload: List[Any]) -> Dict[str, Any]:
    """Expand a JSON list into a multi-source ingestion request."""
    if not payload:
        raise ValueError("Source list input cannot be empty.")

    if all(isinstance(item, dict) for item in payload):
        return _build_base_request(
            name=f"Multi Source Ingestion - {len(payload)} sources",
            source_type_name="multi-source",
            source_details=payload,
        )

    if all(isinstance(item, str) for item in payload):
        return _build_multi_source_request_from_values([str(item) for item in payload])

    raise ValueError(
        "Source list input must contain either only strings or only source objects."
    )


def _parse_multi_source_shorthand_values(raw_value: str) -> Optional[List[str]]:
    """Parse bracketed multi-source shorthand into a list of raw source values."""
    text = str(raw_value or "").strip()
    if not (text.startswith("[") and text.endswith("]")):
        return None

    inner = text[1:-1].strip()
    if not inner:
        raise ValueError("Multi-source shorthand cannot be empty.")

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, list):
        if not parsed:
            raise ValueError("Multi-source shorthand cannot be empty.")
        if not all(isinstance(item, str) for item in parsed):
            raise ValueError(
                "Bracketed multi-source shorthand must contain only string values."
            )
        return [str(item).strip() for item in parsed if str(item).strip()]

    parts = [part.strip() for part in inner.split(",")]
    values: List[str] = []
    for part in parts:
        if not part:
            continue
        normalized = part.strip().strip('"').strip("'")
        if normalized:
            values.append(normalized)

    if not values:
        raise ValueError("Multi-source shorthand cannot be empty.")
    return values


def _build_multi_source_request_from_values(values: List[str]) -> Dict[str, Any]:
    """Build a multi-source request from URL/path shortcut values."""
    if not values:
        raise ValueError("Multi-source shortcut input cannot be empty.")

    source_entries: List[Dict[str, Any]] = []
    for raw_value in values:
        source_type_name, source_details, _ = _build_source_config_from_shorthand(
            raw_value
        )
        entry = dict(source_details)
        entry["source_type_name"] = source_type_name
        source_entries.append(entry)

    return _build_base_request(
        name=f"Multi Source Ingestion - {len(source_entries)} sources",
        source_type_name="multi-source",
        source_details=source_entries,
    )


def _build_source_config_from_shorthand(
    raw_value: str,
) -> Tuple[str, Dict[str, Any], str]:
    """Resolve one shorthand source value into source config details."""
    shorthand = str(raw_value or "").strip()
    if not shorthand:
        raise ValueError("Source shortcut value cannot be empty.")

    if _is_probable_website_url(shorthand):
        parsed = urlparse(shorthand)
        return "Website", _build_website_source_details(shorthand), parsed.netloc or "website"

    source_details, source_label = _build_local_source_details(shorthand)
    return "File Upload", source_details, source_label


def _build_website_source_details(start_url: str) -> Dict[str, Any]:
    """Construct normalized website source details from a start URL."""
    return {
        "start_url": start_url,
        "max_pages": _get_int_env("WEBSITE_MAX_PAGES", DEFAULT_WEBSITE_MAX_PAGES),
        "max_depth": _get_int_env("WEBSITE_MAX_DEPTH", DEFAULT_WEBSITE_MAX_DEPTH),
        "respect_robots_txt": _get_bool_env("WEBSITE_RESPECT_ROBOTS_TXT", True),
        "discover_sitemaps": _get_bool_env("WEBSITE_DISCOVER_SITEMAPS", True),
        "scope_to_start_path": _get_bool_env("WEBSITE_SCOPE_TO_START_PATH", False),
    }


def _build_local_source_details(raw_path: str) -> Tuple[Dict[str, Any], str]:
    """Construct normalized file/folder source details from one local path."""
    resolved_path = _resolve_local_source_path(raw_path)
    if resolved_path.is_dir():
        return {"folder_path": str(resolved_path)}, resolved_path.name

    return {
        "file_path": str(resolved_path),
        "filename": resolved_path.name,
    }, resolved_path.stem


def _build_website_request(start_url: str) -> Dict[str, Any]:
    """Construct a default website ingestion request from a single URL."""
    parsed = urlparse(start_url)
    site_label = parsed.netloc or "website"
    return _build_base_request(
        name=f"Website Ingestion - {site_label}",
        source_type_name="Website",
        source_details=_build_website_source_details(start_url),
    )


def _build_local_source_request(raw_path: str) -> Dict[str, Any]:
    """Construct a default file/folder ingestion request from one local path."""
    source_details, source_label = _build_local_source_details(raw_path)
    if "folder_path" in source_details:
        return _build_base_request(
            name=f"Folder Ingestion - {source_label}",
            source_type_name="File Upload",
            source_details=source_details,
        )

    return _build_base_request(
        name=f"File Ingestion - {source_label}",
        source_type_name="File Upload",
        source_details=source_details,
    )


def _resolve_local_source_path(raw_path: str) -> Path:
    """Resolve a local file/folder name from cwd, testfiles, or the workspace."""
    candidate = Path(str(raw_path or "").strip())
    if not candidate.name:
        raise ValueError("Local source input must include a file or folder name.")

    direct_candidates = [candidate]
    if not candidate.is_absolute():
        direct_candidates.append(Path.cwd() / candidate)
        direct_candidates.append(Path.cwd() / "testfiles" / candidate)

    for direct_candidate in direct_candidates:
        if direct_candidate.exists():
            return direct_candidate.resolve()

    if "\\" in str(candidate) or "/" in str(candidate):
        raise FileNotFoundError(f"Local source was not found: {candidate}")

    matches = []
    search_roots = [Path.cwd() / "testfiles", Path.cwd()]
    for search_root in search_roots:
        if not search_root.exists():
            continue
        for match in search_root.rglob(candidate.name):
            if match.is_file() or match.is_dir():
                matches.append(match.resolve())

    if matches:
        unique_matches = sorted(
            {str(match): match for match in matches}.values(),
            key=lambda item: (
                0 if "testfiles" in {part.lower() for part in item.parts} else 1,
                len(item.parts),
                len(str(item)),
            ),
        )
        return unique_matches[0]

    raise FileNotFoundError(
        f"Could not resolve local source '{raw_path}'. Place it in the workspace "
        "or provide its full path."
    )


def _build_base_request(
    *,
    name: str,
    source_type_name: str,
    source_details: Any,
) -> Dict[str, Any]:
    """Create the normalized base request used by shorthand inputs."""
    return {
        "knowledge_base_id": _get_int_env(
            "KNOWLEDGE_BASE_ID",
            DEFAULT_KNOWLEDGE_BASE_ID,
        ),
        "name": name,
        "source_type_name": source_type_name,
        "chunking_details": {
            "chunking_type": "SEMANTIC",
            "chunkSize": 1024,
            "chunkOverlap": 50,
        },
        "embedding_details": {
            "embedding_model_name": DEFAULT_OPENAI_EMBEDDING_MODEL,
            "dimensions": DEFAULT_OPENAI_EMBEDDING_DIMENSIONS,
        },
        "vector_store_details": {
            "vector_store_id": _get_int_env(
                "VECTOR_STORE_ID",
                DEFAULT_VECTOR_STORE_ID,
            ),
            "collection_name": INGESTION_COLLECTION_NAME,
        },
        "source_details": source_details,
    }


def _get_int_env(name: str, default: int) -> int:
    """Parse an integer env var with fallback."""
    raw_value = os.getenv(name)
    if raw_value in (None, ""):
        return default
    try:
        return int(raw_value)
    except ValueError:
        logger.warning("Invalid integer for %s=%r. Using %s.", name, raw_value, default)
        return default


def _get_bool_env(name: str, default: bool) -> bool:
    """Parse a boolean env var with fallback."""
    raw_value = os.getenv(name)
    if raw_value in (None, ""):
        return default

    normalized = str(raw_value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False

    logger.warning("Invalid boolean for %s=%r. Using %s.", name, raw_value, default)
    return default


def _normalize_request_json_string(raw_value: str) -> str:
    """Normalize backtick-delimited JSON-ish literals into valid JSON strings."""
    if "`" not in raw_value:
        return raw_value

    normalized_parts = []
    in_double_quote = False
    in_single_quote = False
    in_backtick = False
    escape = False
    backtick_buffer = []

    for char in raw_value:
        if in_backtick:
            if escape:
                backtick_buffer.append(char)
                escape = False
                continue

            if char == "\\":
                backtick_buffer.append(char)
                escape = True
                continue

            if char == "`":
                normalized_parts.append(json.dumps("".join(backtick_buffer)))
                backtick_buffer = []
                in_backtick = False
                continue

            backtick_buffer.append(char)
            continue

        if escape:
            normalized_parts.append(char)
            escape = False
            continue

        if char == "\\" and (in_double_quote or in_single_quote):
            normalized_parts.append(char)
            escape = True
            continue

        if char == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
            normalized_parts.append(char)
            continue

        if char == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
            normalized_parts.append(char)
            continue

        if char == "`" and not in_double_quote and not in_single_quote:
            in_backtick = True
            backtick_buffer = []
            continue

        normalized_parts.append(char)

    if in_backtick:
        normalized_parts.append("`")
        normalized_parts.extend(backtick_buffer)

    return "".join(normalized_parts)


def print_request_summary(request_data: Dict[str, Any]):
    """Print request summary"""
    logger.info("")
    logger.info("=" * 70)
    logger.info("REQUEST CONFIGURATION")
    logger.info("=" * 70)

    logger.info(f"Knowledge Base ID: {request_data.get('knowledge_base_id')}")
    logger.info(f"Source Type: {request_data.get('source_type_name')}")
    logger.info(f"Cloud Provider: {request_data.get('cloud_provider_name', 'N/A')}")

    # Chunking details
    chunking = request_data.get('chunking_details', {})
    requested_chunking_type = str(
        chunking.get('chunking_type', 'SEMANTIC')
    ).strip().upper()
    if requested_chunking_type and requested_chunking_type != "SEMANTIC":
        logger.info(
            f"Requested Chunking Type: {requested_chunking_type} (normalized to SEMANTIC)"
        )
    logger.info("Chunking Type: SEMANTIC")
    logger.info(f"Chunk Size: {chunking.get('chunkSize', 1024)}")
    logger.info(f"Chunk Overlap: {chunking.get('chunkOverlap', 50)}")

    # Embedding details
    embedding = request_data.get('embedding_details', {})
    requested_embedding_model = str(
        embedding.get('embedding_model_name', DEFAULT_OPENAI_EMBEDDING_MODEL)
    ).strip()
    normalized_requested_embedding_model = (
        requested_embedding_model.lower().replace("_", "-").replace(" ", "-")
    )
    normalized_resolved_embedding_model = (
        DEFAULT_OPENAI_EMBEDDING_MODEL.lower().replace("_", "-").replace(" ", "-")
    )
    if (
        requested_embedding_model
        and normalized_requested_embedding_model != normalized_resolved_embedding_model
    ):
        logger.info(
            f"Requested Embedding Model: {requested_embedding_model} (normalized to {DEFAULT_OPENAI_EMBEDDING_MODEL})"
        )
    requested_embedding_dimensions = embedding.get(
        'dimensions',
        DEFAULT_OPENAI_EMBEDDING_DIMENSIONS,
    )
    if requested_embedding_dimensions not in (
        None,
        "",
        DEFAULT_OPENAI_EMBEDDING_DIMENSIONS,
    ):
        logger.info(
            f"Requested Embedding Dimensions: {requested_embedding_dimensions} (normalized to {DEFAULT_OPENAI_EMBEDDING_DIMENSIONS})"
        )
    logger.info("Embedding Model: %s", DEFAULT_OPENAI_EMBEDDING_MODEL)
    logger.info("Embedding Provider: OpenAI")
    logger.info("Embedding Dimensions: %s", DEFAULT_OPENAI_EMBEDDING_DIMENSIONS)
    logger.info("Embedding Batch Size: 50")

    # Vector store
    vector_store = request_data.get('vector_store_details', {})
    logger.info(f"Vector Store: {vector_store.get('vector_store_name', 'N/A')}")
    logger.info(f"Vector Store Region: {vector_store.get('region', 'N/A')}")
    requested_collection_name = (
        vector_store.get('collection_name')
        or vector_store.get('QDRANT_COLLECTION_NAME')
        or vector_store.get('qdrant_collection_name')
    )
    if requested_collection_name and str(requested_collection_name).strip() != INGESTION_COLLECTION_NAME:
        logger.info(
            f"Requested Collection: {requested_collection_name} (normalized to {INGESTION_COLLECTION_NAME})"
        )
    logger.info(f"Collection Name: {INGESTION_COLLECTION_NAME}")

    # Source details
    source_details = request_data.get('source_details', {})
    if isinstance(source_details, list):
        source_details = source_details[0] if source_details else {}
    if not isinstance(source_details, dict):
        source_details = {}

    if 'file_path' in source_details:
        logger.info(f"Source File Path: {source_details['file_path']}")
    elif 'folder_path' in source_details:
        logger.info(f"Source Folder Path: {source_details['folder_path']}")
    elif 'start_url' in source_details:
        logger.info(f"Website Start URL: {source_details['start_url']}")
    elif 'root_url' in source_details:
        logger.info(f"Website Start URL: {source_details['root_url']}")
    elif 'website_url' in source_details:
        logger.info(f"Website Start URL: {source_details['website_url']}")
    elif 'url' in source_details:
        logger.info(f"Source URL: {source_details['url']}")
    if 'filename' in source_details:
        logger.info(f"Source Filename: {source_details['filename']}")
    if 'host' in source_details:
        logger.info(f"Database Host: {source_details['host']}")
    if 'max_pages' in source_details:
        logger.info(f"Website Max Pages: {source_details['max_pages']}")
    if 'max_depth' in source_details:
        logger.info(f"Website Max Depth: {source_details['max_depth']}")

    logger.info("=" * 70)
    logger.info("")


async def main() -> int:
    """Main entry point"""
    print_banner()

    logger.info("RAG Ingestion Pipeline — starting")

    try:
        # Parse request JSON
        logger.info("Parsing REQUEST_JSON...")
        request_data = parse_request_json()
        print_request_summary(request_data)

        # Import ingestion service
        from services.ingestion_service import IngestionService

        # Initialize and run ingestion
        logger.info("Initializing ingestion service...")
        service = IngestionService(request_data)

        logger.info("Running ingestion pipeline...")
        results = await service.run_ingestion()

        # Print results
        print("\n" + "=" * 70)
        print("  INGESTION RESULTS")
        print("=" * 70)
        print(f"  Status: {results['status'].upper()}")
        print(f"  Source Type: {results['source_type']}")
        print(f"  Knowledge Base: {results['knowledge_base_id']}")
        print()
        print(f"  Files Found: {results['total_files_discovered']}")
        print(f"  Files Processed: {results['total_files_processed']}")
        print(f"  Files Failed: {results['total_files_failed']}")
        print(f"  Chunks Created: {results['total_chunks_created']}")
        print(f"  Vectors Stored: {results['total_vectors_stored']}")
        print("=" * 70 + "\n")

        if results['errors']:
            print("  Errors:")
            for err in results['errors'][:5]:
                print(f"    - {err}")

        if results['status'] == 'FAILED' or results['total_files_failed'] > 0:
            return 1

        return 0

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\nFatal error: {e}\n")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
