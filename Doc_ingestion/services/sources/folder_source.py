"""
Folder/File Upload Source Handler
"""

import logging
import uuid
from pathlib import Path
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

FILE_ID_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


class FolderSource:
    """Handle local folder or file upload"""

    # Supported file types for processing
    SUPPORTED_TYPES = {
        '.pdf', '.docx', '.doc', '.txt', '.md', '.markdown',
        '.json', '.csv', '.tsv', '.xlsx', '.xls', '.xlsm',
        '.html', '.htm', '.xml', '.zip',
        '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.tiff', '.tif'
    }

    @staticmethod
    def _sanitize_failed_filename_component(value: Any, default: str) -> str:
        """Convert local file or folder paths into safe filename-like values."""
        raw_value = str(value or "").strip()
        if not raw_value:
            return default

        normalized = raw_value.replace("\\", "/").rstrip("/")
        if "/" in normalized:
            normalized = normalized.rsplit("/", 1)[-1]

        import re

        normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", normalized).strip("._-")
        return normalized or default

    @classmethod
    def build_discovery_failed_filename(cls, source_details: Dict[str, Any]) -> str:
        """Build the failed-files entry for folder/file discovery failures."""
        return cls._sanitize_failed_filename_component(
            source_details.get("file_path")
            or source_details.get("folder_path")
            or source_details.get("filename"),
            "file_upload",
        )

    @staticmethod
    def _normalize_file_id_seed_component(value: Any) -> str:
        """Normalize arbitrary values into stable file-id seed components."""
        if value in (None, ""):
            return ""
        return str(value).strip()

    def __init__(self, request_data: Dict[str, Any]):
        """Initialize folder source"""
        self.request_data = request_data
        self.source_details = request_data.get('source_details', {})
        self.source_type = str(request_data.get('source_type_name', '')).strip().lower()
        self.file_path = str(self.source_details.get('file_path', '')).strip()
        self.folder_path = str(self.source_details.get('folder_path', '')).strip()
        self.requested_filename = str(self.source_details.get('filename', '')).strip()

        logger.info("Folder Source initialized")
        logger.info(f"  Source type: {self.source_type or 'N/A'}")
        logger.info(f"  file_path: {self.file_path or 'N/A'}")
        logger.info(f"  folder_path: {self.folder_path or 'N/A'}")
        logger.info(f"  Supported file types: {', '.join(sorted(self.SUPPORTED_TYPES))}")

    def _build_generated_file_id(self, file_path: Path) -> str:
        """Generate a deterministic file_id for local file ingestion."""
        stat = file_path.stat()
        normalized_path = str(file_path.resolve()).replace("\\", "/").lower()
        source_mapping_id = (
            self.source_details.get("source_mapping_id")
            or self.source_details.get("id")
            or self.request_data.get("source_mapping_id")
        )
        knowledge_base_id = (
            self.request_data.get("knowledge_base_id")
            or self.request_data.get("kb_id")
        )

        scope_parts = [
            self._normalize_file_id_seed_component(knowledge_base_id),
            self._normalize_file_id_seed_component(source_mapping_id),
            self._normalize_file_id_seed_component(self.source_type or "file upload"),
            normalized_path,
            self._normalize_file_id_seed_component(file_path.name),
            self._normalize_file_id_seed_component(stat.st_size),
            self._normalize_file_id_seed_component(stat.st_mtime_ns),
        ]
        seed = "|".join(part for part in scope_parts if part)
        return str(uuid.uuid5(FILE_ID_NAMESPACE, seed))

    async def discover(self) -> List[Dict[str, Any]]:
        """Discover files from file_path (preferred) or folder_path (legacy)."""
        if self.file_path:
            return self._discover_single_file(Path(self.file_path), source='file_upload')

        if not self.folder_path:
            raise ValueError(
                "No source path provided. Expected source_details.file_path for File Upload."
            )

        folder = Path(self.folder_path)
        if not folder.exists():
            raise ValueError(f"Folder does not exist: {self.folder_path}")

        if folder.is_file():
            logger.warning(
                "source_details.folder_path points to a file. "
                "Use source_details.file_path for File Upload requests."
            )
            return self._discover_single_file(folder, source='file_upload_legacy')

        return self._discover_folder(folder)

    def _discover_single_file(self, file_path: Path, source: str) -> List[Dict[str, Any]]:
        """Discover and validate exactly one file."""
        if not file_path.exists():
            raise ValueError(f"File does not exist: {file_path}")

        if not file_path.is_file():
            raise ValueError(f"Path must point to a file: {file_path}")

        file_type = file_path.suffix.lower()
        if file_type not in self.SUPPORTED_TYPES:
            raise ValueError(
                f"Unsupported file type '{file_type or '[no extension]'}' for "
                f"{file_path.name}. Supported: {', '.join(sorted(self.SUPPORTED_TYPES))}"
            )

        if self.requested_filename and self.requested_filename != file_path.name:
            logger.warning(
                "Filename mismatch in payload: source_details.filename='%s' but "
                "path basename='%s'",
                self.requested_filename,
                file_path.name,
            )

        resolved_file_id = str(
            self.source_details.get('file_id')
            or self._build_generated_file_id(file_path)
        )

        discovered_file = {
            'filename': file_path.name,
            'file_path': str(file_path),
            'file_type': file_type,
            'content_size_bytes': file_path.stat().st_size,
            'source': source,
            'file_id': resolved_file_id,
        }
        logger.info(f"Using single file input: {file_path}")
        return [discovered_file]

    def _discover_folder(self, folder: Path) -> List[Dict[str, Any]]:
        """Discover supported files recursively from a folder."""
        files = []
        unsupported_count = 0
        
        for file_path in folder.rglob('*'):
            if file_path.is_file():
                file_type = file_path.suffix.lower()
                
                # Only process supported file types
                if file_type in self.SUPPORTED_TYPES:
                    files.append({
                        'filename': file_path.name,
                        'file_path': str(file_path),
                        'file_type': file_type,
                        'content_size_bytes': file_path.stat().st_size,
                        'source': 'folder',
                        'file_id': self._build_generated_file_id(file_path),
                    })
                else:
                    unsupported_count += 1
                    logger.debug(f"Skipping unsupported file type: {file_path.name} ({file_type})")

        logger.info(f"Found {len(files)} supported file(s) in folder")
        if unsupported_count > 0:
            logger.info(f"   Skipped {unsupported_count} unsupported file(s)")
        
        return files
