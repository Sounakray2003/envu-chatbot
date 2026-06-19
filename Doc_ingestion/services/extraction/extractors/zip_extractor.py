"""
ZIP Archive Extractor (Advanced)
Delegates each file inside the archive to the existing extractor registry
Returns a manifest of extracted files with extensive debug logging

Features:
- Per-file delegation to registered extractors
- Nested ZIP support (max depth 5)
- Dynamic extractor registration (only registers extractors that exist)
- Comprehensive logging
"""

import os
import zipfile
import logging
import traceback
import tempfile
import shutil
import uuid
from pathlib import Path, PurePosixPath
from typing import Tuple, Dict, List, Optional, Any
import re

logger = logging.getLogger(__name__)

# ── Supported entry categories ─────────────────────────────────────────────────
TEXT_EXTENSIONS  = {'.txt', '.md', '.csv', '.json', '.xml', '.html', '.htm',
                    '.yaml', '.yml', '.log', '.eml', '.properties', '.vtt',
                    '.markdown', '.py', '.js', '.ts', '.jsx', '.tsx', '.java',
                    '.c', '.cpp', '.h', '.hpp', '.cs', '.go', '.rs', '.sh',
                    '.bat', '.ps1', '.toml', '.env', '.sql', '.ini', '.cfg',
                    '.conf', '.tsv', '.jsonl', '.ndjson'}
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.tiff', '.tif'}
DOC_EXTENSIONS   = {'.pdf', '.docx', '.xlsx', '.xlsm', '.pptx', '.msg', '.epub', '.doc', '.xls'}
ZIP_EXTENSIONS   = {'.zip'}

MAX_TEXT_ENTRY_BYTES  = 512 * 1024   # 512 KB
MAX_NESTED_ZIP_DEPTH  = 5
SAFE_PATH_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1F]')
FILE_ID_NAMESPACE = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')


class ZIPExtractor:
    """
    ZIP Archive Extractor with advanced features.
    
    Features:
    - Dynamically registers only extractors that exist in the folder
    - Per-file delegation to extract various formats
    - Creates individual markdown files per extracted file (no mixing)
    - Preserves folder hierarchy from ZIP
    - Support for nested ZIPs with depth limiting
    - Comprehensive debug logging
    """

    def __init__(self, config: Optional[Dict] = None):
        self.supported_formats = {'zip'}
        self._available_extractors = {}
        self._output_dir = Path.cwd() / "output_files"  # Base output directory
        self._register_available_extractors()
        
        logger.info("ZIP Extractor initialized (Advanced Mode)")

    def _register_available_extractors(self) -> None:
        """
        Only register extractors that actually exist in the extractors folder.
        This prevents ImportError for missing extractors.
        """
        extractors_path = Path(__file__).parent
        
        # Map of file to extractor class name
        extractor_map = {
            'pdf_extractor.py': ('PDFExtractor', '.pdf'),
            'docx_extractor.py': ('DocxExtractor', ('.docx', '.doc')),
            'excel_extractor.py': ('ExcelExtractor', ('.xlsx', '.xlsm', '.xls')),
            'html_extractor.py': ('HTMLExtractor', ('.html', '.htm')),
            'xml_extractor.py': ('XMLExtractor', '.xml'),
            'csv_extractor.py': ('CSVExtractor', ('.csv', '.tsv')),
            'json_extractor.py': ('JSONExtractor', '.json'),
            'text_extractor.py': ('TextExtractor', ('.txt', '.md', '.markdown')),
            'image_extractor.py': ('ImageExtractor', ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.tiff', '.tif')),
        }
        
        for filename, (class_name, suffixes) in extractor_map.items():
            if (extractors_path / filename).exists():
                module_name = filename.replace('.py', '')
                try:
                    module = __import__(f'services.extraction.extractors.{module_name}', 
                                      fromlist=[class_name], level=0)
                    extractor_class = getattr(module, class_name, None)
                    if extractor_class:
                        if isinstance(suffixes, str):
                            self._available_extractors[suffixes] = extractor_class
                        else:
                            for suffix in suffixes:
                                self._available_extractors[suffix] = extractor_class
                        logger.debug(f"✓ Registered {class_name} for {suffixes}")
                except Exception as e:
                    logger.debug(f"Could not register {class_name}: {e}")

    def _get_supported_formats(self) -> List[str]:
        return ["zip"]

    def validate_file(self, file_path: str) -> Tuple[bool, str]:
        """Validate ZIP file — checks existence, extension, emptiness, and valid ZIP magic."""
        path = Path(file_path)

        if not path.exists():
            return False, f"File does not exist: {file_path}"
        if path.suffix.lower() != '.zip':
            return False, f"Not a ZIP file: {path.suffix}"
        if path.stat().st_size == 0:
            return False, "File is empty (0 bytes)"

        if not zipfile.is_zipfile(file_path):
            return False, "File is not a valid ZIP archive (bad magic bytes)"

        try:
            with zipfile.ZipFile(file_path, 'r') as zf:
                if not zf.namelist():
                    return False, "ZIP archive is empty (no entries)"
                first = zf.infolist()[0]
                if first.flag_bits & 0x1:   # bit 0 = encryption flag
                    return False, "ZIP archive is password-protected — decryption not supported"
        except zipfile.BadZipFile as e:
            return False, f"Corrupt ZIP archive: {e}"
        except Exception as e:
            logger.warning(f"Could not inspect ZIP entries: {e}")

        return True, "Valid ZIP archive"

    def extract(self, file_path: str) -> Tuple[str, bool, Dict]:
        """
        Extract ZIP creating individual markdown files per extracted file.
        
        Returns:
        - (manifest_info, success, metadata)
        - manifest_info: Summary of created files
        - metadata: List of created markdown file paths in 'created_files' key
        """
        try:
            logger.info("=" * 80)
            logger.info("ZIP EXTRACTION - START (Per-File Mode)")
            logger.info("=" * 80)
            logger.info(f"File: {file_path}")

            is_valid, msg = self.validate_file(file_path)
            if not is_valid:
                logger.error(f"Validation failed: {msg}")
                return "", False, {'error': msg}

            logger.info("✓ Validation passed")

            zip_stem = Path(file_path).stem
            archive_markdown_root = self._output_dir / "markdown" / zip_stem
            archive_markdown_root.mkdir(parents=True, exist_ok=True)
            
            created_files = []  # Track all created markdown files
            stats = {'entries_processed': 0, 'entries_skipped': 0, 'files_created': 0}

            with zipfile.ZipFile(file_path, 'r') as zf:
                all_names = zf.namelist()
                logger.info(f"Total entries: {len(all_names)}")

                for entry_name in sorted(all_names):
                    if self._should_skip_entry(entry_name):
                        stats['entries_skipped'] += 1
                        continue

                    # Skip directory entries
                    if entry_name.endswith('/'):
                        continue

                    suffix = self._resolve_entry_suffix(entry_name)

                    try:
                        if suffix in ZIP_EXTENSIONS:
                            # Nested ZIP - skip for now or handle separately
                            logger.info(f"Skipping nested ZIP: {entry_name}")
                            continue

                        elif suffix in self._available_extractors or suffix in TEXT_EXTENSIONS or suffix in DOC_EXTENSIONS:
                            # Extract file content
                            logger.info(f"Extracting: {entry_name}")
                            content = self._extract_file_content(zf, entry_name, suffix)
                            
                            if content and content.strip():
                                # Create individual markdown file with folder structure preserved
                                markdown_path = self._create_individual_markdown(
                                    archive_markdown_root, 
                                    entry_name, 
                                    content
                                )
                                if markdown_path:
                                    created_files.append(markdown_path)
                                    stats['entries_processed'] += 1
                                    stats['files_created'] += 1
                                    logger.info(f"✓ Created: {markdown_path}")
                            else:
                                logger.warning(f"No content extracted from: {entry_name}")

                        elif suffix in IMAGE_EXTENSIONS:
                            # Skip images in this pass (handled separately if needed)
                            logger.info(f"Image file (skipped in extraction): {entry_name}")
                            continue

                        else:
                            logger.info(f"Unsupported type: {entry_name} ({suffix})")

                    except Exception as e:
                        logger.warning(f"Error processing {entry_name}: {e}")

            # Create summary
            summary = f"# ZIP Extraction Summary\n\n"
            summary += f"**Archive**: {zip_stem}\n"
            summary += f"**Files Created**: {stats['files_created']}\n"
            summary += f"**Files Processed**: {stats['entries_processed']}\n"
            summary += f"**Files Skipped**: {stats['entries_skipped']}\n"
            summary += f"**Output Directory**: markdown/{zip_stem}/\n\n"
            summary += f"## Created Files\n\n"
            for idx, fpath in enumerate(created_files, 1):
                relative_path = Path(fpath).relative_to(self._output_dir / "markdown" / zip_stem)
                summary += f"{idx}. {relative_path}\n"

            metadata = self.get_metadata(file_path)
            metadata.update({
                'entries_processed': stats['entries_processed'],
                'entries_skipped': stats['entries_skipped'],
                'files_created': stats['files_created'],
                'created_files': created_files,
                'extraction_method': 'per_file_individual',
                'output_directory': str(archive_markdown_root),
            })

            logger.info(f"✓ Extraction complete: {stats['files_created']} files created")
            logger.info("=" * 80)

            return summary, True, metadata

        except Exception as e:
            logger.error(f"CRITICAL ERROR: {str(e)}")
            traceback.print_exc()
            return "", False, {'error': str(e)}

    def extract_file_infos(
        self,
        file_path: str,
        destination_root: str,
        parent_file_info: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Expand a ZIP archive into real files on disk and return pipeline-ready
        file_info dictionaries for each supported archive entry.

        This is used by ingestion to fan out ZIP contents so each inner file can
        run through the existing ingestion pipeline exactly like a standalone
        upload.
        """
        is_valid, msg = self.validate_file(file_path)
        if not is_valid:
            raise ValueError(msg)

        destination = Path(destination_root)
        destination.mkdir(parents=True, exist_ok=True)

        extracted_files: List[Dict[str, Any]] = []
        stats = {
            'entries_processed': 0,
            'entries_skipped': 0,
            'nested_archives': 0,
        }

        archive_label = (
            (parent_file_info or {}).get('filename')
            or Path(file_path).name
        )

        logger.info("📦 Extracting ZIP: %s", archive_label)

        self._collect_extracted_file_infos(
            zip_path=Path(file_path),
            destination_root=destination,
            parent_file_info=parent_file_info or {},
            extracted_files=extracted_files,
            stats=stats,
            depth=0,
            archive_chain=[str(archive_label).replace("\\", "/")],
        )

        logger.info("  ✓ Extracted %d file(s) from %s", len(extracted_files), archive_label)
        logger.info(
            "ZIP expanded for pipeline: %s -> %d supported file(s)",
            file_path,
            len(extracted_files),
        )
        if extracted_files:
            logger.info(
                "ZIP extracted entries: %s",
                [file_info.get('zip_entry_name') for file_info in extracted_files],
            )
        return extracted_files

    def _extract_file_content(self, zf: zipfile.ZipFile, entry_name: str, suffix: str) -> Optional[str]:
        """
        Extract content from a file in ZIP.
        Returns content as markdown string or None if extraction fails.
        """
        try:
            if suffix in self._available_extractors:
                # Use registered extractor
                return self._extract_with_registered_extractor(zf, entry_name, suffix)
            elif suffix in TEXT_EXTENSIONS:
                # Read as text
                content, ok = self._read_text_entry(zf, entry_name)
                if ok and content:
                    return f"```{suffix.lstrip('.')}\n{content}\n```"
            elif suffix in DOC_EXTENSIONS and suffix not in self._available_extractors:
                # Try text fallback for unsupported doc types
                content, ok = self._read_text_entry(zf, entry_name)
                if ok and content:
                    return f"*[Document: {entry_name}]*\n\n{content}"
            
            return None
        except Exception as e:
            logger.error(f"Error extracting {entry_name}: {e}")
            return None

    def _create_individual_markdown(self, archive_root: Path, entry_name: str, content: str) -> Optional[str]:
        """
        Create individual markdown file preserving folder structure from ZIP.
        
        Example:
        - archive_root: output_files/markdown/test_data/
        - entry_name: "New folder/New folder/document.pdf"
        - Creates: output_files/markdown/test_data/New folder/New folder/document.md
        
        Returns: Full path to created markdown file or None
        """
        try:
            markdown_relative_path = self.source_path_to_markdown_path(entry_name)
            markdown_file = archive_root / markdown_relative_path
            markdown_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Write content
            markdown_file.write_text(content, encoding='utf-8')
            
            return str(markdown_file)
        except Exception as e:
            logger.error(f"Error creating markdown for {entry_name}: {e}")
            return None

    def _extract_nested_zip(self, parent_zf: zipfile.ZipFile, entry_name: str, parent_stem: str) -> str:
        """Recursively extract nested ZIP."""
        try:
            raw_bytes = parent_zf.read(entry_name)
            if not raw_bytes:
                return "*(Nested ZIP is empty)*"

            import io
            nested_buf = io.BytesIO(raw_bytes)
            if not zipfile.is_zipfile(nested_buf):
                return f"*(Not a valid ZIP: {entry_name})*"

            nested_buf.seek(0)
            nested_stem = Path(entry_name).stem
            lines = [f"*Nested archive: {entry_name}*\n"]

            with zipfile.ZipFile(nested_buf, 'r') as nested_zf:
                for nested_entry in sorted(nested_zf.namelist()):
                    if self._should_skip_entry(nested_entry):
                        continue

                    lines.append(f"  - `{nested_entry}`")

            return '\n'.join(lines)

        except Exception as e:
            logger.error(f"Nested ZIP error for {entry_name}: {e}")
            return f"*(Error extracting nested ZIP: {e})*"

    def _extract_with_registered_extractor(self, zf: zipfile.ZipFile, entry_name: str, suffix: str) -> Optional[str]:
        """Use registered extractor for this file type."""
        try:
            extractor_class = self._available_extractors.get(suffix)
            if not extractor_class:
                return None

            # Write to temp file
            data = zf.read(entry_name)
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(data)
                tmp_path = tmp.name

            try:
                extractor = extractor_class()
                result = extractor.extract(tmp_path)

                # Handle different return types
                if isinstance(result, tuple) and len(result) >= 2:
                    content, success = result[0], result[1]
                    if success and content and content.strip():
                        return content
                elif isinstance(result, str) and result.strip():
                    return result

                return None
            finally:
                try:
                    os.unlink(tmp_path)
                except:
                    pass

        except Exception as e:
            logger.warning(f"Extractor failed for {entry_name}: {e}")
            return None

    def _collect_extracted_file_infos(
        self,
        zip_path: Path,
        destination_root: Path,
        parent_file_info: Dict[str, Any],
        extracted_files: List[Dict[str, Any]],
        stats: Dict[str, int],
        depth: int,
        archive_chain: List[str],
    ) -> None:
        """Recursively extract supported archive entries into pipeline file_info dicts."""
        if depth > MAX_NESTED_ZIP_DEPTH:
            logger.warning(
                "Skipping nested ZIP beyond max depth %d: %s",
                MAX_NESTED_ZIP_DEPTH,
                zip_path,
            )
            stats['entries_skipped'] += 1
            return

        with zipfile.ZipFile(zip_path, 'r') as zf:
            for info in sorted(zf.infolist(), key=lambda item: item.filename):
                entry_name = info.filename

                if self._should_skip_entry(entry_name):
                    stats['entries_skipped'] += 1
                    continue

                if info.is_dir() or entry_name.endswith('/'):
                    continue

                suffix = self._resolve_entry_suffix(entry_name)

                if suffix in IMAGE_EXTENSIONS:
                    logger.info(f"Skipping image entry inside ZIP: {entry_name}")
                    stats['entries_skipped'] += 1
                    continue

                if (
                    suffix not in ZIP_EXTENSIONS
                    and suffix not in TEXT_EXTENSIONS
                    and suffix not in DOC_EXTENSIONS
                    and suffix not in self._available_extractors
                ):
                    logger.info(f"Skipping unsupported ZIP entry: {entry_name} ({suffix})")
                    stats['entries_skipped'] += 1
                    continue

                extracted_path = destination_root / self.source_path_to_safe_path(entry_name)
                extracted_path.parent.mkdir(parents=True, exist_ok=True)

                with zf.open(info, 'r') as source_stream, open(extracted_path, 'wb') as target_stream:
                    shutil.copyfileobj(source_stream, target_stream)

                normalized_entry_name = entry_name.replace("\\", "/")

                if suffix in ZIP_EXTENSIONS:
                    logger.info(
                        "[ZIP] Found nested archive: %s (depth=%d)",
                        normalized_entry_name,
                        depth + 1,
                    )
                    if depth + 1 > MAX_NESTED_ZIP_DEPTH:
                        logger.warning(
                            "Skipping nested ZIP beyond max depth %d: %s",
                            MAX_NESTED_ZIP_DEPTH,
                            normalized_entry_name,
                        )
                        stats['entries_skipped'] += 1
                        continue

                    stats['nested_archives'] += 1
                    nested_destination = extracted_path.with_suffix("")
                    nested_destination.mkdir(parents=True, exist_ok=True)
                    self._collect_extracted_file_infos(
                        zip_path=extracted_path,
                        destination_root=nested_destination,
                        parent_file_info=parent_file_info,
                        extracted_files=extracted_files,
                        stats=stats,
                        depth=depth + 1,
                        archive_chain=[*archive_chain, normalized_entry_name],
                    )
                    continue

                logger.info(
                    "[ZIP] Extracted entry: %s (%s, %d bytes)",
                    normalized_entry_name,
                    suffix,
                    extracted_path.stat().st_size,
                )
                extracted_files.append(
                    self._build_extracted_file_info(
                        extracted_path=extracted_path,
                        entry_name=normalized_entry_name,
                        file_type=suffix,
                        parent_file_info=parent_file_info,
                        archive_chain=archive_chain,
                        depth=depth,
                    )
                )
                stats['entries_processed'] += 1

    def _build_extracted_file_info(
        self,
        extracted_path: Path,
        entry_name: str,
        file_type: str,
        parent_file_info: Dict[str, Any],
        archive_chain: List[str],
        depth: int,
    ) -> Dict[str, Any]:
        """Build a file_info dict for an extracted ZIP entry."""
        inherited_fields = {
            key: value
            for key, value in parent_file_info.items()
            if key not in {
                'filename',
                'file_path',
                'file_type',
                'size_bytes',
                'content',
                'file_id',
            }
        }

        file_scope = (
            parent_file_info.get('file_id')
            or parent_file_info.get('filename')
            or parent_file_info.get('file_path')
            or archive_chain[0]
        )
        storage_path = "/".join([*archive_chain, entry_name]).replace("\\", "/")
        derived_file_id = str(uuid.uuid5(FILE_ID_NAMESPACE, storage_path + f"::{file_scope}"))

        return {
            **inherited_fields,
            'filename': entry_name,
            'file_path': str(extracted_path),
            'file_type': file_type,
            'size_bytes': extracted_path.stat().st_size,
            'source': 'zip',
            'file_id': derived_file_id,
            'storage_path': storage_path,
            'zip_archive': archive_chain[0],
            'zip_entry_name': entry_name,
            'zip_depth': depth,
            'archive_chain': archive_chain.copy(),
            'parent_source': parent_file_info.get('source'),
        }

    def _read_text_entry(self, zf: zipfile.ZipFile, entry_name: str) -> Tuple[str, bool]:
        """Read text entry from ZIP."""
        try:
            raw = zf.read(entry_name)
            if len(raw) > MAX_TEXT_ENTRY_BYTES:
                raw = raw[:MAX_TEXT_ENTRY_BYTES]
            try:
                return raw.decode('utf-8'), True
            except UnicodeDecodeError:
                return raw.decode('latin-1', errors='replace'), True
        except Exception as e:
            logger.warning(f"Error reading text {entry_name}: {e}")
            return "", False

    @staticmethod
    def _should_skip_entry(entry_name: str) -> bool:
        """Check if entry should be skipped."""
        skip_patterns = {'__MACOSX', '.DS_Store', 'Thumbs.db', '.git', '.svn', 
                        '.hg', '.bzr', '__pycache__'}
        name = Path(entry_name).name
        
        if not name:
            return True
        
        for pattern in skip_patterns:
            if pattern in entry_name:
                return True
        
        return False

    @staticmethod
    def source_path_to_safe_path(source_path: str, forced_suffix: Optional[str] = None) -> Path:
        """Convert an archive entry path to a sanitized relative filesystem path."""
        raw_path = source_path.replace("\\", "/").strip("/")
        if not raw_path:
            return Path("untitled" + (forced_suffix or ""))

        pure_path = PurePosixPath(raw_path)
        safe_parts = [
            ZIPExtractor._sanitize_path_part(part)
            for part in pure_path.parts
            if part not in {"", ".", ".."}
        ]

        if not safe_parts:
            return Path("untitled" + (forced_suffix or ""))

        if forced_suffix is not None:
            final_name = safe_parts[-1]
            final_stem = Path(final_name).stem if Path(final_name).suffix else final_name
            safe_parts[-1] = f"{final_stem}{forced_suffix}"

        return Path(*safe_parts)

    @staticmethod
    def source_path_to_markdown_path(source_path: str) -> Path:
        """Convert source path to markdown path."""
        return ZIPExtractor.source_path_to_safe_path(source_path, forced_suffix=".md")

    @staticmethod
    def _sanitize_path_part(value: str) -> str:
        """Sanitize path component"""
        if value in {"", ".", ".."}:
            return "untitled"
        cleaned = SAFE_PATH_CHARS.sub("_", value).strip()
        if cleaned.startswith(".") and len(cleaned) > 1:
            cleaned = f"_{cleaned}"
        if cleaned in {"", ".", ".."}:
            return "untitled"
        return cleaned

    @staticmethod
    def _resolve_entry_suffix(entry_name: str) -> str:
        """Resolve a file type for normal files and supported dotfiles like `.env`."""
        entry_path = Path(entry_name)
        suffix = entry_path.suffix.lower()
        if suffix:
            return suffix

        filename = entry_path.name.lower()
        if filename in TEXT_EXTENSIONS:
            return filename

        return ""

    def get_metadata(self, file_path: str) -> Dict:
        """Get file metadata."""
        path = Path(file_path)
        return {
            "filename": path.name,
            "size_bytes": path.stat().st_size,
            "extension": path.suffix.lower(),
        }
