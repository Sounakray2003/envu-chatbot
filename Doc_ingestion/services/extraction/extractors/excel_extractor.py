"""
Excel File Extractor  –  XLSX / XLS / XLSM
===========================================

Simple, single-pipeline approach:
  - Every sheet in every Excel file is always extracted row-wise.
  - Each row becomes ONE Qdrant point with text in key: value format.
  - No classifier. No dual pipelines. No markdown fallback.

Row text format:
  Sheet: <sheet_name> | Row <N>: Col1: val1 | Col2: val2 | Col3: val3
"""

import logging
import math
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import tiktoken

logger = logging.getLogger(__name__)

# Values that should be treated as empty/null
_NULL_SENTINELS = {"", "nan", "none", "nat", "<na>", "n/a", "null", "na"}


def _string_to_uuid(text: str) -> str:
    """Deterministic UUID v5 from any string."""
    namespace = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")
    return str(uuid.uuid5(namespace, text))


def _count_tokens(text: str) -> int:
    """
    Estimate token count using tiktoken encoder.
    Falls back to simple estimation if encoder fails.
    """
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        return len(tokens)
    except Exception:
        # Fallback: rough estimation (1 token ≈ 4 characters)
        return max(1, len(text) // 4)


def _clean_dataframe(df: pd.DataFrame, min_non_empty_cells: int = 2) -> pd.DataFrame:
    """
    Clean a DataFrame read with dtype=str, keep_default_na=False.

    - Strips whitespace from every cell.
    - Converts all empty / null-sentinel values to pd.NA.
    - Drops rows where every cell is NA (ghost rows from Excel's used-range).
    - Drops rows with fewer than min_non_empty_cells non-empty values
      (prevents partial/sparse rows from being stored as valid data points).
    
    Args:
        df: DataFrame to clean
        min_non_empty_cells: Minimum number of non-empty cells required to keep a row (default: 2)
    """
    result = df.copy()
    for col in result.columns:
        def _clean_cell(v):
            if v is None:
                return pd.NA
            if isinstance(v, float) and math.isnan(v):
                return pd.NA
            s = str(v).strip()
            return pd.NA if s.lower() in _NULL_SENTINELS else s
        result[col] = result[col].apply(_clean_cell)
    
    # Drop rows with all NaN
    result = result.dropna(how="all")
    
    # Drop rows with too few non-empty cells (sparse rows)
    non_empty_counts = result.notna().sum(axis=1)
    result = result[non_empty_counts >= min_non_empty_cells].reset_index(drop=True)
    
    return result


def _row_to_text(sheet_name: str, row_number: int, row: pd.Series) -> str:
    """
    Convert a DataFrame row into a key: value text string.

    Output:  Sheet: <sheet_name> | Row <N>: Col1: val1 | Col2: val2 | ...

    Skips:
    - Columns named "Unnamed: N" (pandas placeholder headers).
    - Cells that are NA or empty after cleaning.
    """
    parts = []
    for col, val in row.items():
        col_str = str(col)
        if col_str.startswith("Unnamed:"):
            continue
        if pd.isna(val):
            continue
        val_str = str(val).strip()
        if val_str.lower() in _NULL_SENTINELS:
            continue
        parts.append(f"{col_str}: {val_str}")

    body = " | ".join(parts) if parts else "(empty row)"
    return f"Sheet: {sheet_name} | Row {row_number}: {body}"


class ExcelExtractor:
    """
    Extracts Excel files row-by-row into Qdrant vector points.

    Every row in every sheet becomes one point with text in key:value format.
    Supports: .xlsx, .xls, .xlsm
    """

    SUPPORTED_FORMATS = {"xlsx", "xls", "xlsm"}

    def validate_file(self, file_path: str) -> Tuple[bool, str]:
        path = Path(file_path)
        if not path.exists():
            return False, "File does not exist"
        ext = path.suffix.lower().lstrip(".")
        if ext not in self.SUPPORTED_FORMATS:
            return False, f"Unsupported format '{ext}'. Supported: {', '.join(self.SUPPORTED_FORMATS)}"
        if path.stat().st_size == 0:
            return False, "File is empty"
        return True, "Valid Excel file"

    def extract_and_store(
        self,
        file_path: str,
        file_id: str,
        filename: str,
        knowledge_base_id: str,
        knowledge_base_name: str,
        user_id: str,
        vector_store,
        batch_size: int = 8,
        embedding_details: Optional[Dict] = None,
        source_mapping_id: Optional[str] = None,
        is_active: bool = True,
        source_type_name: Optional[str] = None,
    ) -> Tuple[bool, Dict, str]:
        """
        Read every sheet, convert each row to key:value text, embed and store in Qdrant.

        Returns:
            (success, summary_dict, fallback_markdown)
            fallback_markdown is always "" — Excel rows go directly to Qdrant,
            never through the chunking pipeline.
        """
        is_valid, msg = self.validate_file(file_path)
        if not is_valid:
            return False, {"error": msg}, ""

        try:
            xl = pd.ExcelFile(file_path)
        except Exception as exc:
            logger.error("Cannot open Excel file '%s': %s", filename, exc)
            return False, {"error": str(exc)}, ""

        total_rows_stored = 0
        total_rows_failed = 0
        total_tokens = 0
        total_characters = 0
        sheet_detail: Dict = {}

        logger.info("📊 Excel extract_and_store: '%s' — %d sheet(s)", filename, len(xl.sheet_names))

        for sheet_idx, sheet_name in enumerate(xl.sheet_names, start=1):
            try:
                # Read everything as strings so no value is silently coerced to NaN
                df_raw = pd.read_excel(
                    file_path,
                    sheet_name=sheet_name,
                    dtype=str,
                    keep_default_na=False,
                )
                df = _clean_dataframe(df_raw)
            except Exception as exc:
                logger.error("Cannot read sheet '%s': %s", sheet_name, exc)
                sheet_detail[sheet_name] = {"error": str(exc)}
                total_rows_failed += 1
                continue

            if df.empty:
                logger.info("  Sheet '%s' is empty — skipping", sheet_name)
                sheet_detail[sheet_name] = {"rows_stored": 0, "rows_failed": 0, "tokens": 0, "characters": 0}
                continue

            logger.info("  Sheet '%s': %d rows to store", sheet_name, len(df))

            stored, failed, tokens, chars = self._store_rows(
                df=df,
                sheet_name=sheet_name,
                sheet_idx=sheet_idx,
                file_id=file_id,
                filename=filename,
                knowledge_base_id=knowledge_base_id,
                knowledge_base_name=knowledge_base_name,
                user_id=user_id,
                vector_store=vector_store,
                batch_size=batch_size,
                embedding_details=embedding_details,
                source_mapping_id=source_mapping_id,
                is_active=is_active,
                source_type_name=source_type_name,
            )

            total_rows_stored += stored
            total_rows_failed += failed
            total_tokens += tokens
            total_characters += chars
            sheet_detail[sheet_name] = {
                "rows_stored": stored,
                "rows_failed": failed,
                "tokens": tokens,
                "characters": chars,
            }
            logger.info("  ✅ Sheet '%s': stored %d / %d rows", sheet_name, stored, len(df))

        summary = {
            "total_rows_stored": total_rows_stored,
            "total_rows_failed": total_rows_failed,
            "total_chunks": total_rows_stored,
            "total_tokens": total_tokens,
            "total_characters": total_characters,
            "sheet_detail": sheet_detail,
            "extraction_method": "excel_row_wise",
        }

        success = total_rows_stored > 0
        logger.info(
            "🎉 '%s' complete: %d rows stored, %d failed, %d tokens, %d characters",
            filename, total_rows_stored, total_rows_failed, total_tokens, total_characters
        )
        # Empty fallback_markdown — all rows go straight to Qdrant, no chunking pipeline
        return success, summary, ""

    def _store_rows(
        self,
        df: pd.DataFrame,
        sheet_name: str,
        sheet_idx: int,
        file_id: str,
        filename: str,
        knowledge_base_id: str,
        knowledge_base_name: str,
        user_id: str,
        vector_store,
        batch_size: int,
        embedding_details: Optional[Dict] = None,
        source_mapping_id: Optional[str] = None,
        is_active: bool = True,
        source_type_name: Optional[str] = None,
    ) -> Tuple[int, int, int, int]:
        """
        Embed and upsert rows in batches.
        
        Returns: (stored, failed, total_tokens, total_characters)
        """
        from services.embedding_service import create_embedding_service
        embedding_service = create_embedding_service(embedding_details or {})

        total_stored = 0
        total_failed = 0
        total_tokens = 0
        total_characters = 0

        for batch_start in range(0, len(df), batch_size):
            batch = df.iloc[batch_start: batch_start + batch_size]
            chunks: List[Dict] = []

            for local_idx, (_, row) in enumerate(batch.iterrows()):
                row_number = batch_start + local_idx + 1  # 1-based row number
                chunk_id = _string_to_uuid(f"{file_id}::{sheet_name}::row::{row_number}")
                row_text = _row_to_text(sheet_name, row_number, row)
                token_count = _count_tokens(row_text)
                char_count = len(row_text)
                
                chunks.append({
                    "text": row_text,
                    "chunk_id": chunk_id,
                    "chunk_index": row_number - 1,
                    "token_count": token_count,
                    "char_count": char_count,
                    "metadata": {
                        "chunk_type": "excel_row",
                        "sheet_name": sheet_name,
                        "row_index": row_number,
                        "page_number": sheet_idx,
                        "source_file": filename,
                        "file_id": file_id,
                        "source_mapping_id": source_mapping_id,
                        "isActive": is_active,
                        "source_type_name": source_type_name,
                        "knowledge_base_id": knowledge_base_id,
                        "knowledge_base_name": knowledge_base_name,
                        "user_id": user_id,
                    },
                })
                
                total_tokens += token_count
                total_characters += char_count

            try:
                texts = [c["text"] for c in chunks]
                embeddings = embedding_service.embed_documents_with_sparse(texts)
                for chunk, emb in zip(chunks, embeddings):
                    chunk["embedding_dense"] = emb["dense"]
                    chunk["embedding_sparse"] = emb["sparse"]

                stored = vector_store.store_chunks_sync(chunks)
                total_stored += stored
            except Exception as exc:
                logger.error(
                    "Batch %d-%d of sheet '%s' failed: %s",
                    batch_start + 1, batch_start + len(batch), sheet_name, exc,
                )
                total_failed += len(batch)

        return total_stored, total_failed, total_tokens, total_characters
