"""
Semantic chunking service.

This codebase supports semantic chunking only.
"""

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_OPENAI_EMBEDDING_MODEL_NAME = (
    str(os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")).strip()
    or "text-embedding-3-large"
)
_OPENAI_TOKENIZER_ENCODING = "cl100k_base"


class ChunkingService:
    """Chunk documents with the semantic chunker only."""

    def __init__(
        self,
        chunking_details: Dict[str, Any],
        embedding_details: Optional[Dict[str, Any]] = None,
    ):
        self.chunking_details = chunking_details or {}
        self.max_chunk_size = self.chunking_details.get("chunkSize", 300)
        self.chunk_overlap = self.chunking_details.get("chunkOverlap", 50)
        self.strategy = "SEMANTIC"
        self.delimiter = self.chunking_details.get("delimiter")
        self.semantic_similarity_threshold = self.chunking_details.get(
            "similarityThreshold"
        )
        self.max_buffer_tokens = self.chunking_details.get("max_buffer_size", 3)

        requested_strategy = str(
            self.chunking_details.get("chunking_type", "SEMANTIC")
        ).strip().upper()
        if requested_strategy and requested_strategy != "SEMANTIC":
            logger.info(
                "Requested chunking type '%s' is no longer supported; using SEMANTIC.",
                requested_strategy,
            )

        embedding_details = embedding_details or {}
        requested_model_name = str(
            embedding_details.get("embedding_model_name", _OPENAI_EMBEDDING_MODEL_NAME)
        ).strip()
        normalized_model_name = requested_model_name.lower().replace("_", "-").strip()
        expected_model_name = _OPENAI_EMBEDDING_MODEL_NAME.lower().replace("_", "-").strip()
        if normalized_model_name and normalized_model_name != expected_model_name:
            logger.info(
                "Requested tokenizer model '%s' is no longer supported; using %s token counting.",
                requested_model_name,
                _OPENAI_TOKENIZER_ENCODING,
            )
        self.model_name = _OPENAI_TOKENIZER_ENCODING

        self.tokenizer = None
        logger.info(
            "Semantic chunker will use OpenAI-compatible token counting via %s.",
            self.model_name,
        )

        logger.info("Chunking Service initialized: SEMANTIC")
        logger.info(" delimiter: %r", self.delimiter)

    def chunk(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk a document with semantic chunking."""
        content = document.get("content", "")
        metadata = document.get("metadata", {})
        filename = document.get("filename", "unknown")

        if not content or not content.strip():
            logger.warning("Empty content for %s", filename)
            return []

        return self._execute_semantic(content, metadata)

    def _split_by_delimiter(self, text: str) -> List[str]:
        """
        Split text by self.delimiter into non-empty segments.
        If the delimiter is not found, return the whole text as one segment.
        """
        if self.delimiter and self.delimiter in text:
            segments = [s.strip() for s in text.split(self.delimiter) if s.strip()]
            if segments:
                return segments
        return [text.strip()] if text.strip() else []

    def _reindex_chunks(
        self, chunks: List[Dict[str, Any]], offset: int
    ) -> List[Dict[str, Any]]:
        """Shift chunk_index values by offset so multi-segment results merge cleanly."""
        for chunk in chunks:
            if "metadata" in chunk and "chunk_index" in chunk["metadata"]:
                chunk["metadata"]["chunk_index"] += offset
            if "chunk_index" in chunk:
                chunk["chunk_index"] += offset
        return chunks

    def _execute_semantic(
        self, text: str, metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Execute semantic chunking."""
        try:
            if self.max_chunk_size is None or self.chunk_overlap is None:
                logger.error(
                    "SEMANTIC failed: chunking_details missing chunkSize or chunkOverlap"
                )
                return []

            from services.chunking.semantic_chunking import SemanticChunker

            chunker = SemanticChunker(
                target_chunk_tokens=self.max_chunk_size,
                chunk_overlap_tokens=self.chunk_overlap,
                min_chunk_tokens=max(10, self.max_chunk_size // 6),
                max_chunk_tokens=self.max_chunk_size * 3,
                tokenizer_model=self.model_name,
                tokenizer=self.tokenizer,
            )

            segments = self._split_by_delimiter(text)
            logger.info(
                "  SEMANTIC: Split into %d segment(s) using delimiter %r",
                len(segments),
                self.delimiter,
            )

            all_chunks: List[Dict[str, Any]] = []
            for segment in segments:
                seg_chunks = chunker.chunk_semantic(
                    segment,
                    metadata,
                    max_buffer_size=self.max_buffer_tokens,
                    breakpoint_threshold=self.semantic_similarity_threshold,
                )
                all_chunks.extend(self._reindex_chunks(seg_chunks, len(all_chunks)))

            logger.info(
                "  SEMANTIC: Created %d chunk(s) from %d segment(s)",
                len(all_chunks),
                len(segments),
            )
            return all_chunks
        except Exception as exc:
            logger.error("SEMANTIC failed: %s", exc, exc_info=True)
            return []
