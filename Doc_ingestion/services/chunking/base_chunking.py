"""
chunking/base.py
────────────────
Shared utilities for semantic chunking.
"""

from __future__ import annotations

import hashlib
import logging
import re
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import json

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logging.warning("tiktoken not installed – install with: pip install tiktoken")

try:
    import nltk
    NLTK_AVAILABLE = True
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("nltk not installed – install with: pip install nltk")

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Enum
# ─────────────────────────────────────────────────────────────────────────────

class ChunkingStrategy(str, Enum):
    """Supported chunking strategies."""

    SEMANTIC = "semantic"


# ─────────────────────────────────────────────────────────────────────────────
# Token counter - NOW USES tokenizer_service
# ─────────────────────────────────────────────────────────────────────────────

class TokenCounter:
    """Uses tokenizer_service (HuggingFace) first, then falls back to tiktoken."""

    def __init__(self, model: str = "cl100k_base", tokenizer=None):
        self.tokenizer = tokenizer
        self.encoding = None

        # If tokenizer not passed, try to load via tokenizer_service
        if self.tokenizer is None:
            try:
                from services.tokenizer_service import get_tokenizer
                self.tokenizer = get_tokenizer(model)
                logger.info("TokenCounter: Loaded HuggingFace tokenizer for '%s'", model)
            except Exception as e:
                logger.warning("Failed to load from tokenizer_service for '%s': %s", model, e)
                
                # Fallback to tiktoken
                if TIKTOKEN_AVAILABLE:
                    try:
                        self.encoding = tiktoken.get_encoding(model)
                        logger.info("TokenCounter: Loaded tiktoken encoding: %s", model)
                    except Exception as exc:
                        logger.warning("Failed to load tiktoken encoding '%s': %s", model, exc)

        if self.tokenizer is None and self.encoding is None:
            logger.warning("Using character-based token estimation (4 chars ~= 1 token)")

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0

        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception:
                pass

        if self.encoding:
            try:
                return len(self.encoding.encode(text))
            except Exception:
                pass

        return max(1, len(text) // 4)

    def get_tokens(self, text: str) -> List[int]:
        if not text:
            return []

        if self.tokenizer:
            try:
                return self.tokenizer.encode(text)
            except Exception:
                pass

        if self.encoding:
            try:
                return self.encoding.encode(text)
            except Exception:
                pass
        return []

    def decode_tokens(self, tokens: List[int]) -> str:
        if not tokens:
            return ""

        if self.tokenizer and hasattr(self.tokenizer, "decode"):
            try:
                return self.tokenizer.decode(tokens, skip_special_tokens=True)
            except Exception:
                pass

        if self.encoding:
            try:
                return self.encoding.decode(tokens)
            except Exception:
                pass
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Base mixin
# ─────────────────────────────────────────────────────────────────────────────

class _BaseChunker:
    """
    Mixin that provides _create_chunk, shared text helpers, and the
    character-based fixed-size fallback. All strategy classes inherit this.
    """

    # These must be set by the concrete class __init__
    target_chunk_tokens:  int
    chunk_overlap_tokens: int
    min_chunk_tokens:     int
    max_chunk_tokens:     int
    token_counter:        TokenCounter

    def __init__(
        self,
        target_chunk_tokens: int = 300,
        chunk_overlap_tokens: int = 20,
        min_chunk_tokens: int = 50,
        max_chunk_tokens: int = 1000,
        tokenizer_model: str = "cl100k_base",
        tokenizer=None,                    # Optional: can be passed manually
    ) -> None:
        self.target_chunk_tokens = target_chunk_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
        self.min_chunk_tokens = min_chunk_tokens
        self.max_chunk_tokens = max_chunk_tokens
        
        # Use TokenCounter which now leverages tokenizer_service
        self.token_counter = TokenCounter(model=tokenizer_model, tokenizer=tokenizer)

    def chunk(
        self,
        text: str,
        base_metadata: Dict[str, Any],
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError("Concrete chunkers must implement chunk()")

    # ── Chunk factory ─────────────────────────────────────────────────────────

    def _create_chunk(
        self,
        text:          str,
        chunk_index:   int,
        base_metadata: Dict,
        position:      Optional[int] = None,
        total_tokens:  Optional[int] = None,
        chunk_type:    str = "text",
    ) -> Dict[str, Any]:
        page_number = self._extract_page_number(text)
        token_count = self.token_counter.count_tokens(text)
        chunk: Dict[str, Any] = {
            "chunk_id":    self._generate_chunk_id(text, chunk_index, chunk_type, base_metadata),
            "text":        text,
            "chunk_index": chunk_index,
            "char_count":  len(text),
            "token_count": token_count,
            "metadata": {
                "chunk_index": chunk_index,
                "chunk_type":  chunk_type,
                "page_number": page_number,
                **base_metadata,
            },
        }
        if position is not None and total_tokens is not None:
            chunk["metadata"]["token_position"]      = position
            chunk["metadata"]["total_tokens_in_doc"] = total_tokens
        return chunk

    def _generate_chunk_id(
        self,
        text:          str,
        chunk_index:   int,
        chunk_type:    str = "text",
        base_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        base_metadata = base_metadata or {}

        def _scope_value(key: str) -> str:
            value = base_metadata.get(key)
            if value in (None, ""):
                return ""
            if isinstance(value, (dict, list, tuple)):
                return json.dumps(value, sort_keys=True, default=str)
            return str(value)

        scope_parts = [
            _scope_value("knowledge_base_id"),
            _scope_value("kb_id"),
            _scope_value("job_id"),
            _scope_value("source_mapping_id"),
            _scope_value("source_type_name"),
            _scope_value("provider_name"),
            _scope_value("source"),
            _scope_value("file_id"),
            _scope_value("storage_path"),
            _scope_value("s3_key"),
            _scope_value("gcs_blob"),
            _scope_value("azure_blob"),
            _scope_value("folder_path"),
            _scope_value("zip_archive"),
            _scope_value("zip_entry_name"),
            _scope_value("archive_chain"),
            _scope_value("filename"),
        ]
        file_scope = "|".join(part for part in scope_parts if part)
        if not file_scope:
            file_scope = (
                _scope_value("file_path")
                or _scope_value("filename")
                or ""
            )

        normalized_text = text[:100].strip()
        hash_input = (
            f"{file_scope}|{normalized_text}|{len(text)}|"
            f"{chunk_index}|{chunk_type}"
        )
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    def _extract_page_number(self, text: str) -> int:
        matches = re.findall(r"##\s*Page\s+(\d+)", text, re.IGNORECASE)
        return int(matches[0]) if matches else 1

    # ── Shared text helpers ───────────────────────────────────────────────────

    def _split_into_sentences(self, text: str) -> List[str]:
        if NLTK_AVAILABLE:
            try:
                return [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]
            except Exception as exc:
                logger.warning("NLTK sentence tokenisation failed: %s", exc)
        pattern = r"(?<=[.!?])\s+(?=[A-Z])"
        return [s.strip() for s in re.split(pattern, text) if s.strip()]

    def _is_semantically_similar(
        self, text1: str, text2: str, threshold: float = 0.25
    ) -> bool:
        """Jaccard similarity over 4+-character words."""
        w1 = set(re.findall(r"\b\w{4,}\b", text1.lower()))
        w2 = set(re.findall(r"\b\w{4,}\b", text2.lower()))
        if not w1 or not w2:
            return False
        union = len(w1 | w2)
        return (len(w1 & w2) / union) >= threshold if union else False

    def _extract_overlap_tail(self, text: str, overlap_tokens: int) -> str:
        """Return the trailing *overlap_tokens* worth of text (word-boundary safe)."""
        if not text or not overlap_tokens:
            return ""
        words       = text.split()
        tail_words: List[str] = []
        tail_tokens = 0
        for word in reversed(words):
            word_tokens = self.token_counter.count_tokens(word)
            if tail_tokens + word_tokens > overlap_tokens:
                break
            tail_words.insert(0, word)
            tail_tokens += word_tokens
        return " ".join(tail_words)

    def _split_into_structured_units(self, text: str) -> List[Dict[str, str]]:
        """Parse markdown into typed units: heading | code_block | list | paragraph."""
        units:      List[Dict[str, str]] = []
        heading_re  = re.compile(r"^#{1,6}\s+.+$")
        list_re     = re.compile(r"^\s*([-*+]|\d+\.)\s+.+")
        lines       = text.split("\n")
        cur_text    = ""
        cur_type    = "paragraph"
        i           = 0

        while i < len(lines):
            line = lines[i].rstrip()

            if heading_re.match(line):
                if cur_text.strip():
                    units.append({"type": cur_type, "text": cur_text.strip()})
                units.append({"type": "heading", "text": line.strip()})
                cur_text, cur_type = "", "paragraph"
                i += 1
                continue

            if line.strip().startswith("```"):
                if cur_text.strip():
                    units.append({"type": cur_type, "text": cur_text.strip()})
                code_lines = [lines[i]]
                i += 1
                while i < len(lines) and not lines[i].strip().startswith("```"):
                    code_lines.append(lines[i])
                    i += 1
                if i < len(lines):
                    code_lines.append(lines[i])
                    i += 1
                units.append({"type": "code_block", "text": "\n".join(code_lines)})
                cur_text = ""
                continue

            if list_re.match(line):
                if cur_text.strip() and cur_type != "list":
                    units.append({"type": cur_type, "text": cur_text.strip()})
                    cur_text = ""
                cur_type  = "list"
                cur_text += line + "\n"
                i += 1
                continue

            if not line.strip():
                if cur_text.strip():
                    units.append({"type": cur_type, "text": cur_text.strip()})
                    cur_text, cur_type = "", "paragraph"
                i += 1
                continue

            if cur_type != "paragraph":
                if cur_text.strip():
                    units.append({"type": cur_type, "text": cur_text.strip()})
                cur_text, cur_type = "", "paragraph"
            cur_text += line + " "
            i += 1

        if cur_text.strip():
            units.append({"type": cur_type, "text": cur_text.strip()})
        return units

    # ── Character-based fixed-size fallback ───────────────────────────────────

    def _chunk_fixed_size_chars(
        self, text: str, base_metadata: Dict
    ) -> List[Dict[str, Any]]:
        """Character-based fallback when tokenizer fails."""
        target_chars  = self.target_chunk_tokens  * 4
        overlap_chars = self.chunk_overlap_tokens * 4
        stride_chars  = max(1, target_chars - overlap_chars)
        min_chars     = self.min_chunk_tokens * 4

        raw: List[Tuple[int, str]] = []
        pos = 0
        while pos < len(text):
            raw.append((pos, text[pos: pos + target_chars]))
            pos += stride_chars

        if len(raw) >= 2 and len(raw[-1][1]) < min_chars:
            prev_pos, _ = raw[-2]
            last_pos, last_text = raw[-1]
            raw[-2] = (prev_pos, text[prev_pos: last_pos + len(last_text)])
            raw.pop()

        chunks: List[Dict[str, Any]] = []
        for chunk_index, (_, chunk_text) in enumerate(raw, start=1):
            if chunk_text.strip():
                chunks.append(self._create_chunk(chunk_text, chunk_index, base_metadata))
        return chunks
