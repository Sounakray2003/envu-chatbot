"""
chunking/semantic_chunking.py
─────────────────────────────
Strategy: SEMANTIC
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from .base_chunking import _BaseChunker

logger = logging.getLogger(__name__)


class SemanticChunker(_BaseChunker):
    """Semantic / structure-aware chunking strategy."""

    def __init__(
        self,
        target_chunk_tokens: int = 300,
        chunk_overlap_tokens: int = 20,
        min_chunk_tokens: int = 50,
        max_chunk_tokens: int = 400,
        tokenizer_model: str = "cl100k_base",
        tokenizer=None,                    # ← NEW
    ):
        super().__init__(
            target_chunk_tokens=target_chunk_tokens,
            chunk_overlap_tokens=chunk_overlap_tokens,
            min_chunk_tokens=min_chunk_tokens,
            max_chunk_tokens=max_chunk_tokens,
            tokenizer_model=tokenizer_model,
            tokenizer=tokenizer,           # ← Pass to TokenCounter
        )

    def chunk_semantic(
        self,
        text:                 str,
        base_metadata:        Dict[str, Any],
        max_buffer_size:      Optional[int]   = None,
        breakpoint_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Split *text* into semantically coherent chunks.

        Uses markdown structural units (headings, code blocks, lists,
        paragraphs) as the primary segmentation signal, then uses Jaccard
        similarity to decide whether consecutive paragraphs belong together.
        """
        # FIX-1: guard at the very top — was previously dead code
        if not text.strip():
            return []

        # FIX-2: map to correct instance attributes
        effective_max_tokens  = self.target_chunk_tokens
        effective_overlap     = self.chunk_overlap_tokens
        effective_buffer_size = max_buffer_size  # None = unlimited

        # FIX-8: always resolve to a concrete float
        sim_threshold: float = (
            breakpoint_threshold if breakpoint_threshold is not None else 0.5
        )

        units = self._split_into_structured_units(text)
        if not units:
            return self._chunk_by_sentence(text, base_metadata)

        raw_chunks:    List[Dict[str, Any]] = []
        chunk_index    = 1
        cur_parts:     List[str] = []
        cur_text       = ""
        cur_tokens     = 0
        pending_prefix = ""
        overlap_tail   = ""

        def emit_chunk_text(chunk_text: str, chunk_type: str = "semantic") -> None:
            nonlocal chunk_index, overlap_tail
            for piece in self._split_oversized_text(chunk_text, effective_max_tokens):
                raw_chunks.append(self._create_chunk(
                    piece, chunk_index, base_metadata,
                    chunk_type=chunk_type,
                ))
                chunk_index += 1
            overlap_tail = (
                self._extract_overlap_tail(raw_chunks[-1]["text"], effective_overlap)
                if raw_chunks and effective_overlap else ""
            )

        def flush() -> None:
            nonlocal chunk_index, cur_parts, cur_text, cur_tokens, overlap_tail
            if not cur_parts:
                return
            chunk_text = " ".join(cur_parts)
            emit_chunk_text(chunk_text, chunk_type="semantic")
            cur_parts.clear()
            cur_text  = ""
            cur_tokens = 0

        def start_with_overlap() -> None:
            nonlocal cur_parts, cur_text, cur_tokens
            if overlap_tail:
                o_tok      = self.token_counter.count_tokens(overlap_tail)
                cur_parts  = [overlap_tail]
                cur_text   = overlap_tail
                cur_tokens = o_tok
            else:
                cur_parts, cur_text, cur_tokens = [], "", 0

        def buffer_full() -> bool:
            return (
                effective_buffer_size is not None
                and len(cur_parts) >= effective_buffer_size
            )

        for unit in units:
            unit_type   = unit["type"]
            unit_text   = unit["text"]
            unit_tokens = self.token_counter.count_tokens(unit_text)

            if unit_type in ("heading", "code_block", "list"):
                flush()
                if unit_tokens >= self.min_chunk_tokens:
                    full_text = (
                        (pending_prefix + " " + unit_text).strip()
                        if pending_prefix else unit_text
                    )
                    emit_chunk_text(full_text, chunk_type=unit_type)
                    pending_prefix = ""
                else:
                    pending_prefix = (pending_prefix + " " + unit_text).strip()
                continue

            if unit_tokens > effective_max_tokens:
                flush()
                oversized_text = (
                    (pending_prefix + " " + unit_text).strip()
                    if pending_prefix else unit_text
                )
                pending_prefix = ""
                emit_chunk_text(oversized_text, chunk_type="semantic")
                continue

            if pending_prefix and not cur_parts:
                start_with_overlap()
                p_text     = (pending_prefix + " " + cur_text).strip() if cur_text else pending_prefix
                p_tokens   = self.token_counter.count_tokens(p_text)
                cur_parts  = [p_text]
                cur_text   = p_text
                cur_tokens = p_tokens
                pending_prefix = ""

            if not cur_parts and overlap_tail:
                start_with_overlap()

            fits    = cur_tokens + unit_tokens <= effective_max_tokens
            similar = (not cur_parts) or self._is_semantically_similar(
                cur_text, unit_text, threshold=sim_threshold
            )

            if fits and similar and not buffer_full():
                cur_parts.append(unit_text)
                cur_text   += " " + unit_text
                cur_tokens += unit_tokens
            else:
                flush()
                start_with_overlap()
                cur_parts.append(unit_text)
                cur_text   = (cur_text + " " + unit_text).strip() if cur_text else unit_text
                cur_tokens += unit_tokens

        if pending_prefix:
            cur_parts.append(pending_prefix)
            cur_tokens += self.token_counter.count_tokens(pending_prefix)
        flush()

        merged = self._merge_small_chunks(raw_chunks, base_metadata, effective_max_tokens)

        logger.info(
            "Semantic -> %d chunk(s) (max=%dt overlap=%dt buffer=%s sim=%.2f)",
            len(merged), effective_max_tokens, effective_overlap,
            effective_buffer_size, sim_threshold,
        )
        return merged

    # ── Sentence-based fallback ───────────────────────────────────────────────

    def _chunk_by_sentence(
        self, text: str, base_metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Groups sentences until target_chunk_tokens is reached."""
        sentences = self._split_into_sentences(text)
        if not sentences:
            return self._chunk_fixed_size_chars(text, base_metadata)

        chunks:     List[Dict[str, Any]] = []
        chunk_index = 1
        cur_parts:  List[str] = []
        cur_tokens  = 0

        for sentence in sentences:
            s_tokens = self.token_counter.count_tokens(sentence)

            if s_tokens > self.max_chunk_tokens:
                if cur_parts:
                    chunks.append(self._create_chunk(
                        " ".join(cur_parts), chunk_index, base_metadata,
                    ))
                    chunk_index += 1
                    cur_parts, cur_tokens = [], 0
                # Oversized sentence — split with fixed-size chars
                for sc in self._chunk_fixed_size_chars(sentence, base_metadata):
                    sc["chunk_index"] = chunk_index
                    sc["metadata"]["chunk_index"] = chunk_index
                    chunks.append(sc)
                    chunk_index += 1
                continue

            if cur_tokens + s_tokens > self.target_chunk_tokens and cur_parts:
                chunks.append(self._create_chunk(
                    " ".join(cur_parts), chunk_index, base_metadata,
                ))
                chunk_index += 1
                cur_parts, cur_tokens = [], 0

            cur_parts.append(sentence)
            cur_tokens += s_tokens

        if cur_parts:
            chunks.append(self._create_chunk(
                " ".join(cur_parts), chunk_index, base_metadata,
            ))

        # FIX-6: pass max_tokens explicitly
        return self._merge_small_chunks(chunks, base_metadata, self.target_chunk_tokens)

    def _split_oversized_text(self, text: str, max_tokens: int) -> List[str]:
        """
        Split one oversized unit into token-safe pieces.

        This protects semantic chunking from emitting a single paragraph/list/code
        block that exceeds the embedding model's context window.
        """
        normalized = (text or "").strip()
        if not normalized:
            return []

        if self.token_counter.count_tokens(normalized) <= max_tokens:
            return [normalized]

        sentences = self._split_into_sentences(normalized)
        if len(sentences) > 1:
            pieces: List[str] = []
            cur_parts: List[str] = []
            cur_tokens = 0

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                sentence_tokens = self.token_counter.count_tokens(sentence)

                if sentence_tokens > max_tokens:
                    if cur_parts:
                        pieces.append(" ".join(cur_parts))
                        cur_parts = []
                        cur_tokens = 0
                    pieces.extend(self._split_text_by_token_window(sentence, max_tokens))
                    continue

                if cur_parts and cur_tokens + sentence_tokens > max_tokens:
                    pieces.append(" ".join(cur_parts))
                    cur_parts = [sentence]
                    cur_tokens = sentence_tokens
                else:
                    cur_parts.append(sentence)
                    cur_tokens += sentence_tokens

            if cur_parts:
                pieces.append(" ".join(cur_parts))

            return [piece for piece in pieces if piece.strip()]

        return self._split_text_by_token_window(normalized, max_tokens)

    def _split_text_by_token_window(self, text: str, max_tokens: int) -> List[str]:
        """Split text by token windows, falling back to character chunks if needed."""
        tokens = self.token_counter.get_tokens(text)
        if not tokens:
            return [chunk["text"] for chunk in self._chunk_fixed_size_chars(text, {})]

        overlap = min(self.chunk_overlap_tokens, max(0, max_tokens - 1))
        stride = max(1, max_tokens - overlap)
        pieces: List[str] = []
        position = 0

        while position < len(tokens):
            window = tokens[position: position + max_tokens]
            piece = self.token_counter.decode_tokens(window).strip()
            if piece:
                pieces.append(piece)
            position += stride

        return pieces

    # ── Merge small chunks ────────────────────────────────────────────────────

    def _merge_small_chunks(
        self,
        chunks:        List[Dict[str, Any]],
        base_metadata: Dict[str, Any],
        max_tokens:    Optional[int] = None,  # FIX-5: was missing
    ) -> List[Dict[str, Any]]:
        """Two-pass merge to eliminate chunks smaller than min_chunk_tokens."""

        # FIX-4: token_ceiling was undefined; now uses max_tokens with fallback
        token_ceiling = max_tokens if max_tokens is not None else self.max_chunk_tokens

        def _forward(source: List[Dict]) -> List[Dict]:
            result: List[Dict] = []
            i = 0
            while i < len(source):
                cur = source[i]
                if (
                    cur["token_count"] < self.min_chunk_tokens
                    and i + 1 < len(source)
                ):
                    nxt = source[i + 1]
                    combined_text   = cur["text"] + " " + nxt["text"]
                    combined_tokens = cur["token_count"] + nxt["token_count"]
                    if combined_tokens <= token_ceiling:
                        result.append(self._create_chunk(
                            combined_text,
                            len(result) + 1,
                            base_metadata,
                            chunk_type="semantic_merged",
                        ))
                        i += 2
                        continue
                result.append(cur)
                i += 1
            return result

        after_forward = _forward(chunks)

        # Backward pass for trailing tiny chunk
        if (
            len(after_forward) >= 2
            and after_forward[-1]["token_count"] < self.min_chunk_tokens
        ):
            last = after_forward.pop()
            prev = after_forward[-1]
            combined_text   = prev["text"] + " " + last["text"]
            combined_tokens = prev["token_count"] + last["token_count"]
            if combined_tokens <= token_ceiling:
                after_forward[-1] = self._create_chunk(
                    combined_text,
                    prev["metadata"]["chunk_index"],
                    base_metadata,
                    chunk_type="semantic_merged",
                )
            else:
                after_forward.append(last)

        # Re-index after merges
        for new_idx, chunk in enumerate(after_forward, start=1):
            chunk["chunk_index"] = new_idx
            chunk["metadata"]["chunk_index"] = new_idx

        return after_forward
