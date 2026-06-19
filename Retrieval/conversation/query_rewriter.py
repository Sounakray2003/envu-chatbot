import os
import json
import logging
from typing import List, Dict, Any
from services.embedding_service import get_dense_embedding  # reuse embedding helper from retrieval_new if available
from services.vector_store_service import VectorStoreService  # placeholder import, actual implementation may differ
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class QueryRewriter:
    """Rewrite a follow‑up question into a standalone search query.
    Uses the same OpenAI model configured for embeddings.
    """

    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for query rewriting")

    def _call_chat(self, messages: List[Dict[str, str]]) -> str:
        import requests
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.chat_model,
            "messages": messages,
            "temperature": 0.0,
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    def rewrite(self, current_question: str, history: List[Dict[str, str]]) -> str:
        """Create a prompt for the LLM that includes the current question and recent history.
        History is a list of dicts with keys "role" ("user"/"assistant") and "content".
        Returns a single‑line standalone query.
        """
        # Build a concise history string (last 5 turns)
        recent = history[-10:]  # up to 5 user + 5 assistant turns
        hist_str = "\n".join([f"{turn['role'].capitalize()}: {turn['content']}" for turn in recent])
        system_prompt = (
            "You are a helpful assistant that converts a follow‑up conversational question into a "
            "standalone search query. Preserve the original intent while removing pronouns and references."
        )
        user_prompt = (
            f"Conversation history:\n{hist_str}\n\nCurrent question: {current_question}\n\n"
            "Rewrite the above question as a single, self‑contained search query."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        rewritten = self._call_chat(messages)
        logger.info("Rewritten query: %s", rewritten)
        return rewritten
