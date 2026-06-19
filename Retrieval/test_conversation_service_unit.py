import unittest
from dataclasses import dataclass, field
from typing import Any, Dict, List

from services.conversation_service import ConversationService
from services.session_manager import SessionManager


@dataclass
class FakeRetrievalResult:
    text: str
    score: float = 0.9
    metadata: Dict[str, Any] = field(default_factory=dict)


class FakeRewriter:
    def __init__(self):
        self.calls: List[Dict[str, Any]] = []

    def rewrite(self, question, history):
        self.calls.append({"question": question, "history": list(history)})
        return f"rewritten:{question}"


class ConversationServiceTest(unittest.TestCase):
    def setUp(self):
        SessionManager.clear()

    def build_service(self, results=None, answer_generator=None):
        self.retriever_calls = []

        def retriever(**kwargs):
            self.retriever_calls.append(kwargs)
            return list(results or [])

        def default_answer_generator(question, chunks):
            texts = [chunk.text.strip() for chunk in chunks[:3] if chunk.text]
            return " ".join(texts)

        self.rewriter = FakeRewriter()
        return ConversationService(
            query_rewriter=self.rewriter,
            retriever=retriever,
            answer_generator=answer_generator or default_answer_generator,
        )

    def test_generates_stable_auto_session_for_user_id(self):
        service = self.build_service([FakeRetrievalResult("Answer chunk")])

        first = service.handle_question({"user_id": "alice", "question": "What is it?"})
        second = service.handle_question({"user_id": "alice", "question": "And now?"})

        self.assertEqual(first["session_id"], second["session_id"])
        self.assertTrue(first["session_id"].startswith("auto:"))

    def test_raises_for_blank_question_before_retrieval(self):
        service = self.build_service([FakeRetrievalResult("Unused")])

        with self.assertRaises(ValueError):
            service.handle_question({"user_id": "alice", "question": "   "})

        self.assertEqual(self.retriever_calls, [])

    def test_returns_fallback_answer_for_empty_retrieval_results(self):
        service = self.build_service([])

        response = service.handle_question({"session_id": "s1", "question": "Missing topic"})

        self.assertEqual(
            response["answer"],
            "I could not find this information in the Envu India knowledge base.",
        )
        self.assertEqual(response["retrieved_chunks"], [])

    def test_uses_previous_turns_for_follow_up_rewrite(self):
        service = self.build_service(
            [FakeRetrievalResult("First"), FakeRetrievalResult("Second")]
        )

        service.handle_question({"session_id": "s1", "question": "Tell me about Alpha"})
        service.handle_question({"session_id": "s1", "question": "When was it founded?"})

        self.assertEqual(
            self.rewriter.calls[1]["history"][0]["content"],
            "Tell me about Alpha",
        )
        self.assertEqual(self.retriever_calls[1]["query"], "rewritten:When was it founded?")

    def test_answer_uses_top_three_non_empty_chunks(self):
        service = self.build_service(
            [
                FakeRetrievalResult("One"),
                FakeRetrievalResult(""),
                FakeRetrievalResult("Two"),
                FakeRetrievalResult("Ignored"),
            ]
        )

        response = service.handle_question({"session_id": "s1", "question": "Summarize"})

        self.assertEqual(response["answer"], "One Two")

    def test_backend_provided_history_standard_format(self):
        service = self.build_service([FakeRetrievalResult("Response")])
        payload = {
            "session_id": "s_standard",
            "question": "What is the second query?",
            "history": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"}
            ]
        }
        service.handle_question(payload)
        
        # Verify SessionManager has been updated with the history + current turn
        stored_history = SessionManager.get_history("s_standard")
        self.assertEqual(len(stored_history), 4)
        self.assertEqual(stored_history[0]["role"], "user")
        self.assertEqual(stored_history[0]["content"], "Hello")
        self.assertEqual(stored_history[1]["role"], "assistant")
        self.assertEqual(stored_history[1]["content"], "Hi there")
        self.assertEqual(stored_history[2]["role"], "user")
        self.assertEqual(stored_history[2]["content"], "What is the second query?")
        self.assertEqual(stored_history[3]["role"], "assistant")
        self.assertEqual(stored_history[3]["content"], "Response")
        
        # Verify rewriter received the correct history (excluding current turn)
        self.assertEqual(len(self.rewriter.calls), 1)
        self.assertEqual(len(self.rewriter.calls[0]["history"]), 2)
        self.assertEqual(self.rewriter.calls[0]["history"][0]["content"], "Hello")

    def test_backend_provided_history_custom_qa_format(self):
        service = self.build_service([FakeRetrievalResult("Response")])
        payload = {
            "session_id": "s_custom",
            "question": "where do i get reward listing?",
            "history": [
                {
                    "question": "",
                    "answer": "Hi, I am Pest Expert Assistant. How can I help you today?",
                    "role": "assistant"
                },
                {
                    "question": "where do i get reward listing in pest expert 360 app?",
                    "answer": "You can see the reward list..."
                }
            ]
        }
        service.handle_question(payload)
        
        # Verify SessionManager has been updated and custom format normalized + current turn appended
        stored_history = SessionManager.get_history("s_custom")
        self.assertEqual(len(stored_history), 5)
        self.assertEqual(stored_history[0]["role"], "assistant")
        self.assertEqual(stored_history[0]["content"], "Hi, I am Pest Expert Assistant. How can I help you today?")
        self.assertEqual(stored_history[1]["role"], "user")
        self.assertEqual(stored_history[1]["content"], "where do i get reward listing in pest expert 360 app?")
        self.assertEqual(stored_history[2]["role"], "assistant")
        self.assertEqual(stored_history[2]["content"], "You can see the reward list...")
        self.assertEqual(stored_history[3]["role"], "user")
        self.assertEqual(stored_history[3]["content"], "where do i get reward listing?")
        self.assertEqual(stored_history[4]["role"], "assistant")
        self.assertEqual(stored_history[4]["content"], "Response")

    def test_empty_or_absent_history_uses_session_manager(self):
        service = self.build_service([FakeRetrievalResult("Response")])
        
        # First turn
        service.handle_question({"session_id": "s_fallback", "question": "First question"})
        
        # Second turn (no history passed in request, should fallback to SessionManager)
        payload = {
            "session_id": "s_fallback",
            "question": "Second question"
        }
        service.handle_question(payload)
        
        # Verify rewriter received the first turn from SessionManager
        self.assertEqual(len(self.rewriter.calls), 2)
        self.assertEqual(self.rewriter.calls[1]["history"][0]["role"], "user")
        self.assertEqual(self.rewriter.calls[1]["history"][0]["content"], "First question")


if __name__ == "__main__":
    unittest.main()

