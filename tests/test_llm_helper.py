import sys
from pathlib import Path
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm_helper import ask_llm_about_prototype, build_llm_context


class TestLLMHelper(unittest.TestCase):
    def test_build_llm_context_includes_model_and_prototype_evidence(self):
        rows = [{
            "rank": 1,
            "true_text": "Class 0",
            "predicted_text": "Class 1",
            "is_correct": False,
            "sample": [0.1, 0.5, 0.7],
            "index": 12,
        }]
        context = build_llm_context({
            "model_name": "miniRocket_norm",
            "train_score": 0.95,
            "validation_score": 0.8,
            "selection_mode": "central",
            "prototype_count": 3,
        }, rows, "Why did the model predict this prototype incorrectly?")
        self.assertIn("miniRocket_norm", context)
        self.assertIn("central", context)
        self.assertIn("Class 0", context)
        self.assertIn("0.7", context)

    def test_ask_llm_about_prototype_uses_fallback_without_api_key(self):
        """Test that the fallback message is used when no API key is available."""
        with patch.dict("os.environ", {}, clear=True):
            response = ask_llm_about_prototype({
                "model_name": "miniRocket_norm",
                "train_score": 0.95,
                "validation_score": 0.8,
                "selection_mode": "central",
                "prototype_count": 3,
            }, [{
                "rank": 1,
                "true_text": "Class 0",
                "predicted_text": "Class 1",
                "is_correct": False,
                "sample": [0.1, 0.5, 0.7],
                "index": 12,
            }], "Why did the model predict this prototype incorrectly?", api_key=None)
            self.assertIsInstance(response, str)
            self.assertIn("LLM explanation unavailable", response)

    def test_ask_llm_about_prototype_includes_conversation_history(self):
        fake_response = type(
            "Response",
            (),
            {
                "choices": [
                    type(
                        "Choice",
                        (),
                        {
                            "message": type(
                                "Message",
                                (),
                                {"content": "This is an explanation."},
                            )()
                        },
                    )()
                ]
            },
        )()

        fake_client = type(
            "Client",
            (),
            {
                "chat": type(
                    "Chat",
                    (),
                    {
                        "completions": type(
                            "Completions",
                            (),
                            {
                                "create": lambda self, **kwargs: fake_response
                            },
                        )()
                    },
                )()
            },
        )()

        history = [
            {"role": "user", "content": "What does this prototype show?"},
            {"role": "assistant", "content": "It shows a Class 1 pattern."},
        ]

        with patch.dict("os.environ", {"API_KEY": "test-key"}):
            with patch("src.llm_helper.OpenAI", return_value=fake_client):
                response = ask_llm_about_prototype(
                    {
                        "model_name": "miniRocket_norm",
                        "train_score": 0.95,
                        "validation_score": 0.8,
                        "selection_mode": "central",
                        "prototype_count": 3,
                    },
                    [],
                    "Why was it classified as Class 1?",
                    conversation=history,
                )

        self.assertEqual(response, "This is an explanation.")


if __name__ == "__main__":
    unittest.main()

