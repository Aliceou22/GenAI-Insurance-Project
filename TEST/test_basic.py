import os
import unittest
import pandas as pd
from unittest.mock import patch
from SCR.process_transcripts import load_member_conversations, classify_text_with_chat

class TestTranscriptProcessing(unittest.TestCase):

    def setUp(self):
        """Create a mock directory with one fake transcript file."""
        self.test_dir = "test_transcripts"
        os.makedirs(self.test_dir, exist_ok=True)
        self.test_file_path = os.path.join(self.test_dir, "sample.txt")

        with open(self.test_file_path, "w", encoding="utf-8") as f:
            f.write("""Member: Hello, I need help with my insurance claim.\n
PA Agent: Sure, I can assist.\n
Member: Thanks, I think it was denied incorrectly.""")

    def tearDown(self):
        """Clean up the mock directory and file."""
        if os.path.exists(self.test_file_path):
            os.remove(self.test_file_path)
        if os.path.exists(self.test_dir):
            os.rmdir(self.test_dir)

    def test_load_member_conversations(self):
        df = load_member_conversations(self.test_dir)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 1)
        self.assertIn("member_text", df.columns)
        self.assertIn("filename", df.columns)
        self.assertIn("insurance claim", df.loc[0, "member_text"])

    @patch("SCR.process_transcripts.co.chat")
    def test_classify_text_with_chat(self, mock_chat):
        # Mocking Cohere response
        mock_chat.return_value.text = "Sentiment: neutral\nCall Outcome: issue resolved"

        result = classify_text_with_chat("Hello, I need help with my insurance claim.")
        self.assertIn("Sentiment: neutral", result)
        self.assertIn("Call Outcome: issue resolved", result)

if __name__ == '__main__':
    unittest.main()
