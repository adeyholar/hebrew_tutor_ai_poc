import unittest
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ai_modules.whisperx_alignment import generate_whisperx_timestamps

class TestWhisperXAlignment(unittest.TestCase):
    async def test_alignment(self):
        audio_path = 'D:\\AI\\Gits\\hebrew_tutor_ai_poc\\data\\tanakh_audio\\hbofGen_01.mp3'
        hebrew_text_verses = [
            {"verse_num": 1, "text": ["בְּרֵאשִׁית", "בָּרָא", "אֱלֹהִים", "אֵת", "הַשָּׁמַיִם", "וְאֵת", "הָאָרֶץ"]}
        ]
        result = await generate_whisperx_timestamps("Genesis", 1, hebrew_text_verses, audio_path)
        self.assertIsNotNone(result)  # Check that alignment output exists
        self.assertTrue(len(result) > 0)  # Ensure non-empty alignment
        self.assertTrue(all("start" in item and "end" in item for item in result))  # Verify timestamp fields

if __name__ == '__main__':
    unittest.main()