import pytest
import sys
import os
import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ai_modules.whisperx_alignment import generate_whisperx_timestamps

@pytest.mark.asyncio
class TestWhisperXAlignment:
    async def test_alignment(self):
        audio_path = 'D:\\AI\\Gits\\hebrew_tutor_ai_poc\\data\\tanakh_audio\\hbofGen_01.mp3'
        hebrew_text_verses = [
            {"verse_num": 1, "text": ["בְּרֵאשִׁית", "בָּרָא", "אֱלֹהִים", "אֵת", "הַשָּׁמַיִם", "וְאֵת", "הָאָרֶץ"]}
        ]
        logger.debug("WHISPERX: Starting test with audio_path: %s", audio_path)
        try:
            loop = asyncio.get_event_loop()
            logger.debug("WHISPERX: Event loop acquired: %s", loop)
            result = await asyncio.wait_for(
                generate_whisperx_timestamps("Genesis", 1, hebrew_text_verses, audio_path),
                timeout=300
            )
            logger.debug("WHISPERX: Test completed with result length: %d", len(result))
            assert result is not None
            assert len(result) > 0
            assert all("start" in item and "end" in item for item in result)
        except asyncio.TimeoutError:
            logger.error("WHISPERX: Test timed out after 300 seconds")
            pytest.fail("Test timed out")
        except Exception as e:
            logger.error("WHISPERX: Error occurred - %s", str(e))
            raise

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--timeout=300'])