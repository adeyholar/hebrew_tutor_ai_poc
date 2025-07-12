# src/ai_modules/alignment_module.py

import os
import json
import tempfile
import asyncio
from aeneas.executetask import ExecuteTask
from aeneas.task import Task
from aeneas.language import Language
from aeneas.syncmap import SyncMapFormat
from aeneas.textfile import TextFileFormat
import traceback
from typing import List, Dict, Any
import subprocess  # Modular validation for voices

# Modular Paths (secure absolute, scalable for multi-project)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SYNC_MAPS_DIR = os.path.join(PROJECT_ROOT, "data", "sync_maps")
os.makedirs(SYNC_MAPS_DIR, exist_ok=True)  # Idempotent creation

async def generate_timestamps(book_name: str, chapter_num: int, hebrew_text_verses: List[Dict[str, Any]], audio_file_path: str) -> List[Dict[str, Any]]:
    """
    Generates word-level timestamps using Aeneas with fallback to default eSpeak Hebrew.
    Caches results for scalability.

    Args:
        book_name (str): Book name (e.g., "Genesis").
        chapter_num (int): Chapter number.
        hebrew_text_verses (List[Dict[str, Any]]): Verses from get_chapter_text.
        audio_file_path (str): Audio MP3 path.

    Returns:
        List[Dict[str, Any]]: Word timestamps with metadata.
    """
    # Secure filename (sanitize to prevent injection, scalable hashing if needed)
    sanitized_book = "".join(c for c in book_name if c.isalnum() or c in [' ', '_', '-']).replace(" ", "_")
    sync_file_basename = f"{sanitized_book}_ch{chapter_num}.json"
    sync_file_path = os.path.join(SYNC_MAPS_DIR, sync_file_basename)

    # Cache Check (scalable lazy load, secure existence check)
    if os.path.exists(sync_file_path):
        print(f"ALIGNMENT: Loading cached sync map for {book_name} Chapter {chapter_num} from {sync_file_path}")
        with open(sync_file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    # Generation Prep (modular logging)
    print(f"ALIGNMENT: Generating new sync map for {book_name} Chapter {chapter_num}...")

    # Secure PATH for eSpeak-NG (append only if missing, scalable env var)
    espeak_ng_path = r"C:\Program Files\eSpeak NG"
    if espeak_ng_path not in os.environ.get('PATH', ''):
        os.environ['PATH'] = os.environ.get('PATH', '') + ';' + espeak_ng_path
        print(f"ALIGNMENT: Added eSpeak-NG to PATH: {espeak_ng_path}")

    # Full path for eSpeak-NG binary (secure, fixes [WinError 2])
    espeak_bin = r"C:\Program Files\eSpeak NG\bin\espeak-ng.exe"

    # Validate Hebrew Voice (secure subprocess with full path, fallback warning)
    try:
        voices_output = subprocess.run([espeak_bin, '--voices'], capture_output=True, text=True, check=True).stdout
        if 'he' not in voices_output.lower():
            raise ValueError("Hebrew ('he') voice not found in eSpeak-NG.")
        print("ALIGNMENT: Hebrew voice validated in eSpeak-NG.")
    except Exception as e:
        print(f"ALIGNMENT WARNING: Hebrew voice check failed: {e}. Using fallback defaults.")

    # Prepare Text (flatten with indices for frontend mapping, scalable for large chapters)
    all_words_with_original_indices = []
    for verse_idx, verse_data in enumerate(hebrew_text_verses):
        for word_idx, word in enumerate(verse_data['text']):
            all_words_with_original_indices.append({
                "word": word,
                "verseIndex": verse_idx,
                "wordIndex": word_idx,
                "verse_num": verse_data['verse_num']
            })
    
    text_content_for_aeneas = " ".join([item["word"] for item in all_words_with_original_indices])

    temp_text_path = None
    try:
        # Temp Text File (secure tempfile, auto-clean)
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt') as temp_text_file:
            temp_text_file.write(text_content_for_aeneas)
            temp_text_path = temp_text_file.name

        # Aeneas Config (use 'heb' for Hebrew—Aeneas ISO 639-3 code)
        config_string = (
            "tts=espeak-ng|"
            "task_language=heb|"
            "osr=mfcc|"
            "is_text_type=plain|"
            "os_task_file_format=json|"
            "os_task_file_level=3"
        )

        # Create Task (modular, scalable for custom configs)
        task = Task(config_string=config_string)
        task.audio_file_path_absolute = audio_file_path
        task.text_file_path_absolute = temp_text_path
        task.sync_map_file_path_absolute = sync_file_path

        # Execute (async thread for backend elasticity)
        await asyncio.to_thread(ExecuteTask(task).execute)

        # Load & Map Fragments (modular post-process, scalable heuristic for grouping)
        with open(sync_file_path, 'r', encoding='utf-8') as f:
            aeneas_raw_sync_map = json.load(f)

        final_timestamps = []
        current_word_idx_flat = 0

        for fragment in aeneas_raw_sync_map.get("fragments", []):
            if current_word_idx_flat < len(all_words_with_original_indices):
                original_word_data = all_words_with_original_indices[current_word_idx_flat]
                final_timestamps.append({
                    "word": original_word_data["word"],
                    "start": float(fragment["begin"]),
                    "end": float(fragment["end"]),
                    "verseIndex": original_word_data["verseIndex"],
                    "wordIndex": original_word_data["wordIndex"]
                })
                current_word_idx_flat += 1
            else:
                print(f"ALIGNMENT WARNING: Extra fragment: '{fragment['lines'][0]}'")
                break

        # Fill Missing (scalable placeholder for incomplete alignments)
        while current_word_idx_flat < len(all_words_with_original_indices):
            original_word_data = all_words_with_original_indices[current_word_idx_flat]
            last_end_time = final_timestamps[-1]["end"] if final_timestamps else 0.0
            final_timestamps.append({
                "word": original_word_data["word"],
                "start": last_end_time,
                "end": last_end_time + 0.1,
                "verseIndex": original_word_data["verseIndex"],
                "wordIndex": original_word_data["wordIndex"]
            })
            current_word_idx_flat += 1
            print(f"ALIGNMENT WARNING: Filled missing timestamp for word: '{original_word_data['word']}'")

        # Save Enriched (secure JSON write, scalable indent for readability)
        with open(sync_file_path, 'w', encoding='utf-8') as f:
            json.dump(final_timestamps, f, ensure_ascii=False, indent=4)
        print(f"ALIGNMENT: Processed sync map saved to {sync_file_path}")

        return final_timestamps

    except Exception as e:
        print(f"ALIGNMENT ERROR: Failed to generate sync map for {book_name} Chapter {chapter_num}: {e}")
        traceback.print_exc()
        raise ValueError(f"Failed to generate sync map: {str(e)}")

    finally:
        # Clean Temp (secure removal)
        if temp_text_path and os.path.exists(temp_text_path):
            os.remove(temp_text_path)
            print(f"ALIGNMENT: Cleaned up temporary text file: {temp_text_path}")