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
# from aeneas.globalfunctions import safe_mkdirs # REMOVED: No longer needed
import traceback
from typing import List, Dict, Any

# Define paths for storing generated timestamps
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SYNC_MAPS_DIR = os.path.join(PROJECT_ROOT, "data", "sync_maps")
os.makedirs(SYNC_MAPS_DIR, exist_ok=True) # REPLACED: Using standard os.makedirs

async def generate_timestamps(book_name: str, chapter_num: int, hebrew_text_verses: List[Dict[str, Any]], audio_file_path: str) -> List[Dict[str, Any]]:
    """
    Generates word-level timestamps for a given Hebrew chapter audio and text using Aeneas.
    Caches results to avoid re-processing.

    Args:
        book_name (str): The full name of the book (e.g., "Genesis").
        chapter_num (int): The chapter number.
        hebrew_text_verses (List[Dict[str, Any]]): List of verse objects, where each verse is
                                                  {"verse_num": int, "text": [list of words]}.
                                                  This comes directly from the get_chapter_text endpoint.
        audio_file_path (str): The full path to the chapter's audio MP3 file.

    Returns:
        List[Dict[str, Any]]: A list of word objects, each with 'word', 'start', 'end',
                              'verseIndex', 'wordIndex'.
    """
    # Secure filename (sanitize book_name to avoid injection)
    sanitized_book = "".join(c for c in book_name if c.isalnum() or c in [' ', '_', '-']).replace(" ", "_")
    sync_file_basename = f"{sanitized_book}_ch{chapter_num}.json"
    sync_file_path = os.path.join(SYNC_MAPS_DIR, sync_file_basename)

    # Check if timestamps already exist in cache
    if os.path.exists(sync_file_path):
        print(f"ALIGNMENT: Loading cached sync map for {book_name} Chapter {chapter_num} from {sync_file_path}")
        with open(sync_file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    # If not cached, prepare for generation
    print(f"ALIGNMENT: Generating new sync map for {book_name} Chapter {chapter_num}...")

    # Prepare text for Aeneas: a single string with words separated by spaces.
    # We also need to map Aeneas's output fragments back to our original word structure
    # including verse and word indices for frontend highlighting.
    
    # Flatten the verses into a single list of words, retaining their original indices
    all_words_with_original_indices = []
    for verse_idx, verse_data in enumerate(hebrew_text_verses):
        for word_idx, word in enumerate(verse_data['text']):
            all_words_with_original_indices.append({
                "word": word,
                "verseIndex": verse_idx,
                "wordIndex": word_idx,
                "verse_num": verse_data['verse_num'] # Also keep verse_num for potential future use
            })
    
    text_content_for_aeneas = " ".join([item["word"] for item in all_words_with_original_indices])

    temp_text_path = None # Initialize to None for finally block
    try:
        # Create a temporary text file for Aeneas input
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt') as temp_text_file:
            temp_text_file.write(text_content_for_aeneas)
            temp_text_path = temp_text_file.name

        # Aeneas configuration: Hebrew language, MFCC for OSR, word-level sync map, JSON output
        # 'is_text_type=plain' is crucial for text files
        config_string = (
            "task_language=heb|"
            "osr=mfcc|" # Optimal Speech Recognition using Mel-frequency cepstral coefficients
            "sync_map_level=word|" # Get word-level timestamps
            "is_text_type=plain|" # We are providing plain text file
            "os_task_file_format=json" # Output format is JSON
        )

        # Create Aeneas Task
        task = Task(config_string=config_string)
        task.audio_file_path_absolute = audio_file_path
        task.text_file_path_absolute = temp_text_path
        task.sync_map_file_path_absolute = sync_file_path # Aeneas will write directly to our cache

        # Execute alignment in a separate thread to prevent blocking FastAPI's event loop
        await asyncio.to_thread(ExecuteTask(task).execute)
        
        # Aeneas will have written the sync map to sync_file_path
        # Now, we load it and map its fragments back to our original word structure
        with open(sync_file_path, 'r', encoding='utf-8') as f:
            aeneas_raw_sync_map = json.load(f)

        final_timestamps = []
        current_word_idx_flat = 0

        # Iterate through Aeneas fragments and map them to our original words
        for fragment in aeneas_raw_sync_map.get("fragments", []):
            # Aeneas fragment text might be multiple words or normalized.
            # We need to consume words from our original list based on the fragment's content.
            # This is a heuristic and might need fine-tuning for edge cases.
            fragment_text_normalized = fragment["lines"][0].strip() # Get the text Aeneas aligned to
            
            # Simple heuristic: try to match words from our original list
            # A more robust solution might involve fuzzy matching or more complex tokenization
            # but for now, we'll assume a sequential match.
            
            # We need to ensure we don't go out of bounds of all_words_with_original_indices
            # and that we correctly associate the Aeneas fragment's times with the correct original word.
            
            # The most reliable way is to iterate through Aeneas fragments and
            # assign the fragment's start/end times to the corresponding word(s)
            # from our original word list.
            
            # If Aeneas gives word-level fragments, then fragment["lines"][0] should be a single word.
            # If it groups words, we'd need to distribute the time.
            # For 'sync_map_level=word', Aeneas tries to give one fragment per word.
            
            # Let's assume Aeneas is returning one word per fragment for simplicity,
            # and if not, the original_word_data will still be correctly indexed.
            
            if current_word_idx_flat < len(all_words_with_original_indices):
                original_word_data = all_words_with_original_indices[current_word_idx_flat]
                
                # Store the Aeneas fragment's times with our original word's metadata
                final_timestamps.append({
                    "word": original_word_data["word"],
                    "start": float(fragment["begin"]),
                    "end": float(fragment["end"]),
                    "verseIndex": original_word_data["verseIndex"],
                    "wordIndex": original_word_data["wordIndex"]
                })
                current_word_idx_flat += 1
            else:
                print(f"ALIGNMENT WARNING: Aeneas returned more fragments than original words. Fragment: '{fragment['lines'][0]}'")
                # This should ideally not happen if sync_map_level=word is effective
                break 

        # Handle cases where Aeneas might have missed some words (less common with good audio/text)
        # Fill in remaining words with placeholder times if Aeneas returned fewer fragments than words
        while current_word_idx_flat < len(all_words_with_original_indices):
            original_word_data = all_words_with_original_indices[current_word_idx_flat]
            last_end_time = final_timestamps[-1]["end"] if final_timestamps else 0.0
            final_timestamps.append({
                "word": original_word_data["word"],
                "start": last_end_time,
                "end": last_end_time + 0.1, # Assign a small placeholder duration
                "verseIndex": original_word_data["verseIndex"],
                "wordIndex": original_word_data["wordIndex"]
            })
            current_word_idx_flat += 1
            print(f"ALIGNMENT WARNING: Filled missing timestamp for word: '{original_word_data['word']}'")

        # Save the final processed timestamps (which now include verseIndex and wordIndex)
        # This overwrites the raw Aeneas output with our enriched data
        with open(sync_file_path, 'w', encoding='utf-8') as f:
            json.dump(final_timestamps, f, ensure_ascii=False, indent=4)
        print(f"ALIGNMENT: Processed sync map saved to {sync_file_path}")

        return final_timestamps

    except Exception as e:
        print(f"ALIGNMENT ERROR: Failed to generate sync map for {book_name} Chapter {chapter_num}: {e}")
        traceback.print_exc()
        # It's better to raise the exception so FastAPI can return a 500
        raise ValueError(f"Failed to generate sync map: {str(e)}")

    finally:
        # Clean up the temporary text file
        if temp_text_path and os.path.exists(temp_text_path):
            os.remove(temp_text_path)
            print(f"ALIGNMENT: Cleaned up temporary text file: {temp_text_path}")

