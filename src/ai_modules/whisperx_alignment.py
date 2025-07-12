# src/ai_modules/whisperx_alignment.py - Enhanced WhisperX Alignment

import os
import whisperx
import torch
import json
from typing import List, Dict, Any
from difflib import SequenceMatcher  # Fuzzy mapping

# Modular Paths (secure absolute for cross-platform)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SYNC_MAPS_DIR = os.path.join(PROJECT_ROOT, "data", "sync_maps")
os.makedirs(SYNC_MAPS_DIR, exist_ok=True)  # Idempotent creation


async def generate_whisperx_timestamps(
    book_name: str,
    chapter_num: int,
    hebrew_text_verses: List[Dict[str, Any]],
    audio_file_path: str,
) -> List[Dict[str, Any]]:
    """
    Generates word-level timestamps using WhisperX STT alignment.
    Transcribes audio, aligns phonemes, maps to known text with fuzzy matching.

    Args:
        book_name (str): Book name (e.g., "Genesis").
        chapter_num (int): Chapter number.
        hebrew_text_verses (List[Dict[str, Any]]): Verses with 'text' as word list.
        audio_file_path (str): Path to chapter audio MP3.

    Returns:
        List[Dict[str, Any]]: Timestamps with word, start, end, verseIndex, wordIndex.
    """
    # Sanitize filename (secure against injection)
    sanitized_book = "".join(c for c in book_name if c.isalnum() or c in ["_", "-"])
    sync_file_path = os.path.join(
        SYNC_MAPS_DIR, f"{sanitized_book}_ch{chapter_num}_whisperx.json"
    )

    # Cache Check (scalable lazy loading)
    if os.path.exists(sync_file_path):
        print(f"WHISPERX: Loading cached sync map from {sync_file_path}")
        with open(sync_file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    print(f"WHISPERX: Generating new sync map for {book_name} Chapter {chapter_num}...")

    # Device/Compute Type (GPU fallback for elasticity)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"  # Memory-safe

    # Load Whisper Model (large-v3 for accuracy, Hebrew lang)
    model = whisperx.load_model(
        "large-v3", device, compute_type=compute_type, language="he"
    )
    audio = whisperx.load_audio(audio_file_path)

    # Transcribe (batch_size scalable for large audio)
    result = model.transcribe(audio, batch_size=16)

    # Load Alignment Model (Hebrew-specific wav2vec2)
    align_model, metadata = whisperx.load_align_model(
        language_code="he",
        device=device,
        model_name="imvladikon/wav2vec2-large-xlsr-53-hebrew",
    )
    result = whisperx.align(
        result["segments"],
        align_model,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    # Prepare Known Words & Cumulative Indices (for precise mapping)
    known_words = []
    cumulative_lengths = [0]
    for verse_idx, verse in enumerate(hebrew_text_verses):
        known_words.extend(verse["text"])
        cumulative_lengths.append(cumulative_lengths[-1] + len(verse["text"]))

    # Extract Transcribed Words/Timestamps (flatten segments)
    transcribed_data = []
    for segment in result["segments"]:
        for word in segment.get("words", []):
            transcribed_data.append(
                {
                    "word": word["word"],
                    "start": word.get("start", 0.0),
                    "end": word.get("end", 0.0),
                }
            )

    # Fuzzy Mapping (handle transcription variances)
    transcribed_words = [d["word"] for d in transcribed_data]
    matcher = SequenceMatcher(None, transcribed_words, known_words)
    final_timestamps = []
    trans_idx = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("equal", "replace"):  # Allow replaces for close matches
            for k in range(i1, i2):
                known_idx = j1 + (k - i1)
                if known_idx < len(known_words):
                    # Calculate verse/word index
                    verse_index = (
                        next(
                            i
                            for i, cum in enumerate(cumulative_lengths)
                            if cum > known_idx
                        )
                        - 1
                    )
                    word_index = known_idx - cumulative_lengths[verse_index]
                    final_timestamps.append(
                        {
                            "word": known_words[known_idx],
                            "start": transcribed_data[k]["start"],
                            "end": transcribed_data[k]["end"],
                            "verseIndex": verse_index,
                            "wordIndex": word_index,
                        }
                    )
                trans_idx += 1

    # Fallback for Unmatched (append placeholders, log warnings)
    if len(final_timestamps) < len(known_words):
        last_end = final_timestamps[-1]["end"] if final_timestamps else 0.0
        for remaining_idx in range(len(final_timestamps), len(known_words)):
            verse_index = (
                next(
                    i for i, cum in enumerate(cumulative_lengths) if cum > remaining_idx
                )
                - 1
            )
            word_index = remaining_idx - cumulative_lengths[verse_index]
            final_timestamps.append(
                {
                    "word": known_words[remaining_idx],
                    "start": last_end,
                    "end": last_end + 0.5,  # Arbitrary duration; adjust based on avg
                    "verseIndex": verse_index,
                    "wordIndex": word_index,
                }
            )
            last_end += 0.5
            print(
                f"WHISPER WARNING: Placeholder timestamp for unmatched word "
                f"'{known_words[remaining_idx]}'"
            )

    # Save Enriched JSON (secure encoding)
    with open(sync_file_path, "w", encoding="utf-8") as f:
        json.dump(final_timestamps, f, ensure_ascii=False, indent=4)
    print(f"WHISPERX: Processed sync map saved to {sync_file_path}")

    return final_timestamps
