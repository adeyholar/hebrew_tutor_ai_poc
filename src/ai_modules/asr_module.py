# src/ai_modules/asr_module.py

import whisper
import torch
import os
from typing import Optional
import asyncio

# Global variable to hold the Whisper model
whisper_model = None

def load_whisper_model(model_name: str = "small", device: Optional[str] = None):
    """
    Loads the Whisper ASR model.
    Args:
        model_name (str): The name of the Whisper model to load (e.g., "tiny", "base", "small", "medium").
                          For Hebrew, "small" or "medium" are good starting points.
        device (str): The device to load the model on ("cuda" for GPU, "cpu" for CPU).
                      If None, it will default to "cuda" if available, else "cpu".
    """
    global whisper_model
    if whisper_model is None:
        actual_device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading Whisper model '{model_name}' on device: {actual_device}...")
        whisper_model = whisper.load_model(model_name, device=actual_device)
        print(f"Whisper model '{model_name}' loaded successfully on {actual_device}.")
    return whisper_model

async def transcribe_audio(audio_path: str) -> dict:
    """
    Transcribes an audio file using the loaded Whisper model.
    Args:
        audio_path (str): The file path to the audio to transcribe.
    Returns:
        dict: A dictionary containing the transcription result.
    """
    model = load_whisper_model()
    print(f"Transcribing audio: {audio_path}")
    # Explicitly set language to Hebrew ("he") and task to "transcribe"
    result = model.transcribe(audio_path, fp16=torch.cuda.is_available(), language="he", task="transcribe")
    print(f"Transcription complete: {result['text']}")
    return result

if __name__ == "__main__":
    dummy_audio_path = "test_audio.mp3"
    if not os.path.exists(dummy_audio_path):
        print(f"Please place a test audio file at '{dummy_audio_path}' to test this module directly.")
        print("Skipping direct module test.")
    else:
        print("Running direct transcription test...")
        load_whisper_model(model_name="tiny")
        asyncio.run(transcribe_audio(dummy_audio_path))