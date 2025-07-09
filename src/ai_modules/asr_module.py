# src/ai_modules/asr_module.py

import whisper
import torch
import os
from typing import Optional # Added for type hinting
import asyncio # Added for asyncio.run() in __main__ block

# Global variable to hold the Whisper model
# We load it once when the module is imported to avoid reloading on every request
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
        # Determine the actual device to use
        actual_device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading Whisper model '{model_name}' on device: {actual_device}...")
        # Load the model, downloading it if not already present
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
    model = load_whisper_model() # Ensure model is loaded
    print(f"Transcribing audio: {audio_path}")
    # fp16=True uses half-precision floating point for faster inference on GPU
    result = model.transcribe(audio_path, fp16=torch.cuda.is_available())
    print(f"Transcription complete: {result['text']}")
    return result

# Example of how you might use it (for testing within this file)
if __name__ == "__main__":
    # This part will only run if you execute asr_module.py directly
    # For a PoC, you might download a small audio file to test
    # Example: Create a dummy audio file path for testing
    dummy_audio_path = "test_audio.mp3" # You'd replace this with a real audio file
    if not os.path.exists(dummy_audio_path):
        print(f"Please place a test audio file at '{dummy_audio_path}' to test this module directly.")
        print("Skipping direct module test.")
    else:
        print("Running direct transcription test...")
        # Ensure the model is loaded before calling transcribe
        load_whisper_model(model_name="tiny") # Use "tiny" for quick test download
        # Use asyncio.run() to execute the async function
        asyncio.run(transcribe_audio(dummy_audio_path))