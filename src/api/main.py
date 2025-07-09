# src/api/main.py

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
import torch
import os
import shutil
import uuid # For generating unique filenames
import sys # Added for sys.version

from src.ai_modules.asr_module import load_whisper_model, transcribe_audio

# Define paths for static files and temporary uploads
STATIC_FILES_DIR = os.path.join(os.path.dirname(__file__), "..", "ui")
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "uploads") # Temporary upload directory

# Ensure upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title="Hebrew Tutor AI PoC",
    description="Local Proof of Concept for Hebrew Tutor AI with ASR and RAG.",
    version="0.1.0",
)

# Load Whisper model on startup
@app.on_event("startup")
async def startup_event():
    # Load a small model for PoC. Consider "medium" for better Hebrew accuracy if VRAM allows.
    # For initial quick test, you can use "tiny"
    load_whisper_model(model_name="small") # This will download the model if not present
    print("FastAPI startup: Whisper model loaded.")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """
    Serves the main HTML page for the frontend.
    """
    # For PoC, we'll serve a simple HTML file directly.
    # In a real app, you'd use FastAPI's StaticFiles or a templating engine.
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Hebrew Tutor AI PoC</title>
        <style>
            body { font-family: sans-serif; margin: 20px; }
            h1 { color: #333; }
            button { padding: 10px 20px; margin: 5px; cursor: pointer; }
            #output { margin-top: 20px; padding: 10px; border: 1px solid #ccc; background-color: #f9f9f9; }
            #status { margin-top: 10px; font-size: 0.9em; color: #666; }
        </style>
    </head>
    <body>
        <h1>Hebrew Tutor AI PoC</h1>
        <p>This is a local Proof of Concept for the Hebrew Tutor AI.</p>

        <h2>ASR Test (Speech-to-Text)</h2>
        <button id="recordButton">Record Audio</button>
        <button id="stopButton" disabled>Stop Recording</button>
        <button id="uploadButton" disabled>Transcribe Audio</button>
        <div id="status">Ready.</div>
        <div id="output">Transcription will appear here...</div>

        <script>
            let mediaRecorder;
            let audioChunks = [];
            let audioBlob;
            const recordButton = document.getElementById('recordButton');
            const stopButton = document.getElementById('stopButton');
            const uploadButton = document.getElementById('uploadButton');
            const statusDiv = document.getElementById('status');
            const outputDiv = document.getElementById('output');

            recordButton.onclick = async () => {
                outputDiv.textContent = "Transcription will appear here...";
                audioChunks = [];
                statusDiv.textContent = "Requesting microphone access...";
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    mediaRecorder = new MediaRecorder(stream);
                    mediaRecorder.ondataavailable = event => {
                        audioChunks.push(event.data);
                    };
                    mediaRecorder.onstop = () => {
                        audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                        statusDiv.textContent = "Recording stopped. Ready to transcribe.";
                        uploadButton.disabled = false;
                        stream.getTracks().forEach(track => track.stop()); // Stop microphone access
                    };
                    mediaRecorder.start();
                    statusDiv.textContent = "Recording... Click Stop to finish.";
                    recordButton.disabled = true;
                    stopButton.disabled = false;
                    uploadButton.disabled = true;
                } catch (err) {
                    statusDiv.textContent = `Error accessing microphone: ${err.message}`;
                    console.error('Error accessing microphone:', err);
                }
            };

            stopButton.onclick = () => {
                if (mediaRecorder && mediaRecorder.state === 'recording') {
                    mediaRecorder.stop();
                    recordButton.disabled = false;
                    stopButton.disabled = true;
                }
            };

            uploadButton.onclick = async () => {
                if (!audioBlob) {
                    statusDiv.textContent = "No audio recorded!";
                    return;
                }

                statusDiv.textContent = "Transcribing audio... This may take a moment.";
                uploadButton.disabled = true;

                const formData = new FormData();
                formData.append('audio_file', audioBlob, 'recording.webm');

                try {
                    const response = await fetch('/transcribe/', {
                        method: 'POST',
                        body: formData,
                    });

                    if (response.ok) {
                        const result = await response.json();
                        outputDiv.textContent = `Transcription: ${result.text}`;
                        statusDiv.textContent = "Transcription complete!";
                    } else {
                        const errorData = await response.json();
                        outputDiv.textContent = `Error: ${errorData.detail || response.statusText}`;
                        statusDiv.textContent = "Transcription failed.";
                    }
                } catch (error) {
                    outputDiv.textContent = `Network Error: ${error.message}`;
                    statusDiv.textContent = "Transcription failed due to network error.";
                    console.error('Fetch error:', error);
                } finally {
                    uploadButton.disabled = false;
                }
            };
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/status")
async def get_status():
    """
    Endpoint to check system status, including GPU availability.
    """
    gpu_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count()
    gpu_name = ""
    gpu_memory_total_mb = 0

    if gpu_available:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory_total_bytes = torch.cuda.get_device_properties(0).total_memory
        gpu_memory_total_mb = gpu_memory_total_bytes / (1024**2) # Convert bytes to MB

    status_info = {
        "api_status": "running",
        "python_version": sys.version, # Changed from os.sys.version to sys.version
        "torch_version": torch.__version__,
        "cuda_is_available": gpu_available,
        "cuda_device_count": gpu_count,
        "cuda_device_name": gpu_name,
        "cuda_total_memory_mb": f"{gpu_memory_total_mb:.2f} MB" if gpu_available else "N/A",
        "message": "System check complete."
    }
    return status_info

@app.post("/transcribe/")
async def transcribe_audio_endpoint(audio_file: UploadFile = File(...)):
    """
    Receives an audio file, saves it, transcribes it using Whisper, and returns the text.
    """
    if not audio_file.filename:
        raise HTTPException(status_code=400, detail="No audio file provided.")

    # Generate a unique filename to avoid conflicts
    file_extension = os.path.splitext(audio_file.filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)

    try:
        # Save the uploaded file synchronously for simplicity in PoC.
        # For very large files, consider using aiofiles for async writing.
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
        print(f"Saved uploaded audio to: {file_path}")

        # Transcribe the audio
        transcription_result = await transcribe_audio(file_path)
        return {"text": transcription_result["text"]}
    except Exception as e:
        print(f"Error during transcription: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")
    finally:
        # Clean up the temporary audio file
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Cleaned up temporary audio file: {file_path}")