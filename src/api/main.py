# src/api/main.py

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import HTMLResponse
import torch
import os
import shutil
import uuid
import sys
from typing import Dict
import aiofiles # Import aiofiles for async file operations
import traceback # For printing full tracebacks in errors

from src.ai_modules.asr_module import load_whisper_model, transcribe_audio
from src.ai_modules.rag_module import initialize_rag, get_rag_response

# Define paths for static files and temporary uploads
STATIC_FILES_DIR = os.path.join(os.path.dirname(__file__), "..", "ui")
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "uploads") # Temporary upload directory

# Ensure necessary directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
# Ensure embeddings directory exists for FAISS index (created by rag_module)
os.makedirs(os.path.join(os.path.dirname(__file__), "..", "data", "embeddings"), exist_ok=True)


# Initialize FastAPI app
app = FastAPI(
    title="Hebrew Tutor AI PoC",
    description="Local Proof of Concept for Hebrew Tutor AI with ASR and RAG.",
    version="0.1.0",
)

# Load AI models on startup
@app.on_event("startup")
async def startup_event():
    print("FastAPI startup: Initializing AI models...")
    # Load Whisper model (model_name="small" is a good balance for 8GB GPU)
    load_whisper_model(model_name="small")
    print("FastAPI startup: Whisper model loaded.")

    # Initialize RAG components (LLM, embeddings, FAISS index)
    await initialize_rag()
    print("FastAPI startup: RAG module initialized.")
    print("All AI models are ready.")


@app.get("/", response_class=HTMLResponse)
async def read_root():
    """
    Serves the main HTML page for the frontend.
    """
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
            button, input[type="text"] { padding: 10px 20px; margin: 5px; cursor: pointer; border-radius: 5px; border: 1px solid #ccc; }
            input[type="text"] { width: 300px; padding: 10px; }
            #output, #ragOutput { margin-top: 20px; padding: 15px; border: 1px solid #eee; background-color: #f9f9f9; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            #status, #ragStatus { margin-top: 10px; font-size: 0.9em; color: #666; }
            hr { margin: 30px 0; border: 0; border-top: 1px solid #eee; }
            .section { margin-bottom: 40px; padding: 20px; border: 1px solid #ddd; border-radius: 10px; background-color: #fff; box-shadow: 0 4px 8px rgba(0,0,0,0.05); }
        </style>
    </head>
    <body>
        <h1>Hebrew Tutor AI PoC</h1>
        <p>This is a local Proof of Concept for the Hebrew Tutor AI.</p>

        <div class="section">
            <h2>ASR Test (Speech-to-Text)</h2>
            <button id="recordButton">Record Audio</button>
            <button id="stopButton" disabled>Stop Recording</button>
            <button id="uploadButton" disabled>Transcribe Audio</button>
            <div id="status">Ready.</div>
            <div id="output">Transcription will appear here...</div>
        </div>

        <hr>

        <div class="section">
            <h2>RAG Test (Question Answering)</h2>
            <input type="text" id="queryInput" placeholder="Ask a question about the text...">
            <button id="askButton">Ask LLM</button>
            <div id="ragStatus">Ready.</div>
            <div id="ragOutput">LLM response will appear here...</div>
        </div>


        <script>
            // --- ASR Script ---
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

            // --- RAG Script ---
            const queryInput = document.getElementById('queryInput');
            const askButton = document.getElementById('askButton');
            const ragStatusDiv = document.getElementById('ragStatus');
            const ragOutputDiv = document.getElementById('ragOutput');

            askButton.onclick = async () => {
                const query = queryInput.value.trim();
                if (!query) {
                    ragStatusDiv.textContent = "Please enter a question.";
                    return;
                }

                ragOutputDiv.textContent = "LLM response will appear here...";
                ragStatusDiv.textContent = "Asking LLM... This may take a while.";
                askButton.disabled = true;

                try {
                    const response = await fetch('/ask_llm/', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ query: query })
                    });

                    if (response.ok) {
                        const result = await response.json();
                        ragOutputDiv.textContent = `Response: ${result.response}`;
                        ragStatusDiv.textContent = "LLM response complete!";
                    } else {
                        const errorData = await response.json();
                        ragOutputDiv.textContent = `Error: ${errorData.detail || response.statusText}`;
                        ragStatusDiv.textContent = "LLM query failed.";
                    }
                } catch (error) {
                    ragOutputDiv.textContent = `Network Error: ${error.message}`;
                    ragStatusDiv.textContent = "LLM query failed due to network error.";
                    console.error('Fetch error:', error);
                } finally {
                    askButton.disabled = false;
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
        "python_version": sys.version,
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
        # Save the uploaded file asynchronously for better performance, especially with large files.
        async with aiofiles.open(file_path, "wb") as buffer:
            while True:
                chunk = await audio_file.read(1024 * 1024) # Read in 1MB chunks
                if not chunk:
                    break
                await buffer.write(chunk)
        print(f"Saved uploaded audio to: {file_path}")

        # Transcribe the audio
        transcription_result = await transcribe_audio(file_path)
        return {"text": transcription_result["text"]}
    except Exception as e:
        print(f"Error during transcription: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    finally:
        # Clean up the temporary audio file
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Cleaned up temporary audio file: {file_path}")

@app.post("/ask_llm/")
async def ask_llm_endpoint(query_data: Dict[str, str]):
    """
    Receives a text query, sends it to the RAG module, and returns the LLM's response.
    """
    query = query_data.get("query")
    if not query:
        raise HTTPException(status_code=400, detail="No query provided.")

    try:
        response = await get_rag_response(query)
        return {"response": response}
    except Exception as e:
        print(f"Error during LLM query: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"LLM query failed: {str(e)}")