# src/api/main.py

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import torch
import os
import shutil
import uuid
import sys
from typing import Dict, List, Any
import aiofiles
import traceback
import re

from src.ai_modules.asr_module import load_whisper_model, transcribe_audio
from src.ai_modules.rag_module import initialize_rag, get_rag_response, get_loaded_documents 
from src.ai_modules.alignment_module import generate_timestamps # NEW IMPORT

# Define paths for static files and temporary uploads
STATIC_FILES_DIR = os.path.join(os.path.dirname(__file__), "..", "ui")
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "uploads")

# Use an absolute path for AUDIO_DIR to ensure consistency
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
AUDIO_DIR = os.path.join(PROJECT_ROOT, "data", "tanakh_audio")

# Ensure necessary directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(__file__), "..", "data", "embeddings"), exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)

# --- DEBUGGING PRINT STATEMENTS (KEEP THESE FOR NOW) ---
print(f"DEBUG: Current file path: {__file__}")
print(f"DEBUG: Directory of current file: {os.path.dirname(__file__)}")
print(f"DEBUG: Calculated PROJECT_ROOT: {PROJECT_ROOT}")
print(f"DEBUG: Calculated AUDIO_DIR for StaticFiles: {AUDIO_DIR}")
print(f"DEBUG: Does AUDIO_DIR exist? {os.path.exists(AUDIO_DIR)}")
print(f"DEBUG: Contents of AUDIO_DIR:")
try:
    for item in os.listdir(AUDIO_DIR):
        print(f"  - {item}")
except FileNotFoundError:
    print(f"  - AUDIO_DIR not found: {AUDIO_DIR}")
# --- END DEBUGGING PRINT STATEMENTS ---


# Initialize FastAPI app
app = FastAPI(
    title="Hebrew Tutor AI PoC",
    description="Local Proof of Concept for Hebrew Tutor AI with ASR and RAG.",
    version="0.1.0",
)

# Mount the directory containing the Tanakh audio files
app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")


# Load AI models on startup
@app.on_event("startup")
async def startup_event():
    print("FastAPI startup: Initializing AI models...")
    load_whisper_model(model_name="small")
    print("FastAPI startup: Whisper model loaded.")

    await initialize_rag() 
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
            button, input[type="text"], select { padding: 10px 20px; margin: 5px; cursor: pointer; border-radius: 5px; border: 1px solid #ccc; }
            input[type="text"] { width: 300px; padding: 10px; }
            #output, #ragOutput, #hebrewTextDisplay { margin-top: 20px; padding: 15px; border: 1px solid #eee; background-color: #f9f9f9; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            #status, #ragStatus, #audioPlaybackStatus, #textDisplayStatus { margin-top: 10px; font-size: 0.9em; color: #666; }
            hr { margin: 30px 0; border: 0; border-top: 1px solid #eee; }
            .section { margin-bottom: 40px; padding: 20px; border: 1px solid #ddd; border-radius: 10px; background-color: #fff; box-shadow: 0 4px 8px rgba(0,0,0,0.05); }
            .hebrew-verse { 
                direction: rtl; /* Right-to-left for Hebrew */
                text-align: right; 
                font-size: 1.2em; 
                margin-bottom: 10px; 
                line-height: 1.6;
                border-bottom: 1px dotted #ccc;
                padding-bottom: 5px;
            }
            .hebrew-verse span {
                font-weight: bold;
                margin-left: 10px;
                color: #555;
            }
            /* Highlighting style */
            .highlight {
                background-color: #ffeb3b; /* A yellow highlight */
                border-radius: 3px;
                padding: 2px 0;
            }
            /* Loading indicator for alignment */
            #alignmentStatus {
                margin-top: 10px;
                font-size: 0.9em;
                color: #007bff;
                font-weight: bold;
                display: none; /* Hidden by default */
            }
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

        <hr>

        <div class="section">
            <h2>Read and Follow Along (Text & Audio)</h2>
            <div>
                <label for="bookSelect">Select Book:</label>
                <select id="bookSelect">
                    <!-- Options will be populated dynamically -->
                </select>
                <label for="chapterSelect" style="margin-left: 15px;">Select Chapter:</label>
                <select id="chapterSelect" disabled>
                    <!-- Options will be populated dynamically -->
                </select>
                <button id="loadChapterTextButton">Load Chapter Text</button>
                <button id="playChapterAudioButton" disabled>Play Chapter Audio</button> 
            </div>
            <div style="margin-top: 10px;">
                <label for="playbackSpeedSelect">Playback Speed:</label>
                <select id="playbackSpeedSelect">
                    <option value="0.25">25%</option>
                    <option value="0.5">50%</option>
                    <option value="0.75">75%</option>
                    <option value="1.0" selected>100%</option>
                    <option value="1.25">125%</option>
                    <option value="1.5">150%</option>
                    <option value="2.0">200%</option>
                </select>
            </div>
            <div id="alignmentStatus">Generating timestamps...</div> <!-- NEW: Alignment Status -->
            <div id="audioPlaybackStatus" style="margin-top: 10px;">Ready.</div>
            <audio id="audioPlayer" controls style="width: 100%; margin-top: 10px;"></audio>
            
            <div id="textDisplayStatus" style="margin-top: 20px;">Load a book and chapter to see text.</div>
            <div id="hebrewTextDisplay">
                <!-- Hebrew text will be displayed here -->
            </div>
        </div>


        <script>
            // --- Wrap all JavaScript in DOMContentLoaded ---
            document.addEventListener('DOMContentLoaded', (event) => {
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
                        stopButton.disabled = true; // Keep disabled until recording starts again
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

                    statusDiv.textContent = "Transcribing audio... This may may take a moment.";
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
                        ragStatusDiv.textContent = "LLM query failed due to network error.";
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
                            outputDiv.textContent = `Error: ${errorData.detail || response.statusText}`;
                            ragStatusDiv.textContent = "LLM query failed.";
                        }
                    } catch (error) {
                        outputDiv.textContent = `Network Error: ${error.message}`;
                        ragStatusDiv.textContent = "LLM query failed due to network error.";
                        console.error('Fetch error:', error);
                    } finally {
                        askButton.disabled = false;
                    }
                };

                // --- Audio Playback & Text Display Script ---
                const bookSelect = document.getElementById('bookSelect');
                const chapterSelect = document.getElementById('chapterSelect');
                const loadChapterTextButton = document.getElementById('loadChapterTextButton');
                const playChapterAudioButton = document.getElementById('playChapterAudioButton'); 
                const audioPlaybackStatus = document.getElementById('audioPlaybackStatus');
                const audioPlayer = document.getElementById('audioPlayer');
                const playbackSpeedSelect = document.getElementById('playbackSpeedSelect');
                const hebrewTextDisplay = document.getElementById('hebrewTextDisplay');
                const textDisplayStatus = document.getElementById('textDisplayStatus');
                const alignmentStatusDiv = document.getElementById('alignmentStatus'); // NEW

                // Store loaded verses and their timestamps globally for highlighting logic
                let loadedVerses = []; // Raw verse data from /get_chapter_text
                let wordTimestamps = []; // Array of {word: "...", start: X, end: Y, verseIndex: V, wordIndex: W} objects
                let currentHighlightedWordId = null;
                
                // Removed audioOffsetSeconds and totalWordsInChapter as they are replaced by Aeneas timestamps

                // --- DEBUGGING: Check if elements are found ---
                console.log("DEBUG: bookSelect element found:", bookSelect);
                console.log("DEBUG: chapterSelect element found:", chapterSelect);
                console.log("DEBUG: loadChapterTextButton element found:", loadChapterTextButton);
                console.log("DEBUG: playChapterAudioButton element found:", playChapterAudioButton);


                // List of books for frontend display and lookup.
                // 'id' must EXACTLY match the full capitalized book name produced by rag_module's book_abbr_to_full_name_map.
                // 'name' is for display in the dropdown.
                // 'chapters' are approximate max chapters for dropdown population.
                const books = [
                    { id: '1 Chronicles', name: '1 Chronicles', chapters: 29 },
                    { id: '2 Chronicles', name: '2 Chronicles', chapters: 36 },
                    { id: 'Amos', name: 'Amos', chapters: 9 }, 
                    { id: 'Daniel', name: 'Daniel', chapters: 12 },
                    { id: 'Deuteronomy', name: 'Deuteronomy', chapters: 34 },
                    { id: 'Esther', name: 'Esther', chapters: 10 },
                    { id: 'Exodus', name: 'Exodus', chapters: 40 },
                    { id: 'Ezekiel', name: 'Ezekiel', chapters: 48 },
                    { id: 'Ezra', name: 'Ezra', chapters: 10 },
                    { id: 'Genesis', name: 'Genesis', chapters: 50 },
                    { id: 'Habbakuk', name: 'Habbakuk', chapters: 3 },
                    { id: 'Haggai', name: 'Haggai', chapters: 2 },
                    { id: 'Hosea', name: 'Hosea', chapters: 14 },
                    { id: 'Isaiah', name: 'Isaiah', chapters: 66 },
                    { id: 'Jeremiah', name: 'Jeremiah', chapters: 52 },
                    { id: 'Job', name: 'Job', chapters: 42 },
                    { id: 'Joel', name: 'Joel', chapters: 3 },
                    { id: 'Jonah', name: 'Jonah', chapters: 4 },
                    { id: 'Joshua', name: 'Joshua', chapters: 24 },
                    { id: 'Judges', name: 'Judges', chapters: 21 },
                    { id: '1 Kings', name: '1 Kings', chapters: 22 },
                    { id: '2 Kings', name: '2 Kings', chapters: 25 },
                    { id: 'Koheleth (Ecclesiastes)', name: 'Koheleth (Ecclesiastes)', chapters: 12 },
                    { id: 'Lamentations', name: 'Lamentations', chapters: 5 },
                    { id: 'Leviticus', name: 'Leviticus', chapters: 27 },
                    { id: 'Malachi', name: 'Malachi', chapters: 4 },
                    { id: 'Micah', name: 'Micah', chapters: 7 },
                    { id: 'Nahum', name: 'Nahum', chapters: 3 },
                    { id: 'Nehemiah', name: 'Nehemiah', chapters: 13 },
                    { id: 'Numbers', name: 'Numbers', chapters: 36 },
                    { id: 'Obadiah', name: 'Obadiah', chapters: 1 },
                    { id: 'Proverbs', name: 'Proverbs', chapters: 31 },
                    { id: 'Psalms', name: 'Psalms', chapters: 150 },
                    { id: 'Ruth', name: 'Ruth', chapters: 4 },
                    { id: '1 Samuel', name: '1 Samuel', chapters: 31 },
                    { id: '2 Samuel', name: '2 Samuel', chapters: 24 },
                    { id: 'Song of Songs', name: 'Song of Songs', chapters: 8 },
                    { id: 'Zechariah', name: 'Zechariah', chapters: 14 },
                    { id: 'Zephaniah', name: 'Zephaniah', chapters: 3 },
                ];

                // --- DEBUGGING: Check content of books array ---
                console.log("DEBUG: Books array content:", books);
                console.log("DEBUG: Books array length:", books.length);


                // Populate the book select dropdown
                if (bookSelect) {
                    books.forEach(book => {
                        const option = document.createElement('option');
                        option.value = book.id; 
                        option.textContent = book.name;
                        bookSelect.appendChild(option);
                    });
                } else {
                    console.error("ERROR: 'bookSelect' element not found. Cannot populate book dropdown.");
                }


                // Function to populate chapters based on selected book
                bookSelect.onchange = () => {
                    const selectedBookId = bookSelect.value;
                    const selectedBook = books.find(b => b.id === selectedBookId);
                    
                    chapterSelect.innerHTML = '<option value="">Select Chapter</option>'; // Clear existing options
                    chapterSelect.disabled = true;
                    loadChapterTextButton.disabled = true;
                    playChapterAudioButton.disabled = true; // Disable audio button on book change
                    hebrewTextDisplay.innerHTML = ''; // Clear text display
                    textDisplayStatus.textContent = "Select a chapter.";
                    loadedVerses = []; // Clear loaded verses on book change
                    wordTimestamps = []; // Clear timestamps
                    resetHighlight(); // Clear any existing highlight
                    alignmentStatusDiv.style.display = 'none'; // Hide alignment status

                    if (selectedBook) {
                        for (let i = 1; i <= selectedBook.chapters; i++) {
                            const option = document.createElement('option');
                            option.value = i;
                            option.textContent = `Chapter ${i}`;
                            chapterSelect.appendChild(option);
                        }
                        chapterSelect.disabled = false;
                    }
                };

                chapterSelect.onchange = () => {
                    loadChapterTextButton.disabled = !chapterSelect.value;
                    playChapterAudioButton.disabled = !chapterSelect.value; // Enable/disable audio button based on chapter selection
                    hebrewTextDisplay.innerHTML = ''; // Clear text display on chapter change
                    textDisplayStatus.textContent = "Click 'Load Chapter Text'.";
                    loadedVerses = []; // Clear loaded verses on chapter change
                    wordTimestamps = []; // Clear timestamps
                    resetHighlight(); // Clear any existing highlight
                    alignmentStatusDiv.style.display = 'none'; // Hide alignment status
                };

                loadChapterTextButton.onclick = async () => {
                    const selectedBookId = bookSelect.value;
                    const selectedChapter = chapterSelect.value;

                    if (!selectedBookId || !selectedChapter) {
                        textDisplayStatus.textContent = "Please select both a book and a chapter.";
                        return;
                    }

                    hebrewTextDisplay.innerHTML = '';
                    textDisplayStatus.textContent = `Loading ${selectedBookId} Chapter ${selectedChapter}...`;
                    loadedVerses = []; // Reset loaded verses before fetching
                    wordTimestamps = []; // Reset timestamps
                    resetHighlight(); // Clear any existing highlight
                    playChapterAudioButton.disabled = true; // Disable audio button until text and timestamps are ready
                    alignmentStatusDiv.style.display = 'block'; // Show alignment status
                    alignmentStatusDiv.textContent = "Loading text and preparing for audio alignment...";


                    try {
                        // 1. Fetch Chapter Text
                        const textResponse = await fetch(`/get_chapter_text/${selectedBookId}/${selectedChapter}`);
                        if (!textResponse.ok) {
                            const errorData = await textResponse.json();
                            throw new Error(`Error loading text: ${errorData.detail || textResponse.statusText}`);
                        }
                        const textResult = await textResponse.json();

                        if (textResult.verses && textResult.verses.length > 0) {
                            loadedVerses = textResult.verses; // Store loaded verses
                            hebrewTextDisplay.innerHTML = ''; // Clear previous content
                            let totalWordsCount = 0; // Count words for potential fallback or info
                            loadedVerses.forEach((verse, verseIndex) => {
                                const verseDiv = document.createElement('div');
                                verseDiv.className = 'hebrew-verse';
                                let verseHtml = `<span>${verse.verse_num}.</span> `;
                                
                                // Wrap each word in a span with a unique ID for highlighting
                                verse.text.forEach((word, wordIndex) => {
                                    const wordId = `word-${verseIndex}-${wordIndex}`;
                                    verseHtml += `<span id="${wordId}">${word}</span> `;
                                    totalWordsCount++;
                                });
                                verseDiv.innerHTML = verseHtml.trim(); // Trim trailing space
                                hebrewTextDisplay.appendChild(verseDiv);
                            });
                            textDisplayStatus.textContent = `Loaded ${loadedVerses.length} verses (${totalWordsCount} words) for ${selectedBookId} Chapter ${selectedChapter}.`;
                            
                            // 2. Fetch Timestamps for Alignment
                            alignmentStatusDiv.textContent = "Generating/Loading word timestamps (this may take a moment for new chapters)...";
                            const timestampResponse = await fetch(`/get_chapter_timestamps/${selectedBookId}/${selectedChapter}`);
                            if (!timestampResponse.ok) {
                                const errorData = await timestampResponse.json();
                                throw new Error(`Error fetching timestamps: ${errorData.detail || timestampResponse.statusText}`);
                            }
                            wordTimestamps = await timestampResponse.json();
                            console.log("DEBUG: Fetched wordTimestamps:", wordTimestamps);

                            if (wordTimestamps.length > 0) {
                                alignmentStatusDiv.textContent = `Timestamps loaded for ${wordTimestamps.length} words.`;
                                playChapterAudioButton.disabled = false; // Enable audio button once text and timestamps are ready
                            } else {
                                alignmentStatusDiv.textContent = "No timestamps found/generated. Audio highlighting will not work.";
                                playChapterAudioButton.disabled = false; // Still allow playing audio without highlight
                            }

                        } else {
                            hebrewTextDisplay.innerHTML = '<p>No verses found for this chapter.</p>';
                            textDisplayStatus.textContent = `No text found for ${selectedBookId} Chapter ${selectedChapter}.`;
                            playChapterAudioButton.disabled = true; 
                            alignmentStatusDiv.textContent = "No text to align.";
                        }
                    } catch (error) {
                        textDisplayStatus.textContent = `Error: ${error.message}`;
                        alignmentStatusDiv.textContent = `Alignment Error: ${error.message}`;
                        console.error('Loading/Alignment error:', error);
                        playChapterAudioButton.disabled = true; 
                    } finally {
                        // Ensure alignment status is hidden or shows final state
                        // alignmentStatusDiv.style.display = 'none'; // Could hide it completely after success/failure
                    }
                };

                // NEW LOGIC FOR PLAYING CHAPTER AUDIO
                playChapterAudioButton.onclick = () => {
                    const selectedBookId = bookSelect.value;
                    const selectedChapter = chapterSelect.value;

                    if (!selectedBookId || !selectedChapter) {
                        audioPlaybackStatus.textContent = "Please select both a book and a chapter.";
                        return;
                    }
                    if (wordTimestamps.length === 0) {
                        audioPlaybackStatus.textContent = "Warning: No timestamps available. Playing audio without highlighting.";
                    }

                    // Mapping from frontend ID (full name) to audio filename prefix (hbof[BookAbbr])
                    const audioBookPrefixMap = {
                        '1 Chronicles': 'hbof1Ch',
                        '2 Chronicles': 'hbof2Ch',
                        'Amos': 'hbofAmo',
                        'Daniel': 'hbofDan',
                        'Deuteronomy': 'hbofDeu',
                        'Esther': 'hbofEst',
                        'Exodus': 'hbofExo',
                        'Ezekiel': 'hbofEzk',
                        'Ezra': 'hbofEzr',
                        'Genesis': 'hbofGen',
                        'Habbakuk': 'hbofHab',
                        'Haggai': 'hbofHag',
                        'Hosea': 'hbofHos',
                        'Isaiah': 'hbofIsa',
                        'Jeremiah': 'hbofJer',
                        'Job': 'hbofJob',
                        'Joel': 'hbofJol',
                        'Jonah': 'hbofJon',
                        'Joshua': 'hbofJos',
                        'Judges': 'hbofJdg',
                        '1 Kings': 'hbof1Ki',
                        '2 Kings': 'hbof2Ki',
                        'Koheleth (Ecclesiastes)': 'hbofEcc',
                        'Lamentations': 'hbofLam',
                        'Leviticus': 'hbofLev',
                        'Malachi': 'hbofMal',
                        'Micah': 'hbofMic',
                        'Nahum': 'hbofNam', 
                        'Nehemiah': 'hbofNeh',
                        'Numbers': 'hbofNum',
                        'Obadiah': 'hbofOba', 
                        'Proverbs': 'hbofPro',
                        'Psalms': 'hbofPsa', 
                        'Ruth': 'hbofRut',
                        '1 Samuel': 'hbof1Sa',
                        '2 Samuel': 'hbof2Sa',
                        'Song of Songs': 'hbofSng',
                        'Zechariah': 'hbofZec',
                        'Zephaniah': 'hbofZep',
                    };

                    let baseAudioName = audioBookPrefixMap[selectedBookId];
                    let audioFilename;

                    if (!baseAudioName) {
                        audioPlaybackStatus.textContent = `Error: No audio prefix found for book '${selectedBookId}'.`;
                        console.error(`Audio Error: No audio prefix found for book '${selectedBookId}'.`);
                        return;
                    }

                    // Handle special cases for chapter numbering and filenames
                    if (selectedBookId === 'Obadiah') {
                        audioFilename = `${baseAudioName}.mp3`;
                    } else if (selectedBookId === 'Psalms') {
                        // Python's str.zfill() is equivalent to JS padStart for numbers
                        const paddedChapter = String(selectedChapter).zfill(3); // FIX: Use zfill
                        audioFilename = `${baseAudioName}_${paddedChapter}.mp3`;
                    } else {
                        // Python's str.zfill() is equivalent to JS padStart for numbers
                        const paddedChapter = String(selectedChapter).zfill(2); // FIX: Use zfill
                        audioFilename = `${baseAudioName}_${paddedChapter}.mp3`;
                    }
                    
                    const audioUrl = `/audio/${audioFilename}`; 
                    console.log(`DEBUG: Attempting to load audio from URL: ${audioUrl}`); 
                    audioPlayer.src = audioUrl;
                    audioPlayer.load(); 
                    audioPlayer.playbackRate = parseFloat(playbackSpeedSelect.value); 
                    audioPlayer.play()
                        .then(() => {
                            audioPlaybackStatus.textContent = `Playing: ${audioFilename} at ${audioPlayer.playbackRate * 100}% speed.`;
                            console.log(`Successfully initiated playback for: ${audioUrl}`);
                            resetHighlight(); // Reset highlight on new playback
                            // No need to set currentTime here, Aeneas handles the actual start time
                        })
                        .catch(error => {
                            audioPlaybackStatus.textContent = `Error playing audio: ${error.message}. Please ensure the audio file '${audioFilename}' exists in the 'data/tanakh_audio' directory and is not corrupted.`;
                            console.error('Audio playback error:', error);
                            console.error(`Attempted audio URL: ${audioUrl}`);
                        });
                };

                // Function to remove all highlights
                function resetHighlight() {
                    if (currentHighlightedWordId) {
                        const prevHighlightedWord = document.getElementById(currentHighlightedWordId);
                        if (prevHighlightedWord) {
                            prevHighlightedWord.classList.remove('highlight');
                        }
                        currentHighlightedWordId = null;
                    }
                    document.querySelectorAll('.highlight').forEach(el => el.classList.remove('highlight'));
                }

                // --- UPDATED: Audio Time Update Listener for Highlighting (using Aeneas timestamps) ---
                audioPlayer.ontimeupdate = () => {
                    if (wordTimestamps.length === 0) {
                        return; // No timestamps loaded to highlight
                    }

                    const currentTime = audioPlayer.currentTime;
                    
                    // Find the word to highlight based on actual timestamps
                    let foundWordToHighlight = false;
                    for (let i = 0; i < wordTimestamps.length; i++) {
                        const wordData = wordTimestamps[i];
                        const wordId = `word-${wordData.verseIndex}-${wordData.wordIndex}`; // Reconstruct ID

                        // Check if the current time falls within this word's actual start and end times
                        if (currentTime >= wordData.start && currentTime < wordData.end) {
                            if (currentHighlightedWordId !== wordId) {
                                // Remove previous highlight
                                if (currentHighlightedWordId) {
                                    const prevHighlighted = document.getElementById(currentHighlightedWordId);
                                    if (prevHighlighted) prevHighlighted.classList.remove('highlight');
                                }
                                // Apply new highlight
                                const currentWordElement = document.getElementById(wordId);
                                if (currentWordElement) {
                                    currentWordElement.classList.add('highlight');
                                    currentHighlightedWordId = wordId;
                                    // Optional: Scroll to the highlighted word (enable with caution)
                                    // currentWordElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
                                }
                            }
                            foundWordToHighlight = true;
                            break; // Found the word for current time, exit loop
                        }
                    }

                    // If audio has ended or current time is past all words, clear highlight
                    if (!foundWordToHighlight && currentHighlightedWordId && currentTime >= audioPlayer.duration) {
                        resetHighlight();
                    }
                };

                // Event listener for speed change
                playbackSpeedSelect.onchange = () => {
                    if (audioPlayer.src) {
                        audioPlayer.playbackRate = parseFloat(playbackSpeedSelect.value);
                        audioPlaybackStatus.textContent = `Playback speed set to ${audioPlayer.playbackRate * 100}%.`;
                    }
                };

                // Optional: Handle audio player events for status updates
                audioPlayer.onended = () => {
                    audioPlaybackStatus.textContent = "Audio finished.";
                    resetHighlight(); // Clear highlight when audio ends
                };
                audioPlayer.onerror = (e) => {
                    audioPlaybackStatus.textContent = `Audio error: ${audioPlayer.error.message || e.type}. Check console for details.`;
                    console.error('Audio element error:', audioPlayer.error);
                    resetHighlight(); // Clear highlight on error
                };
                
                audioPlayer.onpause = () => {
                    // Decide if you want to clear highlight on pause. For now, let's keep it.
                    // resetHighlight(); 
                };

                audioPlayer.onseeking = () => {
                    resetHighlight(); // Clear highlight when user seeks
                };


                // Initial state: audio button disabled
                playChapterAudioButton.disabled = true;

            }); // End DOMContentLoaded listener

        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


# --- NEW ENDPOINT TO FETCH CHAPTER TEXT ---
@app.get("/get_chapter_text/{book_name}/{chapter_num}", response_class=JSONResponse)
async def get_chapter_text(book_name: str, chapter_num: int):
    """
    Fetches Hebrew text for a specific book and chapter from the RAG documents.
    """
    current_documents = get_loaded_documents() 

    if not current_documents: 
        raise HTTPException(status_code=500, detail="RAG documents not loaded. Please ensure RAG initialization is complete.")

    # The book_name passed from the frontend (e.g., "Genesis", "1 Samuel")
    # should now directly match the capitalized full book names stored in RAG documents.
    formatted_book_name = book_name 

    chapter_verses = []
    # Iterate through all loaded documents to find relevant verses
    for doc_entry in current_documents:
        # Example doc_entry from JSON parsing: "Genesis 1:1: בְּרֵאשִׁ֖ית בָּרָ֣א אֱלֹהִ֑ים אֵ֥ת הַשָּׁמַ֖יִם וְאֵ֥ת הָאָֽרֶץ׃"
        try:
            # Split from the RIGHT on the first colon to separate identifier from text
            parts = doc_entry.rsplit(':', 1) 
            if len(parts) < 2:
                print(f"DEBUG: Skipping malformed doc_entry (no text part after identifier): '{doc_entry}'")
                continue 

            identifier_part = parts[0].strip() 
            verse_text_str = parts[1].strip() 

            match = re.match(r'(.+?)\s+(\d+):(\d+)', identifier_part)
            if match:
                doc_book = match.group(1).strip()
                doc_chapter = int(match.group(2))
                doc_verse = int(match.group(3))

                if doc_book == formatted_book_name and doc_chapter == chapter_num:
                    words_list = verse_text_str.split(' ') 
                    chapter_verses.append({
                        "verse_num": doc_verse,
                        "text": words_list # Return as a list of words for frontend
                    })
        except Exception as e:
            print(f"Error parsing document entry '{doc_entry}': {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Sort verses by verse number
    chapter_verses.sort(key=lambda x: x['verse_num'])

    if not chapter_verses:
        print(f"No verses found for {formatted_book_name} Chapter {chapter_num}.")
        raise HTTPException(status_code=404, detail=f"No verses found for {book_name} Chapter {chapter_num}.")

    return {"book": book_name, "chapter": chapter_num, "verses": chapter_verses}


# NEW ENDPOINT: Get Chapter Timestamps
@app.get("/get_chapter_timestamps/{book_name}/{chapter_num}", response_class=JSONResponse)
async def get_chapter_timestamps(book_name: str, chapter_num: int):
    """
    Fetches or generates word-level timestamps for a specific book and chapter.
    """
    current_documents = get_loaded_documents()
    if not current_documents:
        raise HTTPException(status_code=500, detail="RAG documents not loaded, cannot get text for alignment.")

    # Reconstruct the Hebrew text for the chapter from loaded documents
    # We need to get the raw verse data from the RAG documents
    chapter_raw_verses = []
    for doc_entry in current_documents:
        try:
            parts = doc_entry.rsplit(':', 1)
            if len(parts) < 2:
                continue

            identifier_part = parts[0].strip()
            verse_text_str = parts[1].strip()

            match = re.match(r'(.+?)\s+(\d+):(\d+)', identifier_part)
            if match:
                doc_book = match.group(1).strip()
                doc_chapter = int(match.group(2))
                doc_verse = int(match.group(3))

                if doc_book == book_name and doc_chapter == chapter_num:
                    chapter_raw_verses.append({
                        "verse_num": doc_verse,
                        "text": verse_text_str.split(' ') # Keep as list of words
                    })
        except Exception as e:
            print(f"Error parsing document entry for alignment text: {e}")
            continue
    
    chapter_raw_verses.sort(key=lambda x: x['verse_num'])

    if not chapter_raw_verses:
        raise HTTPException(status_code=404, detail=f"No text found for {book_name} Chapter {chapter_num} for alignment.")

    # Construct the audio file path
    audio_book_prefix_map = {
        '1 Chronicles': 'hbof1Ch', '2 Chronicles': 'hbof2Ch', 'Amos': 'hbofAmo',
        'Daniel': 'hbofDan', 'Deuteronomy': 'hbofDeu', 'Esther': 'hbofEst',
        'Exodus': 'hbofExo', 'Ezekiel': 'hbofEzk', 'Ezra': 'hbofEzr',
        'Genesis': 'hbofGen', 'Habbakuk': 'hbofHab', 'Haggai': 'hbofHag',
        'Hosea': 'hbofHos', 'Isaiah': 'hbofIsa', 'Jeremiah': 'hbofJer',
        'Job': 'hbofJob', 'Joel': 'hbofJol', 'Jonah': 'hbofJon',
        'Joshua': 'hbofJos', 'Judges': 'hbofJdg', '1 Kings': 'hbof1Ki',
        '2 Kings': 'hbof2Ki', 'Koheleth (Ecclesiastes)': 'hbofEcc',
        'Lamentations': 'hbofLam', 'Leviticus': 'hbofLev', 'Malachi': 'hbofMal',
        'Micah': 'hbofMic', 'Nahum': 'hbofNam', 'Nehemiah': 'hbofNeh',
        'Numbers': 'hbofNum', 'Obadiah': 'hbofOba', 'Proverbs': 'hbofPro',
        'Psalms': 'hbofPsa', 'Ruth': 'hbofRut', '1 Samuel': 'hbof1Sa',
        '2 Samuel': 'hbof2Sa', 'Song of Songs': 'hbofSng', 'Zechariah': 'hbofZec',
        'Zephaniah': 'hbofZep',
    }
    base_audio_name = audio_book_prefix_map.get(book_name)
    if not base_audio_name:
        raise HTTPException(status_code=400, detail=f"Audio prefix not found for book '{book_name}'.")

    audio_filename = ""
    if book_name == 'Obadiah':
        audio_filename = f"{base_audio_name}.mp3"
    elif book_name == 'Psalms':
        # FIX: Use zfill() for Python string padding
        padded_chapter = str(chapter_num).zfill(3)
        audio_filename = f"{base_audio_name}_{padded_chapter}.mp3"
    else:
        # FIX: Use zfill() for Python string padding
        padded_chapter = str(chapter_num).zfill(2)
        audio_filename = f"{base_audio_name}_{padded_chapter}.mp3"

    audio_path = os.path.join(AUDIO_DIR, audio_filename)
    if not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail=f"Audio file not found for {book_name} Chapter {chapter_num}: {audio_filename}")

    # Call the alignment module
    try:
        # Pass the original hebrew_text_verses (list of dicts with verse_num and text)
        # The alignment_module will handle flattening and mapping.
        timestamps = await generate_timestamps(book_name, chapter_num, chapter_raw_verses, audio_path)
        
        return timestamps

    except Exception as e:
        print(f"Error generating timestamps: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to generate timestamps: {str(e)}")


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
