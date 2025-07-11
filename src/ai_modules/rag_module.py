# src/ai_modules/rag_module.py

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re
from typing import List, Dict, Tuple, Optional
import asyncio
import json # <--- ADDED THIS IMPORT

# --- Global Variables ---
llm_model = None
llm_tokenizer = None
embedding_model = None
faiss_index = None
documents = [] # Stores original text chunks
document_paths = [] # Stores paths to original files for context

# --- Configuration ---
LLM_MODEL_NAME = "yam-peleg/Hebrew-Mistral-7B"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Point to where your JSON content will be
# Ensure hebrew_bible_with_nikkud.json is in this directory: D:\AI\Gits\hebrew_tutor_ai_poc\data\content\
CONTENT_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "content", "hebrew_bible_with_nikkud.json")
FAISS_INDEX_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "embeddings", "content_faiss.index")
DOCUMENTS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "embeddings", "content_docs.txt")


# --- Model Loading Functions ---

async def load_llm_model():
    """Loads the quantized LLM model and tokenizer."""
    global llm_model, llm_tokenizer
    if llm_model is None:
        print(f"Loading LLM model: {LLM_MODEL_NAME} (quantized for 8GB GPU)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        
        llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
        llm_model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        print(f"LLM model '{LLM_MODEL_NAME}' loaded successfully.") 
    return llm_model, llm_tokenizer

def load_embedding_model():
    """Loads the Sentence Transformer embedding model."""
    global embedding_model
    if embedding_model is None:
        print(f"Loading Embedding model: {EMBEDDING_MODEL_NAME}...")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        if torch.cuda.is_available():
            embedding_model.to('cuda')
            print(f"Embedding model moved to CUDA.")
        print(f"Embedding model '{EMBEDDING_MODEL_NAME}' loaded successfully.")
    return embedding_model

# --- Content Processing & Indexing ---

def load_and_chunk_content(content_file: str) -> Tuple[List[str], List[str]]:
    """
    Loads text content from a JSON file and chunks it by verse.
    Expected JSON structure: { "BookAbbr": [ [ [ "word1", "word2" ], ... ], ... ] }
    """
    global documents, document_paths
    documents = []
    document_paths = []
    print(f"RAG Debug: Attempting to load content from JSON: {content_file}")

    if not os.path.exists(content_file):
        print(f"RAG Debug: Content JSON file NOT FOUND: {content_file}. Please ensure 'hebrew_bible_with_nikkud.json' is placed in the 'data/content' directory.")
        return [], []

    # Mapping for JSON book abbreviations to full names (for frontend display and consistent lookup)
    # These values must EXACTLY match the 'id' and 'name' in the frontend's books array in main.py
    book_abbr_to_full_name_map = {
        'Gen': 'Genesis', 'Exod': 'Exodus', 'Lev': 'Leviticus', 'Num': 'Numbers',
        'Deut': 'Deuteronomy', 'Josh': 'Joshua', 'Judg': 'Judges', 'Ruth': 'Ruth',
        '1Sam': '1 Samuel', '2Sam': '2 Samuel', '1Kgs': '1 Kings', '2Kgs': '2 Kings',
        'Isa': 'Isaiah', 'Jer': 'Jeremiah', 'Ezek': 'Ezekiel', 'Hos': 'Hosea',
        'Joel': 'Joel', 'Amos': 'Amos', 'Obad': 'Obadiah', 'Jonah': 'Jonah',
        'Mic': 'Micah', 'Nah': 'Nahum', 'Hab': 'Habbakuk', 'Zeph': 'Zephaniah',
        'Hag': 'Haggai', 'Zech': 'Zechariah', 'Mal': 'Malachi', 'Ps': 'Psalms',
        'Prov': 'Proverbs', 'Job': 'Job', 'Song': 'Song of Songs', 
        'Lam': 'Lamentations', 'Eccl': 'Koheleth (Ecclesiastes)', 'Esth': 'Esther',
        'Dan': 'Daniel', 'Ezra': 'Ezra', 'Neh': 'Nehemiah', '1Chr': '1 Chronicles',
        '2Chr': '2 Chronicles', 
    }

    try:
        with open(content_file, mode='r', encoding='utf-8') as file:
            data = json.load(file)

        for book_abbr, chapters_data in data.items():
            # Convert JSON book abbreviation (e.g., 'Gen', '1Chr') to full name (e.g., 'Genesis', '1 Chronicles')
            book_name = book_abbr_to_full_name_map.get(book_abbr, book_abbr) # Fallback to abbr if not in map

            for chapter_num, verses_data in enumerate(chapters_data, start=1):
                # Ensure chapter_num is an integer for consistency
                chapter_num_int = int(chapter_num) 
                for verse_num, words_list in enumerate(verses_data, start=1):
                    # Ensure verse_num is an integer
                    verse_num_int = int(verse_num)

                    # Join words to form the full verse text
                    hebrew_text = " ".join(words_list)
                    
                    if hebrew_text:
                        # Format: "FullBookName Chapter:Verse: Word1 Word2 ..."
                        full_identifier = f"{book_name} {chapter_num_int}:{verse_num_int}"
                        documents.append(full_identifier + ": " + hebrew_text)
                        document_paths.append(f"{book_name}:{chapter_num_int}:{verse_num_int}")
                        # Print only a few for brevity during large loads
                        if len(documents) % 1000 == 0: 
                            print(f"RAG Debug: Added chunk {len(documents)}: {full_identifier}: {hebrew_text[:50]}...")
                    else:
                        print(f"RAG Debug: No Hebrew content found for {book_name} {chapter_num_int}:{verse_num_int}. Skipping.")

    except json.JSONDecodeError as e:
        print(f"RAG Debug: JSON Decode Error in {content_file}: {e}. Please check JSON format.")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"RAG Debug: Error processing JSON file {content_file}: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"RAG Debug: Loaded {len(documents)} content chunks from {content_file}.")
    if not documents:
        print("RAG Debug: WARNING: No documents were loaded from JSON. FAISS index will be empty.")
    return documents, document_paths

def create_or_load_faiss_index(docs: List[str]):
    """
    Creates a new FAISS index from content embeddings or loads an existing one.
    The index and the list of documents are saved/loaded to/from the data/embeddings directory.
    """
    global faiss_index, documents
    os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)

    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(DOCUMENTS_PATH):
        print(f"RAG Debug: Loading existing FAISS index from {FAISS_INDEX_PATH}...")
        try:
            faiss_index = faiss.read_index(FAISS_INDEX_PATH)
            with open(DOCUMENTS_PATH, "r", encoding="utf-8") as f:
                documents = [line.strip() for line in f if line.strip()]
            print(f"RAG Debug: FAISS index and {len(documents)} documents loaded.")
        except Exception as e:
            print(f"RAG Debug: Error loading FAISS index or documents: {e}. Recreating index.")
            faiss_index = None
    else:
        print("RAG Debug: Existing FAISS index or documents not found. Creating new index...")

    if faiss_index is None:
        if not docs:
            print("RAG Debug: No documents provided to create index. Skipping FAISS index creation.")
            return

        embedder = load_embedding_model()
        print("RAG Debug: Generating embeddings for content (this may take a while for large datasets)...")
        embeddings: np.ndarray = embedder.encode(docs, convert_to_numpy=True, show_progress_bar=True).astype('float32') # type: ignore
        embedding_dimension = embeddings.shape[1]

        faiss_index = faiss.IndexFlatL2(embedding_dimension)
        faiss_index.add(embeddings) # type: ignore
        print(f"RAG Debug: FAISS index created with {faiss_index.ntotal} vectors.")

        faiss.write_index(faiss_index, FAISS_INDEX_PATH)
        with open(DOCUMENTS_PATH, "w", encoding="utf-8") as f:
            for doc in docs:
                f.write(doc + "\n")
        print(f"RAG Debug: FAISS index saved to {FAISS_INDEX_PATH}.")
        print(f"RAG Debug: Documents list saved to {DOCUMENTS_PATH}.")

# --- RAG Function ---

async def get_rag_response(query: str, top_k: int = 3) -> str:
    """
    Performs Retrieval-Augmented Generation.
    1. Embeds the query.
    2. Retrieves relevant documents using FAISS.
    3. Constructs a prompt with the query and retrieved context.
    4. Generates a response using the LLM.
    """
    # Ensure all necessary models and index are loaded
    if llm_model is None or llm_tokenizer is None:
        await load_llm_model()
    if embedding_model is None:
        load_embedding_model()
    # No need to re-initialize RAG components here, just get the documents
    if faiss_index is None or not documents:
        # This case should ideally not be hit if startup_event runs correctly
        print("RAG Debug: RAG components not fully initialized when calling get_rag_response. This is unexpected.")
        return "Error: RAG content not initialized. No documents found or index could not be created."

    # Assert that models/tokenizer/index are not None for Pylance
    assert llm_model is not None, "LLM model not loaded."
    assert llm_tokenizer is not None, "LLM tokenizer not loaded."
    assert embedding_model is not None, "Embedding model not loaded."
    assert faiss_index is not None, "FAISS index not loaded."

    print(f"Processing RAG query: '{query}'")

    # 1. Embed the query
    query_embedding: np.ndarray = embedding_model.encode([query], convert_to_numpy=True, show_progress_bar=False).astype('float32') # type: ignore
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)

    # 2. Retrieve relevant documents using FAISS
    D, I = faiss_index.search(query_embedding, top_k) # type: ignore

    retrieved_docs = [documents[idx] for idx in I[0] if idx != -1]
    print(f"RAG Debug: Retrieved {len(retrieved_docs)} documents.")
    if not retrieved_docs:
        return "I couldn't find relevant information in the provided content."

    # 3. Construct prompt with context
    context = "\n".join(retrieved_docs)
    prompt = (
        f"Based *only* on the following context, answer the question concisely and directly. "
        f"Do not include any conversational filler, greetings, or suggestions for further questions. "
        f"Do not start your answer with 'Response:'. "
        f"If the answer is not explicitly in the context, state that you cannot answer based on the provided context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        f"Answer:"
    )

    # 4. Generate response using LLM
    inputs = llm_tokenizer(prompt, return_tensors="pt").to(llm_model.device)

    outputs = llm_model.generate(
        **inputs,
        max_new_tokens=200,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.4, # Lowered for more focused responses
        top_p=0.9,
        pad_token_id=llm_tokenizer.eos_token_id
    )
    response_text = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Post-process to extract only the answer part and remove "Response:"
    if "Answer:" in response_text:
        response_text = response_text.split("Answer:", 1)[1].strip()
    # Check for "Response:" at the beginning of the string and remove it
    if response_text.lower().startswith("response:"):
        response_text = response_text[len("response:"):].strip()
    elif "Question:" in response_text:
         response_text = response_text.split("Question:", 1)[0].strip()
    elif "Context:" in response_text:
        response_text = response_text.split("Context:", 1)[0].strip()

    print(f"LLM generated response.")
    return response_text

def get_loaded_documents() -> List[str]:
    """Returns the list of loaded RAG documents."""
    return documents

async def initialize_rag():
    """Initializes all RAG components."""
    print("Initializing RAG module...")
    await load_llm_model()
    load_embedding_model()
    # --- MODIFIED: Pass CONTENT_FILE instead of CONTENT_DIR ---
    docs, _ = load_and_chunk_content(CONTENT_FILE) 
    create_or_load_faiss_index(docs)
    print("RAG module initialization complete.")

if __name__ == "__main__":
    print("Running direct RAG test...")
    asyncio.run(initialize_rag())

    test_query_1 = "What did God create in the beginning?"
    response_1 = asyncio.run(get_rag_response(test_query_1))
    print(f"\nQuery: {test_query_1}")
    print(f"Response: {response_1}")

    test_query_2 = "What was the earth like in Genesis 1:2?"
    response_2 = asyncio.run(get_rag_response(test_query_2))
    print(f"\nQuery: {test_query_2}")
    print(f"Response: {response_2}")

    test_query_3 = "Who was Amos?"
    response_3 = asyncio.run(get_rag_response(test_query_3))
    print(f"\nQuery: {test_query_3}")
    print(f"Response: {response_3}")

    test_query_4 = "What is the capital of France?"
    response_4 = asyncio.run(get_rag_response(test_query_4))
    print(f"\nQuery: {test_query_4}")
    print(f"Response: {response_4}")
