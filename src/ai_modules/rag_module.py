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

# For XML parsing
from lxml import etree # Import lxml for efficient XML parsing

# --- Global Variables ---
llm_model = None
llm_tokenizer = None
embedding_model = None
faiss_index = None
documents = [] # Stores original text chunks
document_paths = [] # Stores paths to original files for context

# --- Configuration ---
# CHANGE THIS LINE: Reverting to the 7B Hebrew-optimized Mistral model
LLM_MODEL_NAME = "yam-peleg/Hebrew-Mistral-7B"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Point to where your XML content will be
CONTENT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "content")
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
            # REMOVED: load_in_8bit_fp32_cpu_offload=True, (was causing unused kwarg warning)
        )

        # REMOVED: custom_device_map, as device_map="auto" should work for 7B models
        
        llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
        llm_model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto", # Revert to auto, as it worked for 7B previously
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

def load_and_chunk_content(content_dir: str) -> Tuple[List[str], List[str]]:
    """
    Loads text content from XML files in a directory and chunks it by verse.
    This version uses lxml to parse the structured XML (UXLC format) and explicitly
    handles XML namespaces to ensure correct element selection.
    """
    global documents, document_paths
    documents = []
    document_paths = []
    print(f"RAG Debug: Attempting to load content from: {content_dir}")
    if not os.path.exists(content_dir):
        print(f"RAG Debug: Content directory NOT FOUND: {content_dir}. Please ensure XML files are placed here.")
        return [], []

    for root_dir, _, files in os.walk(content_dir):
        for file_name in files:
            if file_name.lower().endswith(".xml"): # Look for XML files
                file_path = os.path.join(root_dir, file_name)
                # Skip known metadata XML files if they exist and contain only schema/notes
                if "tanach.xml" in file_name.lower() or "tanach.xsd" in file_name.lower():
                    print(f"RAG Debug: Skipping known metadata XML file: {file_path}")
                    continue

                print(f"RAG Debug: Processing XML file: {file_path}")
                
                try:
                    tree = etree.parse(file_path) # type: ignore # Pylance false positive for missing 'parser' argument
                    root = tree.getroot()

                    # Extract default namespace if present
                    namespace_match = re.match(r'\{([^}]+)\}', root.tag)
                    ns = {'ns': namespace_match.group(1)} if namespace_match else {}
                    
                    print(f"RAG Debug: Detected namespace: {ns}")

                    # Extract book title from XML metadata (using a more general XPath for title)
                    book_title_xpath = "//ns:teiHeader/ns:fileDesc/ns:titleStmt/ns:title[@level='a' and @type='main']" if ns else "//teiHeader/fileDesc/titleStmt/title[@level='a' and @type='main']"
                    book_title_elem = tree.xpath(book_title_xpath, namespaces=ns)
                    book_name = book_title_elem[0].text.strip() if book_title_elem else os.path.splitext(file_name)[0]
                    print(f"RAG Debug: Extracted book name: {book_name} from {file_name}")

                    # Iterate through chapters and verses
                    book_xpath = "//ns:book" if ns else "//book"
                    chapter_xpath = "./ns:chapter" if ns else "./chapter"
                    verse_xpath = "./ns:verse" if ns else "./verse"
                    word_xpath = ".//ns:w" if ns else ".//w"

                    book_elements = tree.xpath(book_xpath, namespaces=ns)
                    if not book_elements:
                        print(f"RAG Debug: No <book> element found in {file_path} using XPath '{book_xpath}'. Skipping file content parsing.")
                        continue
                    
                    for book_elem in book_elements:
                        xml_book_name = book_elem.get("name")
                        xml_book_name_hebrew = book_elem.get("namehebrew")
                        print(f"RAG Debug: Found <book> element: name='{xml_book_name}', namehebrew='{xml_book_name_hebrew}'")

                        for chapter_elem in book_elem.xpath(chapter_xpath, namespaces=ns):
                            chapter_num = chapter_elem.get("n")
                            if not chapter_num:
                                print(f"RAG Debug: No 'n' attribute for chapter in {file_path}. Skipping chapter.")
                                continue
                            
                            for verse_elem in chapter_elem.xpath(verse_xpath, namespaces=ns):
                                verse_num = verse_elem.get("n")
                                if not verse_num:
                                    print(f"RAG Debug: No 'n' attribute for verse in Chapter {chapter_num} of {file_path}. Skipping verse.")
                                    continue
                                
                                hebrew_words = [w.text.strip() for w in verse_elem.xpath(word_xpath, namespaces=ns) if w.text is not None]
                                
                                verse_content = " ".join(hebrew_words)
                                
                                if verse_content:
                                    verse_content = re.sub(r'\s+', ' ', verse_content).strip()
                                    
                                    full_identifier = f"{book_name} {chapter_num}:{verse_num}"
                                    documents.append(full_identifier + ": " + verse_content)
                                    document_paths.append(f"{book_name}:{chapter_num}:{verse_num}")
                                else:
                                    print(f"RAG Debug: No Hebrew content found for verse {book_name} {chapter_num}:{verse_num} in {file_path}. Skipping.")

                except etree.XMLSyntaxError as e:
                    print(f"RAG Debug: XML Syntax Error in {file_path}: {e}. Skipping file.")
                except Exception as e:
                    print(f"RAG Debug: Error processing XML file {file_path}: {e}")
                    import traceback
                    traceback.print_exc()
    
    print(f"RAG Debug: Loaded {len(documents)} content chunks from {content_dir}.")
    if not documents:
        print("RAG Debug: WARNING: No documents were loaded. FAISS index will be empty.")
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
    if faiss_index is None or not documents:
        print("RAG Debug: RAG components not fully initialized. Attempting to re-initialize.")
        docs_from_files, _ = load_and_chunk_content(CONTENT_DIR)
        create_or_load_faiss_index(docs_from_files)
        if faiss_index is None or not documents:
            return "Error: RAG content not initialized. No documents found or index could not be created."

    # Assert that models/tokenizer/index are not None for Pylance
    assert llm_model is not None, "LLM model not loaded."
    assert llm_tokenizer is not None, "LLM tokenizer not loaded."
    assert embedding_model is not None, "Embedding model not loaded."
    assert faiss_index is not None, "FAISS index not loaded."

    print(f"Processing RAG query: '{query}'")

    # 1. Embed the query
    query_embedding: np.ndarray = embedding_model.encode([query], convert_to_numpy=True).astype('float32') # type: ignore
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
        f"Using the following context, answer the question. "
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
        temperature=0.7,
        top_p=0.9,
        pad_token_id=llm_tokenizer.eos_token_id
    )
    response_text = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Post-process to extract only the answer part
    if "Answer:" in response_text:
        response_text = response_text.split("Answer:", 1)[1].strip()
    elif "Question:" in response_text:
         response_text = response_text.split("Question:", 1)[0].strip()
    elif "Context:" in response_text:
        response_text = response_text.split("Context:", 1)[0].strip()

    print(f"LLM generated response.")
    return response_text

# --- Initialization on module import ---
async def initialize_rag():
    """Initializes all RAG components."""
    print("Initializing RAG module...")
    await load_llm_model()
    load_embedding_model()
    docs, _ = load_and_chunk_content(CONTENT_DIR)
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