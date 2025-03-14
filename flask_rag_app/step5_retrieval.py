import requests
import json
import chromadb
import re
from sentence_transformers import SentenceTransformer
from Levenshtein import ratio  # Install with: pip install python-Levenshtein

# === Configuration ===
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama2:7b"
CHROMA_DB_DIR = "/home/sswarna/Documents/oran_docs/oran_rag_pipeline/chroma_index"
COLLECTION_NAME = "oran_docs"
TOP_K = 50  # Limit retrieved chunks

# Load embedding model
EMBEDDING_MODEL_PATH = "/home/sswarna/models/all-MiniLM-L12-v2"
embed_model = SentenceTransformer(EMBEDDING_MODEL_PATH)

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
collection = chroma_client.get_collection(COLLECTION_NAME)

def embed_query(query):
    """Generate query embeddings to match stored embeddings."""
    return embed_model.encode(query).tolist()

def extract_document_name(query):
    """Extract document name if mentioned in query."""
    if not isinstance(query, str) or not query.strip():
        print(f"DEBUG: Invalid query type: {type(query)} or empty string")
        return None
    
    match = re.search(r"(?:document|file)\s+([\w\.-]+)", query, re.IGNORECASE)
    return match.group(1) if match else None

def retrieve_relevant_chunks(query):
    """Retrieve relevant document chunks using metadata and vector search."""
    doc_name = extract_document_name(query)
    retrieved_chunks = []

    # Get stored document titles
    stored_titles = set(metadata.get("title", "Unknown") for metadata in collection.get(include=["metadatas"])['metadatas'])
    
    # If document name is found, use exact metadata search
    if doc_name and doc_name in stored_titles:
        metadata_results = collection.get(
            where={"title": doc_name},
            include=["documents", "metadatas"]
        )
        
        if metadata_results["documents"]:
            retrieved_chunks.extend([
                {
                    "source": metadata_results["metadatas"][i].get("title", "Unknown"),
                    "score": 1.0,
                    "content": metadata_results["documents"][i]
                }
                for i in range(len(metadata_results["documents"]))
            ])
    
    # Perform Vector Search
    query_embedding = embed_query(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"]
    )
    
    retrieved_chunks.extend([
        {
            "source": results["metadatas"][0][i].get("title", "Unknown"),
            "score": results["distances"][0][i],
            "content": results["documents"][0][i]
        }
        for i in range(len(results["documents"][0]))
    ])
    
    return retrieved_chunks[:TOP_K+2]


def generate_generic_llm(query):
    llm_prompt = f"""
    ### Query:
    {query}

    ### Expected Output:
    Provide a structured and accurate response.
    """

    try:
        response = requests.post(OLLAMA_URL, json={"model": OLLAMA_MODEL, "prompt": llm_prompt, "stream": False})
        if response.status_code == 200:
            return response.json().get("response", "⚠️ No response from Ollama.")
        else:
            return f"❌ Error: {response.status_code}"
    except Exception as e:
        return f"❌ Ollama Request Failed: {e}"


def generate_dynamic_prompt_using_llm(query, retrieved_chunks):
    """Generate structured prompt for LLM using retrieved document context."""
    context = "\n".join([f"Source: {chunk['source']}\n{chunk['content']}" for chunk in retrieved_chunks])

    llm_prompt = f"""
    ### Instructions for LLM:
    - **Only answer based on the retrieved context. if present** 
    - **Do not mix Non-RT RIC and Near-RT RIC roles.**
    - **Exclude security considerations unless explicitly mentioned in the retrieved context.**

    ### Retrieved Context:
    {context}

    ### Query:
    {query}

    ### Expected Output:
    Provide a structured and accurate response strictly from the context.
    """

    try:
        response = requests.post(OLLAMA_URL, json={"model": OLLAMA_MODEL, "prompt": llm_prompt, "stream": False})
        if response.status_code == 200:
            return response.json().get("response", "⚠️ No response from Ollama.")
        else:
            return f"❌ Error: {response.status_code}"
    except Exception as e:
        return f"❌ Ollama Request Failed: {e}"

def query_retrieval(user_query):
    """Retrieves relevant document chunks and generates an LLM-based response."""
    retrieved_chunks = retrieve_relevant_chunks(user_query)

    # if not retrieved_chunks:
    #     return "⚠️ No relevant data retrieved."

    structured_response = generate_dynamic_prompt_using_llm(user_query, retrieved_chunks)
    generic_response = generate_generic_llm(user_query)
    return {structured_response, generic_response}
