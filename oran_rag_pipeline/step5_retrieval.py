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
    match = re.search(r"(?:document|file)\s+([\w\.-]+)", query, re.IGNORECASE)
    return match.group(1) if match else None

def approximate_title_match(query_doc, stored_titles):
    """Find the closest document title match using Levenshtein distance."""
    best_match = None
    best_score = 0
    
    for title in stored_titles:
        match_score = ratio(query_doc.lower(), title.lower())
        if match_score > best_score:
            best_match = title
            best_score = match_score
    
    return best_match if best_score > 0.85 else None  # Increased threshold to 85%

def filter_irrelevant_content(text):
    """Remove IPR and legal-related text from the retrieved content."""
    filtered_text = re.sub(r'(?i)(IPR|copyright|patents|trademarks|terms of use).*', '', text)
    return filtered_text.strip()

def retrieve_relevant_chunks(query):
    """Retrieve relevant document chunks using metadata and vector search."""
    doc_name = extract_document_name(query)
    retrieved_chunks = []

    # Get stored document titles
    stored_titles = set(metadata.get("title", "Unknown") for metadata in collection.get(include=["metadatas"])['metadatas'])
    
    # If document name is found, use exact metadata search
    if doc_name:
        print(f"🔍 Detected document name in query: {doc_name}. Using Metadata + Vector Search.")
        
        if doc_name in stored_titles:
            metadata_results = collection.get(
                where={"title": doc_name},
                include=["documents", "metadatas"]
            )
            
            if metadata_results["documents"]:
                retrieved_chunks.extend([
                    {
                        "source": metadata_results["metadatas"][i].get("title", "Unknown"),
                        "score": 1.0,
                        "content": filter_irrelevant_content(metadata_results["documents"][i])
                    }
                    for i in range(len(metadata_results["documents"]))
                ])
        else:
            print("Couldn't find a file")
    
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
            "content": filter_irrelevant_content(results["documents"][0][i])
        }
        for i in range(len(results["documents"][0]))
    ])
    
    return retrieved_chunks[:TOP_K+2]
    ### Instructions for LLM:
    #- Ensure that your foucs on query and only choose context relevant to query
    #- Avoid hallucinations.
    #- Focus on technical aspects and real document references.
def generate_dynamic_prompt_using_llm(query, retrieved_chunks):
    context = "\n".join([f"Source: {chunk['source']}\n{chunk['content']}" for chunk in retrieved_chunks])

    llm_prompt = f"""
    ### Instructions for LLM:
    - **Only answer based on the retrieved context.** 
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

def main():
    print("🎯 Ollama RAG System Ready! Enter your queries below.")
    
    while True:
        query = input("🔍 Enter your query (or 'exit' to stop): ").strip()
        if query.lower() == "exit": 
            break
        
        retrieved_chunks = retrieve_relevant_chunks(query)
        
        if not retrieved_chunks:
            print("⚠️ No relevant data retrieved.")
            continue
        
        structured_prompt = generate_dynamic_prompt_using_llm(query, retrieved_chunks)
        print("\n📝 **Final Answer from LLM:**")
        print(structured_prompt)

if __name__ == "__main__":
    main()
