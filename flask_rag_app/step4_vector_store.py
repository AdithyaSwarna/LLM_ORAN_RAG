import os
import json
import chromadb
from tqdm import tqdm

# === Configuration ===
EMBEDDINGS_INPUT_DIR = "/home/sswarna/Documents/oran_docs/output_all/Step3_Embeddings"
CHROMA_DB_DIR = "/home/sswarna/Documents/oran_docs/oran_rag_pipeline/chroma_index"
COLLECTION_NAME = "oran_docs"

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

# Create or get collection
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

def store_embeddings(input_filepath):
    """Stores embeddings for a single uploaded file in ChromaDB."""
    if not os.path.exists(input_filepath):
        print(f"⚠️ ERROR: File not found: {input_filepath}")
        return

    with open(input_filepath, "r", encoding="utf-8") as f:
        embedding_data = json.load(f)

    print(f"\n📂 Processing File: {os.path.basename(input_filepath)}")

    for chunk in tqdm(embedding_data, desc=f"Storing {os.path.basename(input_filepath)}"):
        chunk_id = f"{chunk['title']}_chunk_{chunk['chunk_index']}"  # <-- Ensure unique chunk ID
        embedding_vector = chunk.get("embedding")

        # Ensure the correct document name is stored in metadata
        metadata = {
            "title": chunk.get("title", "Unknown"),  # <-- Preserve title
            "source": chunk.get("source_file", "Unknown"),
            "token_length": chunk.get("token_length", 0),
            "embedding_model": chunk.get("embedding_model", "Unknown Model"),
        }

        if embedding_vector and isinstance(embedding_vector, list):
            collection.add(
                ids=[chunk_id],
                embeddings=[embedding_vector],
                metadatas=[metadata],
                documents=[chunk.get("chunk_content", "")]
            )
        else:
            print(f"⚠️ Skipped chunk {chunk_id} due to missing or invalid embedding.")

    print("✅ Step 4: Vector Store Updated Successfully!")

# === Process only the uploaded file ===
def process_uploaded_vector_store(uploaded_file_path):
    """Processes only the uploaded file embeddings into ChromaDB."""
    
    if not os.path.exists(uploaded_file_path):
        print(f"⚠️ ERROR: Uploaded file not found: {uploaded_file_path}")
        return

    filename = os.path.basename(uploaded_file_path)
    file_base_name = os.path.splitext(filename)[0]

    print(f"\n🔹 Storing Uploaded File in Vector Store: {filename}\n")

    input_filepath = os.path.join(EMBEDDINGS_INPUT_DIR, f"{file_base_name}_embeddings.json")

    if not os.path.exists(input_filepath):
        print(f"⚠️ ERROR: Embeddings file not found: {input_filepath}. Skipping vector storage.")
        return

    # Store embeddings for this file
    store_embeddings(input_filepath)
