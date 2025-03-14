import os
import json
import chromadb
from tqdm import tqdm

# === Configuration ===
EMBEDDINGS_INPUT_DIR = "/home/sswarna/Documents/oran_docs/output_all/Step3_Embeddings"
CHROMA_DB_DIR = "/home/sswarna/Documents/oran_docs/oran_rag_pipeline/chroma_index"
COLLECTION_NAME = "oran_docs"

# Remove old ChromaDB storage
if os.path.exists(CHROMA_DB_DIR):
    for file in os.listdir(CHROMA_DB_DIR):
        file_path = os.path.join(CHROMA_DB_DIR, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
    print("🗑️ Cleared old ChromaDB storage.")

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

# Create or get collection
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

# Process all embedding files
def store_embeddings(input_dir):
    for year in ["2022", "2023", "2024"]:
        year_dir = os.path.join(input_dir, f"Output_{year}")
        if not os.path.exists(year_dir):
            print(f"⚠️ Skipping missing directory: {year_dir}")
            continue

        print(f"📂 Processing Year: {year}")
        for filename in tqdm(os.listdir(year_dir), desc=f"Year {year}"):
            if not filename.endswith("_embeddings.json"):
                continue

            input_filepath = os.path.join(year_dir, filename)
            with open(input_filepath, "r", encoding="utf-8") as f:
                embedding_data = json.load(f)

            for chunk in embedding_data:
                chunk_id = f"{chunk['title']}_chunk_{chunk['chunk_index']}"  # <-- Ensuring chunk ID is unique
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

# Run the storage function
store_embeddings(EMBEDDINGS_INPUT_DIR)
