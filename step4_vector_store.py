import os
import json
import faiss
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define input and output directories
input_dir = "/home/sswarna/Documents/oran_docs/output_all/step3_embedded"
output_dir = "/home/sswarna/Documents/oran_docs/output_all/step4_vector_store"
os.makedirs(output_dir, exist_ok=True)

# Function to load embeddings from Step 3
def load_embeddings(input_file):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    embeddings = []
    metadata = []
    for chunk in data:
        if "embedding" in chunk and chunk["embedding"]:
            embeddings.append(chunk["embedding"])
            metadata.append(chunk["metadata"])
        else:
            logging.warning(f"Skipping chunk with missing embedding: {chunk.get('chunk_index', 'Unknown')}")
    return np.array(embeddings, dtype="float32"), metadata

# Function to save FAISS index
def save_faiss_index(index, metadata, output_path):
    faiss.write_index(index, f"{output_path}.faiss")
    with open(f"{output_path}_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

# Process each year
for year in ["2022", "2023", "2024"]:
    input_file = os.path.join(input_dir, f"{year}_embedded.json")
    output_path = os.path.join(output_dir, f"{year}_vector_store")

    if not os.path.exists(input_file):
        logging.warning(f"Input file for {year} not found. Skipping...")
        continue

    logging.info(f"Processing {year} embeddings...")

    # Load embeddings and metadata
    embeddings, metadata = load_embeddings(input_file)
    if embeddings.size == 0:
        logging.warning(f"No valid embeddings found for {year}. Skipping...")
        continue

    logging.info(f"Loaded {len(embeddings)} embeddings for {year}.")

    # Create FAISS index
    try:
        index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance
        index.add(embeddings)
        logging.info(f"FAISS index created and populated for {year}.")
    except Exception as e:
        logging.error(f"Error creating FAISS index for {year}: {e}")
        continue

    # Save the index and metadata
    save_faiss_index(index, metadata, output_path)
    logging.info(f"Saved vector store for {year} to {output_path}.")
