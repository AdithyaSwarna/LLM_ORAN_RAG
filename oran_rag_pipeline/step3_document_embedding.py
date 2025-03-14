import os
import json
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# === Configuration ===
CHUNKS_INPUT_BASE_DIR = "/home/sswarna/Documents/oran_docs/output_all/Step2_chunks"
EMBEDDINGS_OUTPUT_BASE_DIR = "/home/sswarna/Documents/oran_docs/output_all/Step3_Embeddings"
MODEL_PATH = "/home/sswarna/models/all-MiniLM-L12-v2"

os.makedirs(EMBEDDINGS_OUTPUT_BASE_DIR, exist_ok=True)

# Load locally stored embedding model
model = SentenceTransformer(MODEL_PATH)

def process_file(input_filepath, output_filepath):
    """Process a chunk file, generate embeddings, and save results."""
    with open(input_filepath, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)

    if "chunks" not in chunks_data or not isinstance(chunks_data["chunks"], list):
        print(f"❌ ERROR: Expected a list in {input_filepath}, but got {type(chunks_data)}")
        return

    # Extract title (ensuring backward compatibility)
    title = chunks_data.get("title", os.path.basename(input_filepath).replace("_chunks.json", ""))

    embeddings_data = []
    for chunk in tqdm(chunks_data["chunks"], desc=f"Processing {os.path.basename(input_filepath)}"):
        if "chunk_content" not in chunk:
            continue

        chunk_text = chunk["chunk_content"]
        embedding_vector = model.encode(chunk_text).tolist()

        embeddings_data.append({
            "title": title,  # <-- ✅ Preserve title in embeddings output
            "chunk_index": chunk["chunk_index"],
            "chunk_content": chunk_text,
            "embedding": embedding_vector,
            "token_length": len(chunk_text.split()),
            "source_file": os.path.basename(input_filepath).replace("_chunks.json", ""),
            "embedding_model": "all-MiniLM-L12-v2"
        })

    # Save the embeddings
    with open(output_filepath, "w", encoding="utf-8") as f:
        json.dump(embeddings_data, f, indent=4)

    print(f"✅ Processed: {input_filepath} → {output_filepath}")

# === Process all chunk files ===
for year in ["2022", "2023", "2024"]:
    year_input_dir = os.path.join(CHUNKS_INPUT_BASE_DIR, f"Output_{year}")
    year_output_dir = os.path.join(EMBEDDINGS_OUTPUT_BASE_DIR, f"Output_{year}")
    os.makedirs(year_output_dir, exist_ok=True)

    print(f"\n🔹 Processing Year: {year}...\n")

    for filename in tqdm(os.listdir(year_input_dir), desc=f"Year {year}"):
        if filename.endswith("_chunks.json"):
            input_filepath = os.path.join(year_input_dir, filename)
            output_filepath = os.path.join(year_output_dir, filename.replace("_chunks.json", "_embeddings.json"))
            process_file(input_filepath, output_filepath)

print("🎯 Step 3: Embedding Generation Completed Successfully!")
