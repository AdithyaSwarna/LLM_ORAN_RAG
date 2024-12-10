import os
import json
import requests
import re

# Define input and output directories
input_dir = "/home/sswarna/Documents/oran_docs/output_all/step2_chunked"
output_dir = "/home/sswarna/Documents/oran_docs/output_all/step3_embedded"
os.makedirs(output_dir, exist_ok=True)

# Define embedding model (available on server)
embedding_model = "nomic-embed-text"
print(f"Using embedding model: {embedding_model}")

# Ollama API endpoint
ollama_endpoint = "http://localhost:11434/api/embeddings"

# Maximum allowed chunk size for embedding (safe limit for API)
max_chunk_length = 1024
retry_chunk_length = 512

# Function to clean text
def clean_text(text):
    # Remove non-printable characters
    text = re.sub(r"[\x00-\x1F\x7F-\x9F]+", " ", text)
    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chunk_length]  # Ensure the text does not exceed the max length

# Function to generate embeddings for a chunk
def generate_embedding(chunk_data):
    chunk_index = chunk_data.get("chunk_index", -1)
    chunk_content = chunk_data.get("chunk_content", "")
    metadata = chunk_data.get("metadata", {})

    if not chunk_content.strip():
        print(f"Skipping chunk {chunk_index}: Empty content.")
        return None

    # Clean and truncate content
    cleaned_content = clean_text(chunk_content)

    try:
        # Using the Ollama API to generate embeddings
        response = requests.post(
            ollama_endpoint,
            json={"model": embedding_model, "prompt": cleaned_content}
        )
        response.raise_for_status()
        embedding = response.json().get("embedding", [])

        return {
            "chunk_index": chunk_index,
            "embedding": embedding,
            "metadata": metadata
        }

    except requests.exceptions.RequestException as e:
        print(f"Error embedding chunk {chunk_index}: {e}. Retrying with shorter content.")

        # Retry with shorter content
        truncated_content = cleaned_content[:retry_chunk_length]
        try:
            response = requests.post(
                ollama_endpoint,
                json={"model": embedding_model, "prompt": truncated_content}
            )
            response.raise_for_status()
            embedding = response.json().get("embedding", [])

            return {
                "chunk_index": chunk_index,
                "embedding": embedding,
                "metadata": metadata
            }
        except requests.exceptions.RequestException as retry_error:
            print(f"Failed again for chunk {chunk_index}: {retry_error}")
            return None

# Function to process a single document
def process_document(input_path, output_path, failed_chunks_path):
    with open(input_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Embedding {len(chunks)} chunks from {input_path}...")
    embedded_chunks = []
    failed_chunks = []

    for chunk in chunks:
        result = generate_embedding(chunk)
        if result:
            embedded_chunks.append(result)
        else:
            failed_chunks.append(chunk)

    # Save the results
    with open(output_path, "w", encoding="utf-8") as out_f:
        json.dump(embedded_chunks, out_f, ensure_ascii=False, indent=2)

    # Save failed chunks
    if failed_chunks:
        with open(failed_chunks_path, "w", encoding="utf-8") as failed_f:
            json.dump(failed_chunks, failed_f, ensure_ascii=False, indent=2)
        print(f"Saved {len(failed_chunks)} failed chunks to {failed_chunks_path}")

    print(f"Embedding completed for {input_path}. Total embedded chunks: {len(embedded_chunks)}")

# Process each year's chunked file
if __name__ == "__main__":
    for year in ['2022', '2023', '2024']:
        input_file = os.path.join(input_dir, f"{year}_chunked.json")
        output_file = os.path.join(output_dir, f"{year}_embedded.json")
        failed_chunks_file = os.path.join(output_dir, f"{year}_failed_chunks.json")

        if os.path.exists(input_file):
            try:
                process_document(input_file, output_file, failed_chunks_file)
            except Exception as e:
                print(f"Error processing {year}: {e}")
        else:
            print(f"Input file not found for {year}: {input_file}")
