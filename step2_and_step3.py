import os
import json
import logging
import subprocess

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define parameters for chunking
chunk_size = 800
chunk_overlap = 150

def chunk_text(text, size, overlap):
    """
    Chunk text into overlapping segments.
    """
    chunks = []
    text_length = len(text)
    start = 0
    while start < text_length:
        end = min(start + size, text_length)
        chunks.append(text[start:end])
        start = start + size - overlap
    return chunks

def generate_embeddings(chunk, retries=3):
    """
    Generate embeddings for a text chunk using the Ollama CLI, with retries for robustness.
    """
    chunk = chunk.replace("\n", " ").strip()  # Clean the chunk
    logging.info(f"Embedding chunk of length {len(chunk)}")

    for attempt in range(retries):
        try:
            process = subprocess.run(
                ["ollama", "embed", "--model", "nomic-embed-text", "--text", chunk],
                text=True,
                capture_output=True,
                check=True
            )
            response = json.loads(process.stdout)
            return response.get("embedding")
        except subprocess.CalledProcessError as e:
            logging.error(f"Attempt {attempt + 1} failed: {str(e)}. Chunk: {chunk[:200]}")
            if attempt + 1 == retries:
                return None

def process_and_chunk_documents(input_dir, output_dir):
    """
    Process JSON text files, chunk the text, and generate embeddings.
    """
    os.makedirs(output_dir, exist_ok=True)

    for file_name in os.listdir(input_dir):
        if file_name.endswith("_text.json"):
            input_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, file_name.replace("_text.json", "_embeddings.json"))

            try:
                # Load text data
                with open(input_path, "r", encoding="utf-8") as f:
                    text_data = json.load(f)

                # Combine all text pages into a single string
                text = "\n".join([v for k, v in text_data.items() if isinstance(v, str)])

                # Chunk the text
                chunks = chunk_text(text, chunk_size, chunk_overlap)

                # Generate embeddings for each chunk
                all_chunks = []
                for idx, chunk in enumerate(chunks):
                    embedding = generate_embeddings(chunk)
                    if embedding:
                        all_chunks.append({
                            "chunk_index": idx,
                            "embedding": embedding,
                            "chunk_content": chunk
                        })

                # Save embeddings
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(all_chunks, f, ensure_ascii=False, indent=2)

                logging.info(f"Processed and chunked embeddings for: {file_name}")

            except Exception as e:
                logging.error(f"Error processing {file_name}: {str(e)}")

def main():
    input_base_dir = "/home/sswarna/Documents/oran_docs/output_all"
    output_base_dir = "/home/sswarna/Documents/oran_docs/output_all_embeddings"

    for year in ["2022", "2023", "2024"]:
        year_input_dir = os.path.join(input_base_dir, f"Output_{year}")
        year_output_dir = os.path.join(output_base_dir, f"Embeddings_{year}")

        if not os.path.exists(year_input_dir):
            logging.warning(f"Input directory for {year} does not exist.")
            continue

        process_and_chunk_documents(year_input_dir, year_output_dir)

if __name__ == "__main__":
    main()
