import os
import json
from math import ceil

# Define parameters
default_chunk_size = 1200
chunk_overlap = 150

# Define input and output directories
input_dir = "/home/sswarna/Documents/oran_docs/output_all"
output_dir = os.path.join(input_dir, "step2_chunked")
os.makedirs(output_dir, exist_ok=True)

# Function to dynamically adjust chunk size
def adjust_chunk_size(text_length, default_size):
    # Adjust chunk size for very short or very long documents
    if text_length < default_size:
        return max(300, text_length // 3)  # Minimum chunk size of 300
    elif text_length > default_size * 10:
        return default_size * 2  # Increase chunk size for very large documents
    return default_size

# Function to chunk text
def chunk_text(text, size, overlap):
    chunks = []
    text_length = len(text)
    start = 0
    while start < text_length:
        end = min(start + size, text_length)
        chunks.append(text[start:end])
        start = start + size - overlap
    return chunks

# Function to process and chunk documents
def process_and_chunk_documents(input_folder, output_file):
    try:
        all_chunks = []
        skipped_documents = []
        doc_count = 0

        for file_name in os.listdir(input_folder):
            if file_name.endswith("_text.json"):
                doc_count += 1
                text_file_path = os.path.join(input_folder, file_name)

                with open(text_file_path, "r", encoding="utf-8") as f:
                    text_data = json.load(f)

                # Combine all text pages into a single string
                text = "\n".join([v for k, v in text_data.items() if isinstance(v, str)])

                # Check for empty documents
                text_length = len(text.strip())
                if text_length == 0:
                    skipped_documents.append(file_name)
                    continue

                # Adjust chunk size based on document length
                chunk_size = adjust_chunk_size(text_length, default_chunk_size)

                print(f"Processing Document {doc_count}: {file_name} (length={text_length}, chunk_size={chunk_size})")

                # Chunk the text
                chunks = chunk_text(text, chunk_size, chunk_overlap)
                for idx, chunk in enumerate(chunks):
                    metadata_file = file_name.replace("_text.json", "_metadata.json")
                    metadata_path = os.path.join(input_folder, metadata_file)

                    metadata = {}
                    if os.path.exists(metadata_path):
                        with open(metadata_path, "r", encoding="utf-8") as meta_f:
                            metadata = json.load(meta_f)

                    # Add chunk with metadata
                    all_chunks.append({
                        "chunk_index": idx,
                        "chunk_content": chunk,
                        "metadata": metadata
                    })

                print(f"Document {doc_count}: {len(chunks)} chunks created.")

        # Save all chunks to the output file
        with open(output_file, "w", encoding="utf-8") as out_f:
            json.dump(all_chunks, out_f, ensure_ascii=False, indent=2)

        print(f"Chunking completed. Total chunks: {len(all_chunks)}. Saved to {output_file}")

        # Log skipped documents
        if skipped_documents:
            print(f"Skipped {len(skipped_documents)} documents due to empty content: {skipped_documents}")

    except Exception as e:
        print(f"Error processing folder {input_folder}: {e}")

# Process each year
for year in ['2022', '2023', '2024']:
    input_folder = os.path.join(input_dir, f"Output_{year}")
    output_file = os.path.join(output_dir, f"{year}_chunked.json")

    print(f"\nProcessing {year} documents...")
    process_and_chunk_documents(input_folder, output_file)
