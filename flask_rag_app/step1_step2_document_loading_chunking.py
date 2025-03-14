import os
import json
import re
import fitz  # PyMuPDF
import docx
from tqdm import tqdm
import tiktoken  # <-- Added for token-based splitting

# === Configuration ===
OUTPUT_BASE_DIR = "/home/sswarna/Documents/oran_docs/output_all"
CHUNKS_OUTPUT_BASE_DIR = os.path.join(OUTPUT_BASE_DIR, "Step2_chunks")
os.makedirs(CHUNKS_OUTPUT_BASE_DIR, exist_ok=True)

# === Function to clean text (remove excessive dots) ===
def clean_text(text):
    return re.sub(r"\.\.{5,}", "", text)  # Remove sequences of 5 or more dots

# === Function to extract text from PDF ===
def extract_text_from_pdf(pdf_path):
    text = ""
    metadata = {}
    try:
        doc = fitz.open(pdf_path)
        metadata = doc.metadata  # Extract metadata
        for page in doc:
            text += page.get_text("text") + "\n"
        text = clean_text(text)
    except Exception as e:
        print(f"❌ Error processing PDF: {pdf_path} - {e}")
    return text, metadata

# === Function to extract text from DOCX ===
def extract_text_from_docx(docx_path):
    text = ""
    try:
        doc = docx.Document(docx_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
        text = clean_text(text)
    except Exception as e:
        print(f"❌ Error processing DOCX: {docx_path} - {e}")
    return text

# === Token-based chunking function ===
def adaptive_chunking(text, chunk_size=512, overlap=100, encoding_name="gpt2"):
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)

    chunks = []
    index = 0
    chunk_index = 0
    
    while index < len(tokens):
        end = min(index + chunk_size, len(tokens))
        chunk_tokens = tokens[index:end]
        chunk_text = encoding.decode(chunk_tokens)
        
        chunks.append({
            "chunk_index": chunk_index,
            "chunk_content": chunk_text
        })
        
        chunk_index += 1
        index += (chunk_size - overlap)
        if index <= 0:
            break
    
    return chunks

# === Process only the uploaded file ===
def process_uploaded_file(uploaded_file_path):
    """Processes only the uploaded file and extracts chunks."""
    
    if not os.path.exists(uploaded_file_path):
        print(f"⚠️ File not found: {uploaded_file_path}")
        return

    filename = os.path.basename(uploaded_file_path)
    file_base_name, file_extension = os.path.splitext(filename)

    print(f"\n🔹 Processing Uploaded File: {filename}\n")

    # 1. Extract text and metadata
    if file_extension.lower() == ".pdf":
        text, metadata = extract_text_from_pdf(uploaded_file_path)
    elif file_extension.lower() == ".docx":
        text = extract_text_from_docx(uploaded_file_path)
        metadata = {"format": "DOCX"}
    else:
        print(f"⚠️ Unsupported file format: {filename}")
        return

    if not text.strip():
        print(f"❌ Skipping empty text file: {filename}")
        return

    metadata["filename"] = file_base_name
    metadata["title"] = file_base_name  # <-- Adding title in metadata

    # 2. Save full text and metadata
    text_output_path = os.path.join(OUTPUT_BASE_DIR, f"{file_base_name}_text.json")
    with open(text_output_path, "w", encoding="utf-8") as f:
        json.dump({"title": file_base_name, "text": text, "metadata": metadata}, f, indent=4)

    # 3. Create token-based chunks
    chunks = adaptive_chunking(text, chunk_size=512, overlap=100)

    # 4. Save chunks in a JSON file (split if > 5000)
    chunk_output_path = os.path.join(CHUNKS_OUTPUT_BASE_DIR, f"{file_base_name}_chunks.json")

    if len(chunks) > 5000:
        for i in range(0, len(chunks), 2000):
            part_num = (i // 2000) + 1
            part_path = chunk_output_path.replace("_chunks.json", f"_chunks_part{part_num}.json")
            with open(part_path, "w", encoding="utf-8") as f:
                json.dump({"title": file_base_name, "chunks": chunks[i:i + 2000]}, f, indent=4)
        print(f"✅ Processed & Split: {filename} | {len(chunks)} chunks → Multiple files")
    else:
        with open(chunk_output_path, "w", encoding="utf-8") as f:
            json.dump({"title": file_base_name, "chunks": chunks}, f, indent=4)
        print(f"✅ Processed: {filename} | {len(chunks)} chunks created")
