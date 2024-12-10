import faiss
import os

# Define paths
vector_store_dir = "/home/sswarna/Documents/oran_docs/output_all/step4_vector_store"

# Function to verify FAISS index
def verify_faiss_index(file_path):
    try:
        index = faiss.read_index(file_path)
        print(f"FAISS index loaded successfully from {file_path}")
        print(f"Number of vectors: {index.ntotal}")
        print(f"Vector dimension: {index.d}")
    except Exception as e:
        print(f"Error verifying FAISS index at {file_path}: {e}")

# List of vector store files
faiss_files = [
    os.path.join(vector_store_dir, "2022_vector_store.faiss"),
    os.path.join(vector_store_dir, "2023_vector_store.faiss"),
    os.path.join(vector_store_dir, "2024_vector_store.faiss"),
]

# Verify each FAISS file
for faiss_file in faiss_files:
    verify_faiss_index(faiss_file)
