from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import faiss
import json
import numpy as np
import os

# Step 1: Load the Embedding Model
embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
print("Embedding model loaded successfully.")

# Load the Language Model
llm_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
if llm_tokenizer.pad_token is None:
    llm_tokenizer.pad_token = llm_tokenizer.eos_token  # Set pad_token to eos_token if not set
llm_model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
print("LLM model loaded successfully.")

# Step 2: Define Paths
vector_store_path = "/home/sswarna/Documents/oran_docs/output_all/step4_vector_store"
yearly_files = {
    "2022": {
        "metadata": os.path.join(vector_store_path, "2022_vector_store_metadata.json"),
        "faiss": os.path.join(vector_store_path, "2022_vector_store.faiss"),
    },
    "2023": {
        "metadata": os.path.join(vector_store_path, "2023_vector_store_metadata.json"),
        "faiss": os.path.join(vector_store_path, "2023_vector_store.faiss"),
    },
    "2024": {
        "metadata": os.path.join(vector_store_path, "2024_vector_store_metadata.json"),
        "faiss": os.path.join(vector_store_path, "2024_vector_store.faiss"),
    },
}

# Step 3: Load FAISS Indices and Metadata for All Years
def load_all_faiss_indices():
    indexes = []
    combined_metadata = []

    for year, paths in yearly_files.items():
        # Load FAISS index
        index = faiss.read_index(paths["faiss"])
        print(f"FAISS index for year {year} loaded successfully.")

        # Load metadata
        with open(paths["metadata"], "r") as f:
            metadata = json.load(f)
            print(f"Metadata for year {year} loaded successfully.")

        indexes.append(index)
        combined_metadata.extend(metadata)

    # Merge all indexes into one
    if indexes:
        d = indexes[0].d  # Dimension from the first index
        combined_index = faiss.IndexFlatL2(d)  # L2 distance metric
        for idx in indexes:
            xb = idx.reconstruct_n(0, idx.ntotal)  # Reconstruct all vectors
            combined_index.add(xb)  # Add vectors to combined index
        print("All FAISS indexes merged successfully.")
    else:
        raise ValueError("No FAISS indices found to merge.")

    return combined_index, combined_metadata

# Step 4: Embed Query
def embed_query(query):
    query_embedding = embedding_model.encode(query, convert_to_numpy=True)
    query_embedding = np.array(query_embedding).astype(np.float32).reshape(1, -1)  # Ensure correct shape
    print(f"Query Embedding Generated (Shape: {query_embedding.shape})")
    return query_embedding

# Step 5: Retrieve Relevant Documents
def retrieve_documents(query_embedding, index, metadata, top_k=5):
    if query_embedding.shape[1] != index.d:
        raise ValueError(
            f"Dimension mismatch: Query Embedding Dimension = {query_embedding.shape[1]}, FAISS Index Dimension = {index.d}"
        )

    # Search FAISS index
    distances, indices = index.search(query_embedding, top_k)
    print(f"Retrieved Indices: {indices}, Distances: {distances}")

    # Fetch Metadata
    retrieved_context = []
    for idx in indices[0]:
        if 0 <= idx < len(metadata):
            retrieved_context.append(metadata[idx])

    print("Retrieved Context:", retrieved_context)
    return retrieved_context

# Step 6: Answer the Question
def answer_question(context, question):
    # Combine retrieved context and question
    full_prompt = f"Context: {json.dumps(context)}\n\nQuestion: {question}\nAnswer: "
    print(f"Prompt sent to the model:\n{full_prompt}")

    # Tokenize the input
    inputs = llm_tokenizer(full_prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024)

    # Generate response
    outputs = llm_model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, max_new_tokens=150, num_return_sequences=1)
    response = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response

if __name__ == "__main__":
    try:
        # Load FAISS index and metadata for all years
        index, metadata = load_all_faiss_indices()

        # Interactive Loop
        print("\nInteractive Q&A Session. Type 'exit' to quit.\n")
        while True:
            question = input("Enter your question: ")
            if question.lower() == "exit":
                print("Exiting the Q&A session.")
                break

            # Embed the query
            query_embedding = embed_query(question)

            # Retrieve documents
            context = retrieve_documents(query_embedding, index, metadata)

            # Get the answer
            answer = answer_question(context, question)
            print("\nFinal Answer:", answer)
            print("\n---\n")

    except Exception as e:
        print(f"An error occurred: {e}")
