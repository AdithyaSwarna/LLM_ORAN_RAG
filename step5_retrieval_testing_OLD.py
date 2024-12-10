import os
import json
import requests
import faiss
import numpy as np

# Load FAISS index and metadata
def load_faiss_index(vector_store_dir):
    """
    Load FAISS indices and corresponding metadata files dynamically.

    :param vector_store_dir: Directory containing FAISS indices and metadata files
    :return: List of FAISS indices and a combined metadata dictionary
    """
    indices = []
    combined_metadata = {}

    for file in os.listdir(vector_store_dir):
        if file.endswith(".faiss"):
            base_name = file.replace("_vector_store.faiss", "")
            index_path = os.path.join(vector_store_dir, file)
            metadata_path = os.path.join(vector_store_dir, f"{base_name}_vector_store_metadata.json")

            if os.path.exists(metadata_path):
                # Load FAISS index
                index = faiss.read_index(index_path)
                indices.append(index)

                # Load metadata
                with open(metadata_path, "r") as f:
                    metadata_list = json.load(f)
                    # Ensure metadata_list is a list and process each entry
                    if isinstance(metadata_list, list):
                        for idx, entry in enumerate(metadata_list):
                            combined_metadata[f"{base_name}_{idx}"] = entry

    if not indices:
        raise FileNotFoundError("No FAISS index files found in the directory.")

    return indices, combined_metadata

# Retrieve documents from FAISS
def retrieve_documents(query, indices, metadata, top_k=5):
    """
    Retrieve top-k relevant documents from FAISS indices based on the query.

    :param query: The input query for retrieval
    :param indices: List of loaded FAISS indices
    :param metadata: Combined metadata dictionary
    :param top_k: Number of top results to retrieve
    :return: Retrieved document snippets as context
    """
    # Convert query to vector (simple embedding for demonstration, replace with actual model encoding)
    query_vector = np.random.rand(indices[0].d)  # Replace this with a real embedding model

    retrieved_contexts = []
    for index in indices:
        # Perform search in FAISS index
        distances, indices = index.search(np.array([query_vector], dtype=np.float32), top_k)

        # Retrieve corresponding metadata
        for idx in indices[0]:
            if idx != -1:  # Check for valid index
                retrieved_contexts.append(metadata.get(str(idx), ""))

    return "\n".join(retrieved_contexts)

# Query the model
def query_model(prompt, model_name, api_url):
    """
    Queries the model with a prompt and retrieves its response.

    :param prompt: The question or prompt to query the model
    :param model_name: The name of the model to use
    :param api_url: The base URL of the API
    :return: Processed response from the model
    """
    headers = {'Content-Type': 'application/json'}

    payload = {
        "model": model_name,
        "prompt": prompt,
        "temperature": 0.7,
        "stream": False
    }

    response = requests.post(
        f"{api_url}/api/generate",
        headers=headers,
        json=payload
    )

    if response.status_code == 200:
        return response.json()["response"]
    else:
        raise Exception(f"Error: {response.status_code}, {response.text}")

# Interactive QA loop
def interactive_qa(vector_store_dir, model_name, api_url):
    """
    Interactive question-answering loop.

    :param vector_store_dir: Path to the directory containing FAISS vector store
    :param model_name: The name of the model to use
    :param api_url: The base URL of the API
    """
    print("\nInteractive Question-Answering Session Started\n")
    print("Type your question below. Type 'exit' or '/bye' to quit.\n")

    # Load FAISS indices and metadata
    indices, metadata = load_faiss_index(vector_store_dir)

    while True:
        user_query = input("Your question: ")
        if user_query.lower() in ['exit', '/bye']:
            print("Exiting the session. Goodbye!")
            break

        try:
            # Retrieve relevant context from FAISS vector store
            context = retrieve_documents(user_query, indices, metadata)

            if not context.strip():
                print("\nNo relevant context found in the documents.\n")
                continue

            # Combine user query with retrieved context
            full_prompt = (
                f"You are a knowledgeable assistant. Use the following context "
                f"retrieved from O-RAN documents to answer the question.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {user_query}\nAnswer:"
            )

            # Query the model
            answer = query_model(full_prompt, model_name, api_url)

            # Display the answer
            print(f"\nModel Answer: {answer}\n")

        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    # Define paths
    vector_store_directory = "/home/sswarna/Documents/oran_docs/output_all/step4_vector_store"

    # Define model and API details
    model = "llama3.2:latest"
    api_base_url = "http://localhost:11434"

    # Start the interactive QA session
    interactive_qa(vector_store_directory, model, api_base_url)
