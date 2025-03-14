import os
import json
import time
import requests
import numpy as np
import faiss
import pandas as pd
from prettytable import PrettyTable
from sentence_transformers import SentenceTransformer, util
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from chromadb import PersistentClient

# === Configuration ===
CHROMA_DB_DIR = "/home/sswarna/Documents/oran_docs/oran_rag_pipeline/chroma_index"
COLLECTION_NAME = "oran_docs"
EMBEDDING_MODEL_PATH = "/home/sswarna/models/all-MiniLM-L12-v2"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama2:7b"
TOP_K = 50  # Increased from 5 to 50

# Load local embedding model
embed_model = SentenceTransformer(EMBEDDING_MODEL_PATH)

# Initialize ChromaDB
chroma_client = PersistentClient(path=CHROMA_DB_DIR)
collection = chroma_client.get_collection(COLLECTION_NAME)

# === Test Queries and Expected Answers ===
test_queries = {
    "Factual Questions (Basic Retrieval)": [
        "What are the security measures in O-RAN?",
        "Describe the architecture of O-RAN Near-RT RIC."
    ],
    "Comparative Questions (Complex Retrieval)": [
        "Compare the roles of Near-RT RIC and Non-RT RIC in O-RAN.",
        "How does O-RAN differ from traditional RAN architectures?"
    ],
    "Summarization Questions": [
        "Summarize the O-RAN security framework.",
        "Provide a high-level overview of O-RAN architecture."
    ],
    "Ambiguous Queries (Robustness Check)": [
        "Tell me about RIC.",
        "What’s new in O-RAN?"
    ],
    "Misleading Queries (Hallucination Prevention)": [
        "Does O-RAN use blockchain for security?",
        "What is the O-RAN 6G standard?"
    ]
}

expected_answers = {
    "Factual Questions (Basic Retrieval)": [
        "O-RAN security includes authentication, encryption, access control mechanisms, and network slicing protection.",
        "O-RAN Near-RT RIC provides real-time control of RAN functions via xApps."
    ],
    "Comparative Questions (Complex Retrieval)": [
        "Near-RT RIC handles real-time optimizations, whereas Non-RT RIC focuses on AI-driven long-term optimization.",
        "O-RAN decouples hardware and software, enabling interoperability."
    ],
    "Summarization Questions": [
        "The O-RAN security framework includes authentication, encryption, and network slicing protection.",
        "O-RAN consists of Near-RT RIC, Non-RT RIC, O-CU, O-DU, and O-RU."
    ],
    "Ambiguous Queries (Robustness Check)": [
        "RIC consists of Near-RT RIC for real-time decision-making and Non-RT RIC for policy-driven, ML-based optimization.",
        "O-RAN updates frequently, focusing on AI-driven automation and security."
    ],
    "Misleading Queries (Hallucination Prevention)": [
        "No, O-RAN does not use blockchain for security.",
        "There is no official O-RAN 6G standard."
    ]
}

# === Evaluation Metrics Functions ===
def compute_bleu(reference, generated):
    """Compute BLEU score with smoothing for short responses."""
    smooth = SmoothingFunction().method1
    return sentence_bleu([reference.split()], generated.split(), smoothing_function=smooth)

def compute_rouge(reference, generated):
    """Compute ROUGE-1 score."""
    rouge = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    scores = rouge.score(reference, generated)
    return scores["rouge1"].fmeasure

def compute_semantic_similarity(references, generations):
    """Compute cosine similarity between generated and reference answers using local embeddings."""
    ref_embeddings = embed_model.encode(references, convert_to_tensor=True)
    gen_embeddings = embed_model.encode(generations, convert_to_tensor=True)
    return util.cos_sim(gen_embeddings, ref_embeddings).mean().item()

def knowledge_grounding_score(retrieved, generated):
    """Check if generated content aligns with retrieved context."""
    retrieved_embeds = embed_model.encode(retrieved, convert_to_tensor=True)
    generated_embeds = embed_model.encode(generated, convert_to_tensor=True)
    return util.cos_sim(generated_embeds, retrieved_embeds).mean().item()

# === Improved Retrieval Function ===
def retrieve_relevant_chunks(query, top_k=TOP_K):
    """Retrieve relevant document chunks and remove noise (tables, lists)."""
    query_embedding = embed_model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    raw_chunks = [results["documents"][0][i] for i in range(len(results["documents"][0]))]

    # Remove irrelevant metadata (tables, figures, list items)
    cleaned_chunks = [chunk for chunk in raw_chunks if not any(tag in chunk.lower() for tag in ["table", "figure", "list", "appendix"])]
    
    return cleaned_chunks if cleaned_chunks else raw_chunks  # Fallback if everything is filtered

# === Improved LLM Query Function ===
def call_ollama_llm(query, retrieved_chunks):
    """Calls Ollama LLM API to generate an answer using retrieved context."""
    context = "\n".join(retrieved_chunks)

    # Improved prompt for **precise & concise** answers
    llm_prompt = f"""
    ### Instructions for LLM:
    - **Strictly answer based on retrieved context only.**
    - **Keep responses under 50 words.**
    - **Use exact document phrasing where applicable.**
    
    ### Retrieved Context:
    {context}

    ### Query:
    {query}
    """

    response = requests.post(OLLAMA_URL, json={"model": OLLAMA_MODEL, "prompt": llm_prompt, "stream": False})
    return response.json().get("response", "⚠️ No response from Ollama.") if response.status_code == 200 else "Error"

# === Run Evaluation ===
results = []
table_data = []

for category, queries in test_queries.items():
    print(f"\n🔍 Evaluating Category: {category}\n")
    for i, query in enumerate(queries):
        retrieved_chunks = retrieve_relevant_chunks(query)
        generated_answer = call_ollama_llm(query, retrieved_chunks)
        reference_answer = expected_answers[category][i]

        # Compute Evaluation Metrics
        bleu = compute_bleu(reference_answer, generated_answer)
        rouge = compute_rouge(reference_answer, generated_answer)
        semantic_similarity = compute_semantic_similarity([reference_answer], [generated_answer])
        kg_score = knowledge_grounding_score([" ".join(retrieved_chunks)], generated_answer)

        # Store Results
        results.append({
            "category": category,
            "query": query,
            "generated_answer": generated_answer,
            "BLEU": bleu,
            "ROUGE-1": rouge,
            "Semantic Similarity": semantic_similarity,
            "KG Score": kg_score
        })

        # Store Table Data
        table_data.append([category, query, f"{bleu:.4f}", f"{rouge:.4f}", f"{semantic_similarity:.4f}", f"{kg_score:.4f}"])

# Save results
with open("rag_evaluation_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)

# === Print Summary Table ===
summary_table = PrettyTable()
summary_table.field_names = ["Category", "Query", "BLEU", "ROUGE-1", "Semantic Similarity", "KG Score"]
for row in table_data:
    summary_table.add_row(row)

print("\n📊 **Final Evaluation Summary** 📊")
print(summary_table)
print("\n🎯 Evaluation Complete! Results saved to rag_evaluation_results.json")
