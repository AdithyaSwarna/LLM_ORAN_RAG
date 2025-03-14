#!/bin/bash
echo "🚀 Starting the O-RAN RAG Pipeline Execution... 🔥"

BASE_DIR="/home/sswarna/Documents/oran_docs/oran_rag_pipeline"

echo "🔹 Running Step 1 & 2: Document Loading & Chunking..."
python3 $BASE_DIR/step1_step2_document_loading_chunking.py && echo "✅ Step 1 & 2 Completed!"

echo "🔹 Running Step 3: Document Embedding..."
python3 $BASE_DIR/step3_document_embedding.py && echo "✅ Step 3 Completed!"

echo "🔹 Running Step 4: Vector Store Creation..."
python3 $BASE_DIR/step4_vector_store.py && echo "✅ Step 4 Completed!"

echo "🔹 Running Step 5: Retrieval Testing..."
python3 $BASE_DIR/step5_retrieval.py && echo "✅ Step 5 Completed!"

echo "🎉 O-RAN RAG Pipeline Execution Completed Successfully! 🚀"
