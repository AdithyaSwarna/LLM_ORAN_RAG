# LLM_ORAN_RAG
# Retrieval-Based Question Answering for O-RAN Documents

This repository contains code, resources, and outputs for processing, analyzing, and utilizing O-RAN specifications using Large Language Models (LLMs). The goal of this project is to develop a pipeline for extracting, summarizing, and querying information from O-RAN specification documents, enabling efficient insights and applications.

---

## Repository Structure

### Code/
Contains Python scripts for building and executing the O-RAN LLM pipeline. Key files include:
- `step1_document_loading.py`: Handles the loading and preprocessing of O-RAN specification documents (PDFs and Word files).
- `step2_document_chunking.py`: Splits the preprocessed documents into manageable chunks for embedding.
- `step3_document_embedding.py`: Generates vector embeddings for document chunks.
- `step4_vector_store.py`: Stores vector embeddings and metadata using FAISS for efficient similarity searches.
- `step5_retrieval_testing.py`: Tests retrieval and querying capabilities using the FAISS index.
- `Pipeline1` and `Pipeline2`: Visualization images of the pipeline.

### Output Files/
Contains the processed output of each step, organized by year:
- **Output_2022/**
- **Output_2023/**
- **Output_2024/**
- **step2_chunked/**: Chunks created from documents.
- **step3_embedded/**: Generated embeddings for the document chunks.
- **step4_vector_store/**: FAISS vector stores created for similarity searching.

### Key PNGs
- `Output_Check1.png` and `Output_Check2.png`: Visual checks for validation of the pipeline outputs.

---

## Features

### 1. Metadata and Text Extraction
Extracts structured metadata and clean text from O-RAN specification documents in PDF and Word formats.

### 2. Document Chunking
Splits documents into smaller chunks with metadata for efficient processing.

### 3. Embedding Generation
Generates vector embeddings for the document chunks using transformer-based models.

### 4. Vector Store
Stores the embeddings and metadata in a FAISS index for fast and accurate retrieval.

### 5. Retrieval Testing
Implements question-answering over O-RAN documents using the FAISS index and a language model.

---

## Prerequisites

- **Python 3.8+**
- Required Libraries:
  - `faiss`
  - `transformers`
  - `PyMuPDF`
  - `python-docx`

---

## Usage

### Step 1: Document Loading
Run `step1_document_loading.py` to preprocess and load O-RAN specification documents.

### Step 2: Document Chunking
Run `step2_document_chunking.py` to chunk documents into smaller manageable parts.

### Step 3: Embedding Generation
Run `step3_document_embedding.py` to create vector embeddings for the chunks.

### Step 4: Vector Store
Run `step4_vector_store.py` to create FAISS indices for the embeddings.

### Step 5: Retrieval Testing
Run `step5_retrieval_testing.py` to test retrieval and question-answering on the documents.

---
## Input Files / Dataset
https://drive.google.com/file/d/1IJHSt6wzksOw-2Pp5YyE_uShxrG3O6nT/view?usp=sharing

---
## Outputs
Processed outputs are saved in the **Output Files/** directory. Use the latest version files for accurate results.

---
## Output Files Link
https://drive.google.com/file/d/1FPor8zpnXYHCUE6oyRmdgCp05BtZK4QZ/view?usp=sharing
---
