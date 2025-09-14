# RAG-based Factual Chatbot with Data Ingestion


## Overview

This project implements a complete LangChain pipeline that leverages pre-trained models from Hugging Face. It ingests a research document (PDF, Word, or plain text), extracts meaningful facts, generates natural question-answer pairs, augments them for paraphrase robustness, and builds a semantic retrieval backend so that a chatbot can answer user queries even when they are paraphrased or reworded. It is designed to run end-to-end in Google Colab but can be adapted for production/chatbot backends.

### Key Capabilities
- Document normalization and cleaning (headers/footers removal, hyphenation fixing, quote normalization)
- Sentence-level decomposition
- Hybrid fact extraction and question generation
- Paraphrase augmentation for question diversity
- Semantic embedding of questions using models from HuggingFace
- Fast retrieval (FAISS) with cross-encoder reranking for precision
- Interactive querying demonstrating robustness to paraphrased user questions
- Orchestration of the entire RAG pipeline using the LangChain framework, connecting document loaders, text splitters, embedding models, and vector stores into a seamless workflow.

## Components / Pipeline Description

### 1. Document Ingestion & Cleaning
- Supports `.pdf`, `.docx`, and `.txt`.
- Loads the document using LangChain's loaders.
- Cleans the text by:
  - Fixing hyphenations.
  - Normalizing whitespace and quotes.
  - Detecting and stripping repeated header/footer lines.

### 2. Sentence-Level Breakdown
- Uses spaCy to split cleaned text into individual sentences.
- Assigns each sentence a unique ID for traceability.

### 3. Hybrid Fact & Question Generation
- Chunks nearby sentences to create local context windows.
- Summarizes each chunk into a concise “fact” using a summarization model(`facebook/bart-large-cnn`).
- Generates a natural question for each fact using a question-generation model (`valhalla/t5-base-qg-hl`), overwriting the original placeholder question.

### 4. Paraphrase Augmentation 
- For each generated question, produces paraphrased variants using a paraphrasing model (e.g., `ramsrigouthamg/t5_paraphraser`).
- These paraphrases are treated as alternative question surface forms pointing to the same answer, improving robustness to varied user phrasing.

### 5. Semantic Store Construction
- Embeds all canonical questions (and optionally their paraphrases) using a sentence embedding model (`sentence-transformers/all-MiniLM-L6-v2`).
- Builds a FAISS index over these embeddings to allow fast semantic similarity search (cosine similarity via normalized inner product).
- Persists both the index and the QA store for reuse.

### 6. Retrieval with Reranking
- Incoming user query is embedded into the same semantic space.
- FAISS retrieves top candidate stored questions.
- A cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) rescoring refines the best match by comparing the user query to candidate questions.
- The top match’s answer (and optional metadata) is returned, even if the user question is paraphrased.

### 7. Interactive Demo / Query Loop
- Provides a prompt-based loop for testing: type arbitrary questions and see which canonical question is matched, the answer, context snippet, and confidence scores.

### 8. Output Files and Their Purpose

Running this project will generate the following files:

| Filename                     | Description |
|-----------------------------|-------------|
| `raw_data.json`             | Contains the cleaned and normalized raw text extracted from the uploaded document. This includes removal of headers, footers, hyphenations, inconsistent quotes, and extra whitespace. |
| `processed_data.json`       | A sentence-level breakdown of the document. Each sentence is tagged with its paragraph ID and sentence ID. This structured format is used to generate question–answer pairs. |
| `qa_with_generated_questions.json` | The final list of generated QA pairs. Each entry includes a question, answer (fact), context (surrounding text), and a reference to its source. This serves as the core factual knowledge base for the chatbot. |
| `faiss_question.index`      | The FAISS vector index of question embeddings. It enables fast and efficient semantic similarity search between user queries and stored questions. |
| `qa_store.pkl`              | A serialized Python object storing metadata, including the list of QA pairs and the embedding model used. This allows reloading the chatbot without reprocessing the document. |

These files support efficient document-based question answering and make the chatbot persistent across sessions.


## Installation & Setup (Colab-ready)

This is meant to be run in **Google Colab**, but can be adapted elsewhere with an appropriate GPU and Python environment.

> **P.S.: After all the installations, you must restart the Colab kernel.**  
> Don’t worry if you see an error after restarting the kernel — just **comment out the installation lines**, and rerun the notebook.

