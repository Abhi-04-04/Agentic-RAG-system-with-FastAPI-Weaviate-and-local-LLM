Agentic Document Question Answering System

A Multimodal RAG Application using FastAPI, LangChain, Weaviate & Local
LLM

This project implements an agentic document question-answering
system  that allows users to upload documents and ask natural language
questions over them. The system supports **multimodal ingestion** (text,
PDF, and images), performs **semantic retrieval** using a vector
database, and generates answers using a **local Large Language Model
(LLM)**.

The core idea is to move beyond simple retrieval by introducing an
**agentic workflow** where complex user queries are decomposed into
smaller sub-questions, each retrieved independently, and finally
synthesized into a coherent answer.

This project was developed as part of an academic assignment to
demonstrate real-world RAG (Retrieval-Augmented Generation) architecture
and evaluation.

------------------------------------------------------------------------

 Key Features

-    **Document Ingestion**
    -   Supports `.txt`, `.pdf`, and image files (`.png`, `.jpg`)
    -   PDF parsing includes paragraphs and tables
    -   Images are processed using OCR
-    **Vector-Based Retrieval**
    -   Documents are chunked and embedded using Sentence Transformers
    -   Embeddings are stored in **Weaviate** for semantic search
-    **Agentic Query Processing**
    -   User queries are decomposed into multiple sub-questions
    -   Each sub-question triggers its own retrieval
    -   Retrieved contexts are merged and synthesized
-    **Local LLM Inference**
    -   Uses a local LLM (via Ollama)
    -   No external API calls or cloud dependency
-    **FastAPI Backend**
    -   `/ingest` → Upload and index documents\
    -   `/query` → Ask questions and receive answers
-    **Evaluation Ready**
    -   Designed to integrate with RAG evaluation frameworks such as
        RAGAS
    -   Supports metrics like context precision, recall, and answer
        relevance

------------------------------------------------------------------------

##  System Architecture (High Level)

    User
      │
      ▼
    FastAPI Backend
      │
      ├── /ingest
      │     └─ Multimodal Loader → Chunking → Embeddings → Weaviate
      │
      └── /query
            └─ Query Decomposition (LLM)
                   ├─ Sub-query 1 → Retrieve
                   ├─ Sub-query 2 → Retrieve
                   └─ Sub-query N → Retrieve
                             ↓
                     Context Aggregation
                             ↓
                     Local LLM Synthesis
                             ↓
                          Final Answer

------------------------------------------------------------------------

##  Project Structure

    docqa/
    │
    ├── app/
    │   ├── main.py        # FastAPI application
    │   ├── ingest.py      # Standalone ingestion script
    │   ├── loaders.py     # Multimodal file loaders (txt, pdf, image)
    │   ├── decompose.py   # Query decomposition logic
    │   └── llm.py         # Local LLM configuration
    │
    ├── data/
    │   └── sample.txt
    │
    ├── eval_data.json     # Evaluation dataset
    ├── requirements.txt
    └── README.md

------------------------------------------------------------------------

##  Getting Started

### 1. Install Dependencies

``` bash
pip install -r requirements.txt
```

Ensure that: - Weaviate is running locally - Ollama is installed and a
model (e.g., `mistral`) is available

------------------------------------------------------------------------

### 2. Run the API

``` bash
uvicorn app.main:app --reload
```

Open in browser:

    http://127.0.0.1:8000/docs

This provides an interactive Swagger UI.

------------------------------------------------------------------------

### 3. Use the System

1.  Upload a document using `/ingest`
2.  Ask a question using `/query`
3.  The system will:
    -   Decompose your query
    -   Retrieve relevant chunks for each sub-question
    -   Merge contexts
    -   Generate a final synthesized answer

------------------------------------------------------------------------

##  Learning Outcomes

Through this project, I learned:

-   How real-world RAG systems are designed end-to-end\
-   How to integrate vector databases with LLMs\
-   The importance of chunking and embeddings\
-   How agentic workflows improve retrieval for complex queries\
-   How to evaluate LLM-based systems using formal metrics\
-   How to build production-style APIs using FastAPI

This project bridges academic concepts with practical, industry-style
implementation.

------------------------------------------------------------------------

##  Future Enhancements

-   Frontend UI for end users\
-   Streaming responses\
-   Role-based document access\
-   Caching and performance optimization\
-   Support for video and audio inputs

------------------------------------------------------------------------

This repository demonstrates a complete, modular, and extensible
**Agentic RAG System** suitable for academic evaluation as well as
real-world experimentation.
