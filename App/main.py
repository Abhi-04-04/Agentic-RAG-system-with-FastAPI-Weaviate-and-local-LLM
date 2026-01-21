from fastapi import FastAPI, UploadFile, File
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Weaviate
from langchain.schema import Document
import weaviate
import os

from llm import get_llm
from loaders import load_file
from decompose import decompose_question

app = FastAPI()

WEAVIATE_URL = "http://localhost:8080"
INDEX_NAME = "Docs"


def get_db():
    client = weaviate.Client(WEAVIATE_URL)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return Weaviate(
        client,
        index_name=INDEX_NAME,
        text_key="text",
        embedding=embeddings,
    )


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    os.makedirs("uploads", exist_ok=True)
    path = f"uploads/{file.filename}"

    with open(path, "wb") as f:
        f.write(await file.read())

    # Multimodal loading (txt, pdf, image)
    texts = load_file(path)

    docs = [Document(page_content=t, metadata={"source": file.filename}) for t in texts]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)

    client = weaviate.Client(WEAVIATE_URL)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    Weaviate.from_documents(
        chunks,
        embeddings,
        client=client,
        index_name=INDEX_NAME
    )

    return {"status": "success", "chunks": len(chunks)}


@app.post("/query")
async def query(question: str):
    db = get_db()
    retriever = db.as_retriever()

    # 1. Decompose
    sub_questions = decompose_question(question)

    all_docs = []

    # 2. Retrieve for each sub-question
    for sq in sub_questions:
        docs = retriever.get_relevant_documents(sq)
        all_docs.extend(docs)

    # Remove duplicates
    unique_texts = list(dict.fromkeys([d.page_content for d in all_docs]))
    context = "\n".join(unique_texts)

    # 3. Final synthesis
    llm = get_llm()
    prompt = f"""
    You are given context from multiple documents.

    Context:
    {context}

    Original Question:
    {question}

    Provide a clear, well-structured answer by synthesizing all relevant points.
    """

    answer = llm(prompt)

    return {
        "question": question,
        "sub_questions": sub_questions,
        "answer": answer,
        "chunks_used": len(unique_texts)
    }
