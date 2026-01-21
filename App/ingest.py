from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Weaviate
from langchain.schema import Document
import weaviate

from loaders import load_file


def ingest(path):
    # Load any file type (txt, pdf, image)
    texts = load_file(path)

    # Convert to LangChain Documents
    docs = [Document(page_content=t, metadata={"source": path}) for t in texts]

    # Chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Weaviate
    client = weaviate.Client("http://localhost:8080")

    Weaviate.from_documents(
        chunks,
        embeddings,
        client=client,
        index_name="Docs"
    )

    print(f"Ingestion complete! Chunks created: {len(chunks)}")


if __name__ == "__main__":
    ingest("data/sample.txt")   # you can change this to a PDF or image
