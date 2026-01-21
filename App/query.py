from langchain_community.vectorstores import Weaviate
from langchain_community.embeddings import HuggingFaceEmbeddings
import weaviate
from llm import get_llm

def ask(question):
    client = weaviate.Client("http://localhost:8080")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = Weaviate(
        client,
        index_name="Docs",
        text_key="text",
        embedding=embeddings
    )

    retriever = db.as_retriever()
    docs = retriever.get_relevant_documents(question)

    context = "\n".join([d.page_content for d in docs])

    llm = get_llm()
    prompt = f"""
    Answer the question using only this context:

    {context}

    Question: {question}
    """

    answer = llm(prompt)
    print(answer)

if __name__ == "__main__":
    ask("What is this document about?")
