from llm import get_llm

def decompose_question(question: str):
    llm = get_llm()

    prompt = f"""
    Break the following question into 2–4 smaller, independent sub-questions.
    Each sub-question should be answerable from documents.

    Question: {question}

    Return only the sub-questions, one per line.
    """

    response = llm(prompt)

    subs = [q.strip("- ").strip() for q in response.split("\n") if q.strip()]
    return subs
