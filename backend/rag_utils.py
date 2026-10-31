
from langchain_openai import ChatOpenAI
import os

# Retrieve the top-k relevant chunks from the vector store
def retrieve(query: str, k: int = 2,vectorstore=None) -> list[str]:
    results = vectorstore.similarity_search(query, k=k)

    # Return each chunk with its content and metadata
    return [
        f"Content:\n{r.page_content}\n\nMetadata:\n{r.metadata}"
        for r in results
    ]

# Define the generation function that prompts the LLM with retrieved context
def generate(query, retrieved_chunks,model) -> str:

    # Same rules as in the notebook, so the deployed system behaves like the evaluated one
    rules = """
    1. Use only information explicitly stated in the retrieved policy information. Do not use outside knowledge, assumptions, or typical hotel practice.
    2. Answer the question that was asked, directly and in the first sentence, then give the supporting details. Do not include unrelated information.
    3. Include every condition, exception, deadline, fee amount, limit, and approval requirement that affects the answer (for example local property time, "subject to availability", manager approval, or the reservation confirmation taking priority).
    4. Property-specific details (check-in/check-out times, parking, pet fees, breakfast) apply only to the property named in the question. Never transfer a fee or rule from one property to another. If the question names a property that does not appear in the retrieved information, say that it is not covered and advise checking the live property profile or contacting the property team.
    5. If the question needs information from more than one policy section or document, combine it and answer every part of the question.
    6. Do not promise refunds, fee waivers, exceptions, or upgrades. When a manager or Finance approval is required, say so.
    7. Quote amounts, times, and limits exactly as written in the retrieved information.
    8. Keep the answer concise and clear for hotel staff. Use a short list only when the answer has several parts.
    9. If the retrieved information is insufficient, say: "The available policy documents do not contain enough information to answer this question." and briefly state what is missing. Do not guess. If the retrieved information conflicts, point out the conflict.
    """

    prompt = f"""
    You are an AI assistant answering questions about hotel policies.

    User Query:
    {query}

    Retrieved Policy Information:
    {retrieved_chunks}

    Rules:
    {rules}

    Answer:
    """

    return model.invoke(prompt)

# Define the full RAG pipeline combining retrieval and generation
def rag(
    query: str,
    k: int = 2,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
    vectorstore=None
):
    # Create the generator model
    model = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        openai_api_base=os.getenv('OPENAI_API_BASE')
    )

    # Retrieve relevant chunks
    retrieved_chunks = retrieve(query=query, k=k,vectorstore=vectorstore)

    # Generate answer using retrieved chunks
    answer = generate(
        query=query,
        retrieved_chunks=retrieved_chunks,
        model=model
    )

    return answer.content, retrieved_chunks
