from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaEmbeddings
from qdrant_client import QdrantClient

from src.vector_store import build_vector_store

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


if __name__ == "__main__":
    path = "responses/gemini_generated_image_dkbrzgdkbrzgdkbr.md"
    model_name = "nomic-embed-text"
    client = QdrantClient(path="./qdrant_storage")
    collection_name = "finance_data"
    model_embeddings = OllamaEmbeddings(model=model_name)
    vector_store = build_vector_store(model_embeddings, collection_name, client)

    tools = [retrieve_context]
    prompt = (
        "You have access to a tool that retrieves context from a financial document. "
        "Use the tool to help answer user queries."
    )
    agent = create_agent(model, tools, system_prompt=prompt)
    query = "what is the number of employees in 2022?"
    for event in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="values",
    ):
        event["messages"][-1].pretty_print()
    client.close()
