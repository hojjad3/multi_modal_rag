import hashlib
import uuid

from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams


def load_data(path):
    loader = TextLoader(path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=12)
    chunks = text_splitter.split_documents(documents)
    return chunks


def compute_doc_hash(doc_content):
    return hashlib.sha256(doc_content.encode()).hexdigest()


def hashing(chunks):
    unique_chunks = []
    ids = []
    for chunk in chunks:
        chunk_hash = compute_doc_hash(chunk.page_content)
        chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_hash))
        chunk.metadata["hash"] = chunk_hash
        unique_chunks.append(chunk)
        ids.append(chunk_id)
    return unique_chunks, ids


def create_collection(client, collection_name):
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )
        print(f"Collection '{collection_name}' created.")
    else:
        print(f"Collection '{collection_name}' already exists.")


def build_vector_store(model_embeddings, collection_name, client):
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=model_embeddings,
    )
    return vector_store


def add_documents(vector_store, unique_chunks, ids):
    vector_store.add_documents(documents=unique_chunks, ids=ids)


if __name__ == "__main__":
    path = "responses/gemini_generated_image_dkbrzgdkbrzgdkbr.md"
    model_name = "nomic-embed-text"
    client = QdrantClient(path="./qdrant_storage")
    collection_name = "finance_data"
    model_embeddings = OllamaEmbeddings(model=model_name)

    chunks = load_data(path)
    unique_chunks, ids = hashing(chunks)
    create_collection(client, collection_name)
    vector_store = build_vector_store(model_embeddings, collection_name, client)
    add_documents(vector_store, unique_chunks, ids)
    client.close()
