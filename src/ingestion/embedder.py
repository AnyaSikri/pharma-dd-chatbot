import hashlib
import logging
import chromadb
from openai import OpenAI

EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 100

logger = logging.getLogger(__name__)


class Embedder:
    def __init__(self, openai_api_key: str, chroma_path: str = "./chroma_db"):
        self._openai = OpenAI(api_key=openai_api_key)
        self._chroma = chromadb.PersistentClient(path=chroma_path)

    def embed_query(self, text: str) -> list[float]:
        response = self._openai.embeddings.create(
            model=EMBEDDING_MODEL, input=[text],
        )
        return response.data[0].embedding

    def embed_and_store(self, chunks: list, collection_name: str) -> None:
        if not chunks:
            return
        collection = self._chroma.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )
        all_embeddings = []
        texts = [c["text"] for c in chunks]
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            try:
                response = self._openai.embeddings.create(model=EMBEDDING_MODEL, input=batch)
                all_embeddings.extend([item.embedding for item in response.data])
            except Exception as e:
                logger.error("OpenAI embedding API call failed: %s", e)
                raise RuntimeError(f"Failed to generate embeddings: {e}") from e
        ids = [hashlib.md5(chunk["text"].encode()).hexdigest() for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        collection.upsert(ids=ids, documents=texts, embeddings=all_embeddings, metadatas=metadatas)

    def get_collection(self, collection_name: str):
        return self._chroma.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )
