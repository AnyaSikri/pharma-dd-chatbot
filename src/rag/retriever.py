from src.ingestion.embedder import Embedder, EMBEDDING_MODEL


class Retriever:
    def __init__(self, embedder: Embedder):
        self._embedder = embedder

    def retrieve_for_report(self, collection_name: str, company: str) -> list:
        collection = self._embedder.get_collection(collection_name)
        results = collection.get(
            include=["documents", "metadatas"],
        )
        chunks = []
        for doc, meta in zip(results["documents"], results["metadatas"]):
            chunks.append({"text": doc, "metadata": meta})
        return chunks

    def retrieve_for_chat(self, collection_name: str, query: str, n_results: int = 10) -> list:
        collection = self._embedder.get_collection(collection_name)
        query_embedding = self._embedder.embed_query(query)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0],
        ):
            chunks.append({"text": doc, "metadata": meta, "distance": dist})
        return chunks
