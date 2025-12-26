from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

def create_hybrid_retriever(documents, vectorstore):
    """
    Combines keyword (BM25) + vector (FAISS) retrievers
    """

    # Keyword retriever
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 4

    # Vector retriever
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # Hybrid retriever (weighted)
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.4, 0.6]  # vector slightly more important
    )

    return hybrid_retriever
