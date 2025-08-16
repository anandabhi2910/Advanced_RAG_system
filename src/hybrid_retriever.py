import os
import sys
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# Local utility for text processing
from text_processor import read_and_preprocess_documents

# --- Configuration ---
RAW_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')

def create_hybrid_retriever():
    """
    Creates a hybrid retriever that combines semantic search (FAISS)
    and keyword search (BM25).
    """
    print("Loading documents and creating hybrid retriever...")
    documents = read_and_preprocess_documents(RAW_DATA_PATH)
    if not documents:
        sys.exit("Failed to load documents. Exiting.")

    # Convert documents to a format compatible with LangChain's Document class
    from langchain.docstore.document import Document
    lc_documents = [Document(page_content=doc) for doc in documents]

    # Initialize the semantic retriever (FAISS)
    faiss_vectorstore = FAISS.from_documents(lc_documents, HuggingFaceEmbeddings())
    faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 2})

    # Initialize the keyword retriever (BM25)
    bm25_retriever = BM25Retriever.from_documents(lc_documents)
    bm25_retriever.k = 2

    # Create the hybrid retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=[0.5, 0.5] # Assign equal weight to both methods
    )
    
    print("Hybrid retriever created successfully.")
    return ensemble_retriever

if __name__ == "__main__":
    retriever = create_hybrid_retriever()
    
    query = "What is machine learning?"
    print(f"\nPerforming hybrid retrieval for query: '{query}'")
    
    retrieved_docs = retriever.get_relevant_documents(query)
    
    print("\n--- Retrieved Documents (Hybrid Search) ---")
    for doc in retrieved_docs:
        print(f"Content: {doc.page_content[:150]}...\n")