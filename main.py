import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from backend.chunking_manager2 import ChunkingManager
from backend.simple_vectordb_manager import SimpleVectorManager  # ✅ Fixed typo: simple_vectordb_manager → simple_vector_manager

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ✅ Initialize Vector Database
print("🚀 Initializing Vector Database...")
vector_manager = SimpleVectorManager()  # ✅ Using SimpleVectorManager instead of Milvus

# ------------------------------------------------------
# Load and Process Documents
# ------------------------------------------------------
chunker = ChunkingManager()

# Load documents
pdf_text = chunker.load_document("data/kaiser_medical_claims.pdf")
txt_text = chunker.load_document("data/medical_claims.txt")
csv_text = chunker.load_document("data/kaiser_medical_claims.csv")

# Apply chunking strategies
page_chunks = chunker.page_level_chunking("data/kaiser_medical_claims.pdf")
semantic_chunks = chunker.semantic_chunking(txt_text)
recursive_chunks = chunker.recursive_chunking(txt_text)
sliding_chunks = chunker.sliding_window_chunking(txt_text)
token_chunks = chunker.token_based_chunking(txt_text)

# Merge all chunks
documents = list(set(page_chunks + semantic_chunks + recursive_chunks + sliding_chunks + token_chunks))

print(f"📄 Total chunks loaded: {len(documents)}")

# ✅ Insert documents into Vector Database
print("💾 Storing documents in Vector Database...")
vector_manager.insert_claims(documents)  # ✅ Updated to vector_manager
print(f"✅ Vector Database stats: {vector_manager.get_collection_stats()}")  # ✅ Updated to vector_manager

# ------------------------------------------------------
# BM25 Keyword Retriever (Keep for hybrid search)
# ------------------------------------------------------
tokenized_docs = [doc.lower().split() for doc in documents]
bm25 = BM25Okapi(tokenized_docs)

# ------------------------------------------------------
# Cross-Encoder Re-Ranker
# ------------------------------------------------------
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# ------------------------------------------------------
# Enhanced Hybrid Retrieval with Vector Database
# ------------------------------------------------------
def hybrid_retrieve(query, alpha=0.6, top_k=6, filters=None):
    """Enhanced hybrid retrieval using Vector Database + BM25"""
    
    # Get BM25 results
    tokenized_query = query.lower().split()
    bm25_scores = np.array(bm25.get_scores(tokenized_query))
    bm25_indices = np.argsort(bm25_scores)[::-1][:top_k*2]
    bm25_docs = [documents[i] for i in bm25_indices]
    
    # Use Vector Database for hybrid search
    candidate_docs = vector_manager.hybrid_search(  # ✅ Updated to vector_manager
        query=query,
        bm25_results=bm25_docs,
        alpha=alpha,
        top_k=top_k
        # Note: SimpleVectorManager doesn't support filters yet
    )
    
    print(f"🔍 Hybrid search found {len(candidate_docs)} candidates")
    return candidate_docs

# ------------------------------------------------------
# Enhanced Retrieval (Simple version without filters)
# ------------------------------------------------------
def retrieve_semantic(query, top_k=5):
    """Retrieve claims using semantic search only"""
    return vector_manager.semantic_search(query, top_k=top_k)  # ✅ Updated to vector_manager

# ------------------------------------------------------
# LLM Setup (LangChain or fallback)
# ------------------------------------------------------
prompt_template = """
You are a medical claims assistant.
Use the following retrieved context to answer the user's question clearly and accurately.

Context:
{context}

Question:
{question}

Answer:
"""

print("🔧 Starting LLM setup...")

try:
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    from langchain.llms import HuggingFaceHub
    
    print("✅ All LangChain imports successful!")
    
    # Check if API token is loaded from .env
    api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not api_token:
        print("❌ HUGGINGFACEHUB_API_TOKEN not found in .env file")
        raise Exception("Missing HuggingFace API token in .env")
    
    print(f"🔑 API Token: {api_token[:10]}...")
    
    prompt = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question"]
    )
    print("✅ PromptTemplate created successfully!")
    
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",
        model_kwargs={
            "temperature": 0.2,
            "max_length": 512
        }
    )
    print("✅ HuggingFaceHub LLM initialized!")
    
    qa_chain = LLMChain(llm=llm, prompt=prompt)
    print("✅ LangChain LLM setup successful!")
    
except Exception as e:
    print(f"❌ LangChain setup failed: {e}")
    print("⚠️ Using fallback LLM")

    class _FallbackChain:
        def run(self, inputs):
            context = inputs.get("context", "")
            question = inputs.get("question", "")
            return f"(Fallback) Based on retrieved context: {context[:300]}...\n\nQuestion: {question}"

    qa_chain = _FallbackChain()

# ------------------------------------------------------
# Re-ranking Function
# ------------------------------------------------------
def rerank_with_crossencoder(query, candidate_docs, top_k=3):
    pairs = [(query, doc) for doc in candidate_docs]
    scores = cross_encoder.predict(pairs)
    reranked_indices = np.argsort(scores)[::-1][:top_k]
    reranked_docs = [candidate_docs[i] for i in reranked_indices]
    return reranked_docs

# ------------------------------------------------------
# QA Function (for Gradio)
# ------------------------------------------------------
def get_answer(query):
    print(f"🔍 Processing query: '{query}'")
    
    candidates = hybrid_retrieve(query, alpha=0.6, top_k=6)
    print(f"📄 Retrieved {len(candidates)} candidate chunks")
    
    reranked = rerank_with_crossencoder(query, candidates, top_k=3)
    print(f"🎯 Reranked to {len(reranked)} chunks")
    
    context = "\n".join(reranked)
    print(f"📝 Context length: {len(context)} characters")
    
    try:
        print("🤖 Calling LLM...")
        response = qa_chain.run({"context": context, "question": query})
        print(f"✅ LLM Response generated: {response[:100]}...")
        return response
    except Exception as e:
        print(f"❌ LLM Error: {e}")
        return f"Error: {str(e)}"
    
    """import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from backend.chunking_manager2 import ChunkingManager
from backend.simple_vectordb_manager import SimpleVectorManager

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ✅ Initialize Milvus
print("🚀 Initializing Milvus vector database...")
#milvus_manager = MilvusManager(host='localhost', port='19530')
vector_manager = SimpleVectorManager()

# ------------------------------------------------------
# Load and Process Documents
# ------------------------------------------------------
chunker = ChunkingManager()

# Load documents
pdf_text = chunker.load_document("data/kaiser_medical_claims.pdf")
txt_text = chunker.load_document("data/medical_claims.txt")
csv_text = chunker.load_document("data/kaiser_medical_claims.csv")

# Apply chunking strategies
page_chunks = chunker.page_level_chunking("data/kaiser_medical_claims.pdf")
semantic_chunks = chunker.semantic_chunking(txt_text)
recursive_chunks = chunker.recursive_chunking(txt_text)
sliding_chunks = chunker.sliding_window_chunking(txt_text)
token_chunks = chunker.token_based_chunking(txt_text)

# Merge all chunks
documents = list(set(page_chunks + semantic_chunks + recursive_chunks + sliding_chunks + token_chunks))

print(f"📄 Total chunks loaded: {len(documents)}")

# ✅ Insert documents into Milvus
#print("💾 Storing documents in Milvus...")
#milvus_manager.insert_claims(documents)
vector_manager.insert_claims(documents)
print(f"✅ Milvus collection stats: {milvus_manager.get_collection_stats()}")

# ------------------------------------------------------
# BM25 Keyword Retriever (Keep for hybrid search)
# ------------------------------------------------------
tokenized_docs = [doc.lower().split() for doc in documents]
bm25 = BM25Okapi(tokenized_docs)

# ------------------------------------------------------
# Cross-Encoder Re-Ranker
# ------------------------------------------------------
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# ------------------------------------------------------
# Enhanced Hybrid Retrieval with Milvus
# ------------------------------------------------------
def hybrid_retrieve(query, alpha=0.6, top_k=6, filters=None):
  
    filters = {}
    if patient_name:
        filters["patient_name"] = patient_name
    if provider:
        filters["provider"] = provider  
    if status:
        filters["status"] = status
        
    return milvus_manager.semantic_search(query, top_k=top_k, filters=filters)

# Keep the rest of your existing code (LLM setup, reranking, etc.)
# ... [your existing LLM setup and other functions] ..."""