import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from backend.chunking_manager2 import ChunkingManager  # ✅ new import

# ------------------------------------------------------
# Load and Chunk Documents Dynamically
# ------------------------------------------------------
chunker = ChunkingManager()

# Load and combine data from PDF + TXT + CSV
pdf_text = chunker.load_document("data/kaiser_medical_claims.pdf")
txt_text = chunker.load_document("data/medical_claims.txt")

# Apply multiple chunking strategies
page_chunks = chunker.page_level_chunking("data/kaiser_medical_claims.pdf")
semantic_chunks = chunker.semantic_chunking(txt_text)
recursive_chunks = chunker.recursive_chunking(txt_text)
sliding_chunks = chunker.sliding_window_chunking(txt_text)
token_chunks = chunker.token_based_chunking(txt_text)

# ✅ Merge all chunks (hybrid document pool)
documents = list(set(page_chunks + semantic_chunks + recursive_chunks + sliding_chunks + token_chunks))

print(f"Total chunks loaded: {len(documents)}")

# ------------------------------------------------------
# BM25 Keyword Retriever
# ------------------------------------------------------
tokenized_docs = [doc.lower().split() for doc in documents]
bm25 = BM25Okapi(tokenized_docs)

# ------------------------------------------------------
# Dense Embedding Retriever
# ------------------------------------------------------
dense_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
doc_embeddings = dense_model.encode(documents, normalize_embeddings=True)

# ------------------------------------------------------
# Cross-Encoder Re-Ranker
# ------------------------------------------------------
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# ------------------------------------------------------
# Hybrid Retrieval Function
# ------------------------------------------------------
def hybrid_retrieve(query, alpha=0.5, top_k=6):
    tokenized_query = query.lower().split()
    bm25_scores = np.array(bm25.get_scores(tokenized_query))
    bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-6)

    query_embedding = dense_model.encode(query, normalize_embeddings=True)
    dense_scores = util.cos_sim(query_embedding, doc_embeddings)[0].cpu().numpy()
    dense_scores = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min() + 1e-6)

    hybrid_scores = alpha * bm25_scores + (1 - alpha) * dense_scores
    top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
    candidate_docs = [documents[i] for i in top_indices]
    return candidate_docs

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

try:
    import importlib
    PromptTemplate = None
    LLMChain = None
    HuggingFaceHub = None

    mod = importlib.import_module("langchain.prompts")
    PromptTemplate = getattr(mod, "PromptTemplate", None)
    mod = importlib.import_module("langchain.chains")
    LLMChain = getattr(mod, "LLMChain", None)
    mod = importlib.import_module("langchain.llms")
    HuggingFaceHub = getattr(mod, "HuggingFaceHub", None)

    if PromptTemplate and LLMChain and HuggingFaceHub:
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        llm = HuggingFaceHub(repo_id="google/flan-t5-base", model_kwargs={"temperature": 0.2, "max_length": 256})
        qa_chain = LLMChain(llm=llm, prompt=prompt)
    else:
        raise ImportError("LangChain components missing")

except Exception as e:
    print("⚠️ Using fallback LLM. Error:", e)

    class _FallbackChain:
        def run(self, inputs):
            context = inputs.get("context", "")
            question = inputs.get("question", "")
            truncated = context[:500] + ("..." if len(context) > 500 else "")
            return f"(Fallback) Based on retrieved context: {truncated}\n\nQuestion: {question}"

    qa_chain = _FallbackChain()

# ------------------------------------------------------
# QA Function (for Gradio)
# ------------------------------------------------------
def get_answer(query):
    candidates = hybrid_retrieve(query, alpha=0.6, top_k=6)
    reranked = rerank_with_crossencoder(query, candidates, top_k=3)
    context = "\n".join(reranked)
    response = qa_chain.run({"context": context, "question": query})
    return response