import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np

class SimpleVectorManager:
    def __init__(self, collection_name="medical_claims"):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(collection_name)
        self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        print("✅ Simple Vector Manager initialized with ChromaDB")
        
    def insert_claims(self, documents):
        """Insert documents into vector database"""
        if not documents:
            return
            
        print(f"📊 Generating embeddings for {len(documents)} documents...")
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(documents).tolist()
        
        # Insert into ChromaDB
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            ids=[f"doc_{i}" for i in range(len(documents))]
        )
        print(f"✅ Inserted {len(documents)} documents into vector database")
        
    def semantic_search(self, query, top_k=5):
        """Semantic search using ChromaDB"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k
            )
            return results['documents'][0]
        except Exception as e:
            print(f"❌ Semantic search error: {e}")
            return []
    
    def hybrid_search(self, query, bm25_results, alpha=0.5, top_k=10):
        """Combine ChromaDB semantic search with BM25"""
        try:
            # Get semantic results
            semantic_results = self.semantic_search(query, top_k=top_k*2)
            
            if not semantic_results:
                print("⚠️ No semantic results, using BM25 only")
                return bm25_results[:top_k]
            
            # Create score mapping
            semantic_scores = {}
            for i, doc in enumerate(semantic_results):
                semantic_scores[doc] = (len(semantic_results) - i) / len(semantic_results)
            
            # Combine scores
            combined_results = []
            for i, bm25_doc in enumerate(bm25_results):
                semantic_score = semantic_scores.get(bm25_doc, 0.0)
                bm25_score = 1.0 / (i + 1)  # Simple BM25 score approximation
                
                combined_score = alpha * bm25_score + (1 - alpha) * semantic_score
                combined_results.append({
                    "content": bm25_doc,
                    "bm25_score": bm25_score,
                    "semantic_score": semantic_score,
                    "combined_score": combined_score
                })
            
            # Sort by combined score and return top_k
            combined_results.sort(key=lambda x: x["combined_score"], reverse=True)
            return [result["content"] for result in combined_results[:top_k]]
            
        except Exception as e:
            print(f"❌ Hybrid search error: {e}")
            return bm25_results[:top_k]

    def get_collection_stats(self):
        """Get collection statistics"""
        try:
            return {"total_documents": self.collection.count()}
        except:
            return {"total_documents": 0}
        