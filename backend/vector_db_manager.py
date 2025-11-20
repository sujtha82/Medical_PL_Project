import numpy as np
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from sentence_transformers import SentenceTransformer
import logging

class MilvusManager:
    def __init__(self, host='localhost', port='19530', collection_name="medical_claims"):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.collection = None
        self._connect()
        
    def _connect(self):
        """Connect to Milvus server"""
        try:
            connections.connect(alias="default", host=self.host, port=self.port)
            print(f"✅ Connected to Milvus at {self.host}:{self.port}")
            
            # Create collection if it doesn't exist
            if not utility.has_collection(self.collection_name):
                self._create_collection()
            else:
                self.collection = Collection(self.collection_name)
                
        except Exception as e:
            print(f"❌ Failed to connect to Milvus: {e}")
            raise e

    def _create_collection(self):
        """Create Milvus collection for medical claims"""
        # Define fields
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="claim_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="patient_name", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="provider", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="diagnosis", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="procedure", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="status", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=4000),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384)
        ]
        
        schema = CollectionSchema(fields=fields, description="Medical Claims Vector Database")
        self.collection = Collection(name=self.collection_name, schema=schema)
        
        # Create index for faster search
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128}
        }
        self.collection.create_index(field_name="embedding", index_params=index_params)
        print(f"✅ Created Milvus collection: {self.collection_name}")

    def insert_claims(self, documents, metadata_list=None):
        """Insert medical claims into Milvus"""
        if not documents:
            return
            
        # Generate embeddings
        embeddings = self.embedding_model.encode(documents, normalize_embeddings=True)
        
        # Prepare data for insertion
        entities = []
        
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            # Extract metadata if provided
            if metadata_list and i < len(metadata_list):
                meta = metadata_list[i]
            else:
                meta = {}
                
            entities.append({
                "claim_id": meta.get("claim_id", f"claim_{i}"),
                "patient_name": meta.get("patient_name", "Unknown"),
                "provider": meta.get("provider", "Unknown"),
                "diagnosis": meta.get("diagnosis", "Unknown"),
                "procedure": meta.get("procedure", "Unknown"),
                "status": meta.get("status", "Unknown"),
                "content": doc,
                "embedding": embedding.tolist()
            })
        
        # Insert into Milvus
        insert_result = self.collection.insert(entities)
        self.collection.flush()
        
        print(f"✅ Inserted {len(entities)} medical claims into Milvus")
        return insert_result

    def semantic_search(self, query, top_k=10, filters=None):
        """Semantic search with optional filters"""
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)
        
        # Search parameters
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10}
        }
        
        # Build filter expression
        expr = None
        if filters:
            filter_parts = []
            if filters.get("patient_name"):
                filter_parts.append(f'patient_name == "{filters["patient_name"]}"')
            if filters.get("provider"):
                filter_parts.append(f'provider == "{filters["provider"]}"')
            if filters.get("status"):
                filter_parts.append(f'status == "{filters["status"]}"')
            if filters.get("diagnosis"):
                filter_parts.append(f'diagnosis like "%{filters["diagnosis"]}%"')
            
            if filter_parts:
                expr = " and ".join(filter_parts)
        
        # Perform search
        results = self.collection.search(
            data=query_embedding.tolist(),
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["claim_id", "patient_name", "provider", "diagnosis", "procedure", "status", "content"]
        )
        
        # Format results
        formatted_results = []
        for hits in results:
            for hit in hits:
                formatted_results.append({
                    "id": hit.id,
                    "claim_id": hit.entity.get("claim_id"),
                    "patient_name": hit.entity.get("patient_name"),
                    "provider": hit.entity.get("provider"),
                    "diagnosis": hit.entity.get("diagnosis"),
                    "procedure": hit.entity.get("procedure"),
                    "status": hit.entity.get("status"),
                    "content": hit.entity.get("content"),
                    "distance": hit.distance,
                    "score": 1 - hit.distance  # Convert distance to similarity score
                })
        
        return formatted_results

    def hybrid_search(self, query, bm25_results, alpha=0.5, top_k=10, filters=None):
        """Combine Milvus semantic search with BM25 results"""
        # Get semantic results from Milvus
        semantic_results = self.semantic_search(query, top_k=top_k*2, filters=filters)
        
        # Create a mapping of content to semantic scores
        semantic_scores = {result["content"]: result["score"] for result in semantic_results}
        
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

    def get_collection_stats(self):
        """Get statistics about the collection"""
        if not self.collection:
            return None
            
        stats = {
            "total_claims": self.collection.num_entities,
            "collection_name": self.collection_name
        }
        return stats

    def close(self):
        """Close connection"""
        connections.disconnect("default")
        print("✅ Disconnected from Milvus")