from backend.chunking_manager2 import ChunkingManager

manager = ChunkingManager()

# Test PDF
pdf_chunks = manager.page_level_chunking("data/kaiser_medical_claims.pdf")
print("\n🔹 PDF Page-level Chunks:")
for i, chunk in enumerate(pdf_chunks):
    print(f"\nChunk {i+1}:\n{chunk[:300]}...")  # show first 300 chars

# Test Recursive Chunking
joined_text = "\n".join(pdf_chunks)
recursive_chunks = manager.recursive_chunking(joined_text)
print(f"\n🔹 Total Recursive Chunks: {len(recursive_chunks)}")
print(recursive_chunks[0][:500])

# Test Semantic Chunking
semantic_chunks = manager.semantic_chunking(joined_text)
print(f"\n🔹 Total Semantic Chunks: {len(semantic_chunks)}")
