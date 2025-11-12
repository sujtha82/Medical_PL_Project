#  Hybrid Medical Claims RAG Project

This project is a Retrieval-Augmented Generation (RAG)-based system designed to process and answer questions from medical claims documents using LangChain, ChromaDB, and Sentence Transformers.  
It combines document chunking, semantic embeddings, and a conversational interface powered by Gradio.

---

##  Features

-  PDF and text document ingestion (via PyMuPDF)
-  Intelligent text chunking using `RecursiveCharacterTextSplitter` and `TokenTextSplitter`
-  Embedding generation using `sentence-transformers`
-  Vector storage using **ChromaDB** or **FAISS**
-  Query answering pipeline using LangChain retrievers
-  Interactive Gradio-based web UI
-  Support for BM25 ranking and hybrid retrieval



