"# rag-with-bedrock-claude-llama3-faiss" 

An end-to-end Retrieval-Augmented Generation (RAG) system using AWS Bedrock and FAISS.
This project demonstrates how to ingest PDF documents, split them into chunks, create embeddings using Amazon Titan via Bedrock, and store them in a FAISS vector database. At query time, the system retrieves semantically relevant document chunks based on similarity search, constructs a prompt, and uses a Large Language Model (LLM) from AWS Bedrock (e.g., Claude or LLaMA 3) to generate contextual answers.
It’s a complete architecture for document-based question answering built with LangChain and Streamlit.

PDF ingestion and preprocessing

Text chunking using RecursiveCharacterTextSplitter

Embedding generation with Amazon Titan (via Bedrock)

FAISS-based vector store

Semantic search for retrieving relevant chunks

Prompt construction and LLM response via Bedrock

Streamlit frontend for user interaction


