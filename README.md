# Legal AI Assistant (Antigravity-Legal)

A domain-specific, law-aware AI assistant for Indian Criminal Law.

## Features
- **Hybrid RAG**: Semantic (FAISS) + Keyword (BM25) search with Cross-Encoder re-ranking.
- **Custom Transformer**: From-scratch PyTorch implementation of a legal reasoning model.
- **BPE Tokenizer**: Optimized for legal terminology.
- **Grounding**: Strict verification against official law documents (IPC, CrPC, IT Act).

## Setup
1. **Backend**:
   ```powershell
   pip install -r backend/requirements.txt
   python backend/main.py
   ```
2. **Frontend**:
   ```powershell
   cd frontend
   npm install
   npm run dev
   ```
