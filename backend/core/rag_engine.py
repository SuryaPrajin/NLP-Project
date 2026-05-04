import faiss
import json
import os
import numpy as np
import PyPDF2
import re
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from backend.services.embedding_service import embedding_service

class RAGEngine:
    def __init__(self):
        self.index = None
        self.chunks = [] # Stores {'text': str, 'source': str, ...}
        self.bm25 = None
        self.dimension = 384  # MiniLM embedding dimension
        self.law_books_path = "ML/Criminal Law"
        
        # Load re-ranker on demand or at init
        print("DEBUG: Initializing Re-ranker (MS-MARCO-MiniLM)...")
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
        
        self._initialize_index()

    def _extract_text_from_pdf(self, pdf_path):
        """Extract text from a PDF file."""
        text = ""
        try:
            with open(pdf_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    content = page.extract_text()
                    if content:
                        text += content + "\n"
        except Exception as e:
            print(f"Error reading {pdf_path}: {e}")
        return text

    def _chunk_text(self, text, source_name, chunk_size=1500, overlap=150):
        """Split text into strictly section-bound chunks with precise metadata."""
        text = text.replace('\x00', '').replace('\ufffd', '')
        
        # Determine act name
        act_name = source_name.replace(".pdf", "").replace("_", " ").replace("-", " ").lower()
        if "penal" in act_name: act_name = "IPC"
        elif "evidence" in act_name or "a187209" in act_name: act_name = "Indian Evidence Act"
        elif "procedure" in act_name: act_name = "CrPC"
        elif "it act" in act_name or "information technology" in act_name: act_name = "IT Act"
        elif "case laws" in act_name: act_name = "Case Law"
        elif "faq" in act_name: act_name = "FAQ"
        else: act_name = act_name.title()

        chunks = []
        
        # Find all section boundaries. 
        pattern = r'\n(\d+[A-Z]?)\.\s+'
        matches = list(re.finditer(pattern, text))
        
        if not matches:
            # Fallback to fixed size if no sections found
            for i in range(0, len(text), chunk_size - overlap):
                chunks.append({
                    "section_number": "Unknown",
                    "act_name": act_name,
                    "title": "General",
                    "content": text[i:i+chunk_size],
                    "source": source_name,
                    "topic": "General"
                })
            return chunks

        for i, match in enumerate(matches):
            section_num = match.group(1)
            start_idx = match.end()
            end_idx = matches[i+1].start() if i + 1 < len(matches) else len(text)
            
            section_content = text[start_idx:end_idx].strip()
            
            # Extract title
            title_match = re.search(r'^([^\n.]+)', section_content)
            title = title_match.group(1).strip() if title_match else "General Provision"
            
            # Topic detection (simple keyword based)
            topic = "General"
            if any(k in section_content.lower() for k in ["theft", "robbery", "burglary"]): topic = "Theft/Robbery"
            elif any(k in section_content.lower() for k in ["murder", "homicide", "killing"]): topic = "Homicide"
            elif any(k in section_content.lower() for k in ["assault", "hurt", "violence"]): topic = "Physical Offense"
            elif any(k in section_content.lower() for k in ["fraud", "cheat", "deceit"]): topic = "Fraud"

            full_text = f"{section_num}. {section_content}"
            
            # Split large sections
            part = 1
            curr = 0
            while curr < len(full_text):
                chunk_text = full_text[curr:curr+chunk_size]
                part_title = title if part == 1 else f"{title} (Part {part})"
                
                chunks.append({
                    "section_number": section_num,
                    "act_name": act_name,
                    "title": part_title,
                    "content": chunk_text.strip(),
                    "source": source_name,
                    "topic": topic
                })
                curr += chunk_size - overlap
                part += 1
                
        return chunks

    def _initialize_index(self):
        """Load law books, chunk them, and build both FAISS and BM25 indices."""
        print("Initializing Legal Hybrid RAG Index...")
        all_chunks = []
        
        if os.path.exists(self.law_books_path):
            for file in os.listdir(self.law_books_path):
                if file.endswith(".pdf"):
                    path = os.path.join(self.law_books_path, file)
                    print(f"Processing {file}...")
                    text = self._extract_text_from_pdf(path)
                    file_chunks = self._chunk_text(text, file)
                    all_chunks.extend(file_chunks)
        
        if not all_chunks:
            print("No content found to index.")
            return

        self.chunks = all_chunks
        
        # 1. Build FAISS Index (Semantic)
        self.index = faiss.IndexFlatL2(self.dimension)
        embeddings = []
        for i, chunk in enumerate(self.chunks):
            if i % 200 == 0: print(f"Embedding chunk {i}/{len(self.chunks)}...")
            emb = embedding_service.get_embedding(chunk["content"])
            embeddings.append(emb)
        
        if embeddings:
            self.index.add(np.vstack(embeddings))
            print(f"FAISS Index built with {self.index.ntotal} chunks.")

        # 2. Build BM25 Index (Keyword)
        tokenized_corpus = [chunk["content"].lower().split() for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print("BM25 Index built successfully.")

    def retrieve(self, query: str, k: int = 4, top_n_rerank: int = 50):
        """Hybrid search (FAISS + BM25) with Cross-Encoder Re-ranking."""
        if not self.chunks:
            return "No legal context available.", 0.0

        # 1. Semantic Retrieval (Top N)
        query_vector = embedding_service.get_query_embedding(query)
        faiss_distances, faiss_indices = self.index.search(np.array([query_vector]), top_n_rerank)
        
        semantic_results = []
        for i, idx in enumerate(faiss_indices[0]):
            if idx != -1:
                semantic_results.append(self.chunks[idx])

        # 2. Keyword Retrieval (Top N)
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_top_indices = np.argsort(bm25_scores)[-top_n_rerank:][::-1]
        
        keyword_results = []
        for idx in bm25_top_indices:
            if bm25_scores[idx] > 0:
                keyword_results.append(self.chunks[idx])

        # 3. Combine and Deduplicate
        seen_ids = set()
        combined_candidates = []
        for chunk in keyword_results + semantic_results:
            chunk_id = f"{chunk['act_name']}_{chunk['section_number']}_{chunk['content'][:50]}"
            if chunk_id not in seen_ids:
                combined_candidates.append(chunk)
                seen_ids.add(chunk_id)

        # 4. Re-ranking (Cross-Encoder)
        if not combined_candidates:
            return "[]", 0.0

        pairs = [[query, c["content"]] for c in combined_candidates]
        rerank_scores = self.reranker.predict(pairs)
        
        # Sort candidates by re-ranker score
        for i, score in enumerate(rerank_scores):
            combined_candidates[i]["rerank_score"] = float(score)

        # 5. Domain Filtering (Aggressive Metadata Booster)
        query_lower = query.lower()
        query_numbers = re.findall(r'\b\d+[a-zA-Z]?\b', query)
        
        target_act = None
        if "ipc" in query_lower or "penal code" in query_lower: target_act = "IPC"
        elif "crpc" in query_lower or "procedure" in query_lower: target_act = "CrPC"
        elif "evidence" in query_lower: target_act = "Indian Evidence Act"
        elif "it act" in query_lower or "cyber" in query_lower: target_act = "IT Act"
        elif "case law" in query_lower or "precedent" in query_lower: target_act = "Case Law"

        for chunk in combined_candidates:
            # Boost specific section match
            if chunk["section_number"] in query_numbers:
                chunk["rerank_score"] += 2.0
            
            # Boost specific act match
            if target_act and chunk["act_name"] == target_act:
                chunk["rerank_score"] += 2.0
            
            # Ultra boost for exact Act + Section match
            if target_act and chunk["act_name"] == target_act and chunk["section_number"] in query_numbers:
                chunk["rerank_score"] += 10.0 # Absolute priority
        
        final_sorted = sorted(combined_candidates, key=lambda x: x["rerank_score"], reverse=True)
        top_k = final_sorted[:k]

        # Format output
        structured_results = []
        for chunk in top_k:
            structured_results.append({
                "act": chunk.get('act_name', 'Unknown'),
                "section": chunk.get('section_number', 'Unknown'),
                "title": chunk.get('title', 'General'),
                "content": chunk['content'][:1500]
            })
            
        json_context = json.dumps(structured_results, indent=2)
        avg_score = np.mean([c["rerank_score"] for c in top_k]) if top_k else 0.0
        
        return json_context, avg_score

    def warmup(self):
        """Pre-initialize the index on startup."""
        if not self.chunks:
            self._initialize_index()

# Singleton
rag_engine = RAGEngine()
