import json
import os
from datetime import datetime

class AuditLogger:
    def __init__(self, log_path="backend/logs/audit.jsonl"):
        self.log_path = log_path
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def log_interaction(self, query, retrieved_metadata, llm_response, similarity_score, verified=True, topic=None):
        """Log a single chat interaction with evaluation metrics."""
        # Calculate retrieval 'Hit' - did we find what we were looking for?
        # Simple heuristic: if similarity score is high (above 1.0 for cross-encoder), it's a hit.
        is_hit = similarity_score > 1.0
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "topic": topic,
            "retrieved_count": len(retrieved_metadata) if isinstance(retrieved_metadata, list) else 0,
            "retrieved_sections": [
                {"sec": r.get("section"), "act": r.get("act")} for r in retrieved_metadata
            ] if isinstance(retrieved_metadata, list) else [],
            "metrics": {
                "relevance_score": float(round(similarity_score, 3)),
                "is_retrieval_hit": bool(is_hit),
                "is_verified": bool(verified)
            },
            "llm_response": {
                "query_type": llm_response.get("query_type"),
                "relevant_sections": llm_response.get("relevant_sections")
            }
        }
        
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            print(f"Error logging interaction: {e}")

# Singleton
audit_logger = AuditLogger()
