import time
import re

class MemoryStore:
    """
    Session-based conversation history store with Topic Tracking.
    """
    def __init__(self):
        # Format: {session_id: {"history": [...], "profile": {"name": None, "current_topic": None}}}
        self.sessions = {}
        self.max_history = 5 # Context window limit

    def get_history(self, session_id: str):
        """Retrieve sliding window of conversation history."""
        if session_id not in self.sessions:
            return []
        
        history = self.sessions[session_id].get("history", [])
        return [{"role": m["role"], "content": m["content"]} for m in history[-self.max_history:]]

    def get_profile(self, session_id: str):
        """Retrieve user profile metadata."""
        if session_id not in self.sessions:
            return {}
        return self.sessions[session_id].get("profile", {})

    def add_message(self, session_id: str, role: str, content: str):
        """Add a message and track current topic."""
        if session_id not in self.sessions:
            self.sessions[session_id] = {"history": [], "profile": {"name": None, "current_topic": None}}
        
        if role == "user":
            content_lower = content.lower()
            # Advanced Topic Tracking (Keyword + Section based)
            if any(k in content_lower for k in ["theft", "stolen", "378", "379"]): 
                self.sessions[session_id]["profile"]["current_topic"] = "Theft"
            elif any(k in content_lower for k in ["murder", "homicide", "killing", "300", "302"]): 
                self.sessions[session_id]["profile"]["current_topic"] = "Homicide"
            elif any(k in content_lower for k in ["arrest", "police", "power"]): 
                self.sessions[session_id]["profile"]["current_topic"] = "Rights/Arrest"
            elif any(k in content_lower for k in ["cheat", "fraud", "420"]): 
                self.sessions[session_id]["profile"]["current_topic"] = "Fraud"
            elif any(k in content_lower for k in ["hacking", "cyber", "online", "computer", "internet"]):
                self.sessions[session_id]["profile"]["current_topic"] = "Cyber"
            
            # Name extraction
            name_match = re.search(r"my name is ([\w\s]+)", content_lower)
            if name_match:
                self.sessions[session_id]["profile"]["name"] = name_match.group(1).strip().capitalize()

        self.sessions[session_id]["history"].append({
            "role": role,
            "content": content,
            "timestamp": time.time()
        })
        
        # Keep history within bounds
        if len(self.sessions[session_id]["history"]) > self.max_history * 2:
            self.sessions[session_id]["history"] = self.sessions[session_id]["history"][-self.max_history * 2:]

    def clear_session(self, session_id: str):
        """Clear history for a specific session."""
        if session_id in self.sessions:
            del self.sessions[session_id]

# Singleton instance
memory_store = MemoryStore()
