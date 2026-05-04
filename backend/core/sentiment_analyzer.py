

class SentimentAnalyzer:
    """
    Analyzes user sentiment to drive escalation logic.
    Scores range from -1.0 (Very Frustrated) to 1.0 (Happy).
    """
    def __init__(self):
        self.negative_keywords = [
            "frustrated", "angry", "annoyed", "useless", "scam", 
            "horrible", "awful", "wait too long", "still not",
            "legal", "lawyer", "police", "report you"
        ]

    def analyze(self, text: str) -> float:
        """
        Simple rule-based sentiment scoring.
        Can be upgraded to LLM-based scoring for better accuracy.
        """
        text_lower = text.lower()
        score = 0.0
        
        # Keyword checks
        for word in self.negative_keywords:
            if word in text_lower:
                score -= 0.3
        
        # Intensity checks (Caps, exclamation marks)
        if text.isupper():
            score -= 0.4
        
        if "!" in text:
            score -= 0.1
            
        # Clamp score between -1 and 1
        return max(-1.0, min(1.0, score))

# Singleton
sentiment_analyzer = SentimentAnalyzer()
