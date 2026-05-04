from pydantic import BaseModel
from typing import Optional, Dict, Any

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    message: str
    query_type: Optional[list] = None
    relevant_sections: Optional[list] = None
    extracted_issues: Optional[list] = None
    disclaimer: str = "This is for informational purposes only and not legal advice."
