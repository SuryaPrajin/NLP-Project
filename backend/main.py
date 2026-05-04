import sys
import os
import site

# Ensure local user packages are found
user_site = site.getusersitepackages()
if user_site not in sys.path:
    sys.path.append(user_site)

from dotenv import load_dotenv

# Ensure logs are visible immediately and handle emojis/UTF-8
sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')

load_dotenv()

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from backend.routes import chat
from backend.core.rag_engine import rag_engine

# Warmup cache on startup
rag_engine.warmup()

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import json

app = FastAPI(title="Legal AI Assistant API")

# Initialize Rate Limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Admin Observability Stats
@app.get("/api/v1/admin/stats")
@limiter.limit("10/minute")
async def get_stats(request: Request):
    log_path = "backend/logs/chat_logs.json"
    if not os.path.exists(log_path):
        return {"message": "No data available yet."}
    
    with open(log_path, "r") as f:
        try:
            logs = json.load(f)
        except:
            logs = []
            
    total_chats = len(logs)
    escalation_count = sum(1 for log in logs if log.get("is_escalated", False))
    avg_sentiment = sum(log.get("sentiment", 0) for log in logs) / total_chats if total_chats > 0 else 0
    
    return {
        "total_sessions": total_chats,
        "escalation_rate": f"{(escalation_count/total_chats)*100:.1f}%" if total_chats > 0 else "0%",
        "average_sentiment": round(avg_sentiment, 2)
    }

# Feedback Endpoint (Evaluation Dataset)
@app.post("/api/v1/feedback")
async def save_feedback(data: dict):
    feedback_file = "backend/logs/feedback.json"
    try:
        feedback = []
        if os.path.exists(feedback_file):
            with open(feedback_file, "r") as f:
                feedback = json.load(f)
        feedback.append(data)
        with open(feedback_file, "w") as f:
            json.dump(feedback, f)
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Include routes
app.include_router(chat.router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "Legal AI Assistant API (Lawyer Bot) is online."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
