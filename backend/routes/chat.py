from fastapi import APIRouter, Request
import json
import re

from backend.schemas.chat_schema import ChatRequest, ChatResponse
from backend.core.memory_store import memory_store
from backend.core.rag_engine import rag_engine
from backend.core.intent_classifier import intent_classifier
from backend.core.response_validator import response_validator
from backend.services.llm_service import llm_service
from backend.core.logger import audit_logger

router = APIRouter()

def expand_query(query: str, intents: list, topic: str = None) -> str:
    """Expands query with domain-specific legal terms and session topic."""
    # Check if user mentioned a specific section already
    has_specific_section = bool(re.search(r'\b\d+[A-Z]?\b', query))
    
    expanded = query
    if "PUNISHMENT" in intents:
        expanded += " penalty jail imprisonment fine sentence"
    if "LEGAL_DEFINITION" in intents:
        expanded += " definition meaning refers to constitutes"
    
    # Topic-based expansion - Only if no specific section is mentioned
    if topic and not has_specific_section:
        expanded += f" {topic}"
        if topic == "Theft": expanded += " ipc 378 379"
        elif topic == "Homicide": expanded += " ipc 300 302"
        elif topic == "Fraud": expanded += " ipc 420"
        elif topic == "Cyber": expanded += " it act hacking phishing"
    
    # Keyword based expansion (adds synonyms but avoid competing section numbers if possible)
    if "theft" in query.lower() and not has_specific_section: 
        expanded += " ipc 378 379"
    if "murder" in query.lower() and not has_specific_section: 
        expanded += " ipc 300 302"
    if "hacking" in query.lower() or "cyber" in query.lower():
        expanded += " it act 2000 section 66"
        
    return expanded

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, fast_request: Request):
    session_id = request.session_id
    user_message = request.message

    # 1. Multi-Label Intent Classification
    classification = intent_classifier.classify(user_message)
    intents = classification["intents"]
    
    # 2. Domain Intelligence: Query Expansion with Topic
    profile = memory_store.get_profile(session_id)
    expanded_query = expand_query(user_message, intents, topic=profile.get("current_topic"))
    print(f"DEBUG: Expanded Query: {expanded_query}")
    
    # 3. Context Retrieval (RAG) with Confidence Gate
    context, similarity_score = rag_engine.retrieve(expanded_query)
    print(f"DEBUG: Similarity Score: {similarity_score}")
    
    # Threshold for re-ranker score
    CONFIDENCE_THRESHOLD = -2.0 
    
    if similarity_score < CONFIDENCE_THRESHOLD:
        context = "No relevant legal context found in the provided database for this specific query."
    
    # 4. Conversation History
    history = memory_store.get_history(session_id)
    
    # 5. LLM Response Generation
    llm_output = await llm_service.generate_response(user_message, history, context, profile, matched_intents=intents)
    print(f"DEBUG: LLM Output: {llm_output}")
    
    # 6. Answer Verification Layer
    is_verified = True
    unverified_sections = []
    
    if context and "No relevant legal context" not in context:
        try:
            retrieved_json = json.loads(context)
            retrieved_sec_nums = [str(r.get("section")) for r in retrieved_json]
            
            cited_sections = llm_output.get("relevant_sections", [])
            for cite in cited_sections:
                cite_match = re.search(r'(\d+[A-Z]?)', cite)
                if cite_match:
                    sec_num = cite_match.group(1)
                    if sec_num not in retrieved_sec_nums:
                        is_verified = False
                        unverified_sections.append(cite)
        except:
            pass

    if not is_verified:
        llm_output["message"] = f"⚠️ [UNVERIFIED SOURCE]: The following citation(s) {unverified_sections} were not found in my immediate legal database and are based on internal knowledge. Please verify with a legal professional.\n\n" + llm_output["message"]

    # 7. Response Validation
    is_valid = response_validator.validate(llm_output, user_message)
    if not is_valid:
        llm_output = {
            "message": "I understand this is important. Let me double-check my information or escalate this for you.",
            "query_type": ["ERROR"],
            "relevant_sections": [],
            "extracted_issues": []
        }

    # 8. Update History & Topic
    memory_store.add_message(session_id, "user", user_message)
    memory_store.add_message(session_id, "assistant", llm_output["message"])

    # 9. Audit Logging
    try:
        retrieved_meta = json.loads(context) if context and "No relevant" not in context else []
    except:
        retrieved_meta = []
        
    audit_logger.log_interaction(
        query=user_message,
        retrieved_metadata=retrieved_meta,
        llm_response=llm_output,
        similarity_score=similarity_score,
        verified=is_verified,
        topic=profile.get("current_topic")
    )

    return ChatResponse(
        message=llm_output["message"],
        query_type=llm_output.get("query_type"),
        relevant_sections=llm_output.get("relevant_sections"),
        extracted_issues=llm_output.get("extracted_issues"),
        disclaimer=llm_output.get("disclaimer", "This is for informational purposes only and not legal advice.")
    )
