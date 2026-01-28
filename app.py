from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
from datetime import datetime
import json
import uuid

from ollama import Client
from playbook import get_playbook, __version__ as playbook_version
from reflector_agent import run_reflector_analysis
from summarizer_agent import generate_summary
from curator_agent import run_curator, semantic_deduplication

# ✅ Ollama Cloud Configuration
OLLAMA_API_KEY = os.getenv('OLLAMA_API_KEY')
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'https://ollama.com')

if not OLLAMA_API_KEY:
    print("⚠️  WARNING: OLLAMA_API_KEY not set. Please set it in .env file")

ollama_client = Client(host=OLLAMA_BASE_URL)
print(f"🌐 FastAPI App initialized with Ollama Cloud at {OLLAMA_BASE_URL}")

# FastAPI app
app = FastAPI(
    title="ACE Medical Assistant API",
    description="Agentic Context Engineering (ACE) Medical Intake Assistant",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for active conversations
# In production, use Redis or a database
conversations: Dict[str, Dict[str, Any]] = {}


# ==================== Pydantic Models ====================

class ChatMessage(BaseModel):
    message: str


class ChatResponse(BaseModel):
    conversation_id: str
    assistant_response: str


class ConversationStart(BaseModel):
    model: Optional[str] = 'deepseek-v3.1:671b-cloud'


class ConversationStartResponse(BaseModel):
    conversation_id: str
    playbook_version: str
    playbook_stats: Dict[str, Any]


class AnalysisRequest(BaseModel):
    run_curation: Optional[bool] = True
    run_deduplication: Optional[bool] = True


class AnalysisResponse(BaseModel):
    summary: str
    reflector_analysis: Dict[str, Any]
    curation_results: Optional[Dict[str, Any]] = None
    deduplication_results: Optional[List[Dict[str, Any]]] = None
    playbook_changes: Dict[str, Any]


# ==================== Helper Functions ====================

def save_log(conversation: list, conversation_id: str):
    """Saves the conversation transcript to a timestamped file."""
    if not os.path.exists('logs'):
        os.makedirs('logs')

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/conversation_{conversation_id}_{timestamp}.json"

    try:
        with open(log_filename, 'w') as f:
            json.dump(conversation, f, indent=2)
        print(f"[LOG] Conversation saved to {log_filename}")
        return log_filename
    except IOError as e:
        print(f"[ERROR] Failed to save conversation log: {e}")
        return None


def save_ace_report(report_data: dict, conversation_id: str):
    """Saves a complete ACE analysis report."""
    if not os.path.exists('logs'):
        os.makedirs('logs')

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"logs/ace_report_{conversation_id}_{timestamp}.json"

    try:
        with open(report_filename, 'w') as f:
            json.dump(report_data, f, indent=2)
        print(f"[LOG] ACE report saved to {report_filename}")
        return report_filename
    except IOError as e:
        print(f"[ERROR] Failed to save ACE report: {e}")
        return None


# ==================== API Endpoints ====================

@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "name": "ACE Medical Assistant API",
        "version": "1.0.0",
        "description": "Agentic Context Engineering Medical Intake Assistant",
        "endpoints": {
            "start_conversation": "POST /conversations/start",
            "send_message": "POST /conversations/{conversation_id}/message",
            "end_conversation": "POST /conversations/{conversation_id}/end",
            "analyze": "POST /conversations/{conversation_id}/analyze",
            "get_playbook_stats": "GET /playbook/stats",
            "health": "GET /health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test Ollama connection
        test_response = ollama_client.chat(
            model='gpt-oss:120b-cloud',
            messages=[{'role': 'user', 'content': 'test'}]
        )
        ollama_status = "connected"
    except Exception as e:
        ollama_status = f"error: {str(e)}"

    return {
        "status": "healthy",
        "ollama_status": ollama_status,
        "ollama_base_url": OLLAMA_BASE_URL,
        "api_key_set": bool(OLLAMA_API_KEY),
        "active_conversations": len(conversations),
        "playbook_version": playbook_version
    }


@app.post("/conversations/start", response_model=ConversationStartResponse)
async def start_conversation(request: ConversationStart):
    """
    Start a new conversation session

    Returns a conversation_id to use for subsequent messages
    """
    conversation_id = str(uuid.uuid4())

    # Initialize playbook
    playbook = get_playbook()
    playbook_text = playbook.to_prompt_text(
        sections=['core_rules', 'communication_style', 'questioning_strategy', 'error_prevention'],
        top_k=20
    )

    # Store conversation state
    conversations[conversation_id] = {
        "model": request.model,
        "playbook": playbook,
        "history": [{'role': 'system', 'content': playbook_text}],
        "started_at": datetime.now().isoformat(),
        "playbook_before": playbook.get_statistics()
    }

    stats = playbook.get_statistics()

    return ConversationStartResponse(
        conversation_id=conversation_id,
        playbook_version=playbook_version,
        playbook_stats=stats
    )


@app.post("/conversations/{conversation_id}/message", response_model=ChatResponse)
async def send_message(conversation_id: str, message: ChatMessage):
    """
    Send a message in an existing conversation

    The assistant will respond based on the playbook and conversation history
    """
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")

    conv = conversations[conversation_id]

    # Add user message to history
    conv["history"].append({'role': 'user', 'content': message.message})

    try:
        # ✅ Use Ollama Cloud client
        response = ollama_client.chat(
            model=conv["model"],
            messages=conv["history"]
        )
        assistant_response = response['message']['content']

        # Add assistant response to history
        conv["history"].append({'role': 'assistant', 'content': assistant_response})

        return ChatResponse(
            conversation_id=conversation_id,
            assistant_response=assistant_response
        )

    except Exception as e:
        error_message = f"Error communicating with Ollama Cloud: {e}"
        print(error_message)
        raise HTTPException(
            status_code=503,
            detail="I'm sorry, I'm having trouble connecting to my services right now. Please try again later."
        )


@app.post("/conversations/{conversation_id}/end")
async def end_conversation(conversation_id: str, background_tasks: BackgroundTasks):
    """
    End a conversation and save logs

    This doesn't trigger analysis - use /analyze endpoint for that
    """
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")

    conv = conversations[conversation_id]

    # Save conversation log in background
    background_tasks.add_task(save_log, conv["history"], conversation_id)

    conv["ended_at"] = datetime.now().isoformat()

    return {
        "message": "Conversation ended and saved",
        "conversation_id": conversation_id,
        "total_messages": len([m for m in conv["history"] if m["role"] != "system"]),
        "duration": conv["ended_at"]
    }


@app.post("/conversations/{conversation_id}/analyze", response_model=AnalysisResponse)
async def analyze_conversation(
        conversation_id: str,
        request: AnalysisRequest,
        background_tasks: BackgroundTasks
):
    """
    Analyze a conversation using ACE framework

    Steps:
    1. Generate clinical summary (Summarizer Agent)
    2. Analyze conversation quality (Reflector Agent)
    3. Optionally update playbook (Curator Agent)
    4. Optionally deduplicate bullets (Deduplication)

    Returns full analysis and playbook changes
    """
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")

    conv = conversations[conversation_id]
    playbook = conv["playbook"]
    history = conv["history"]

    try:
        # Phase 1: Summarization
        print(f"[API] Generating summary for conversation {conversation_id}")
        summary = generate_summary(history)

        # Phase 2: Reflection
        print(f"[API] Running reflector analysis for conversation {conversation_id}")
        analysis_result = run_reflector_analysis(history)

        # Phase 3: Curation (optional)
        curation_result = None
        if request.run_curation:
            print(f"[API] Running curator for conversation {conversation_id}")
            curation_result = run_curator(playbook, analysis_result, history)

        # Phase 4: Deduplication (optional)
        dedup_results = None
        if request.run_deduplication:
            print(f"[API] Running deduplication for conversation {conversation_id}")
            dedup_results = semantic_deduplication(playbook)

        # Get playbook changes
        playbook_after = playbook.get_statistics()
        playbook_before = conv["playbook_before"]

        playbook_changes = {
            "before": playbook_before,
            "after": playbook_after,
            "bullets_added": playbook_after['total_bullets'] - playbook_before['total_bullets'],
            "confidence_change": playbook_after['average_confidence'] - playbook_before['average_confidence'],
            "helpful_feedback_added": playbook_after['total_helpful_feedback'] - playbook_before[
                'total_helpful_feedback'],
            "harmful_feedback_added": playbook_after['total_harmful_feedback'] - playbook_before[
                'total_harmful_feedback']
        }

        # Save complete report in background
        report_data = {
            "conversation_id": conversation_id,
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
            "reflector_analysis": analysis_result,
            "curation_results": curation_result,
            "deduplication_results": dedup_results,
            "playbook_changes": playbook_changes
        }
        background_tasks.add_task(save_ace_report, report_data, conversation_id)

        return AnalysisResponse(
            summary=summary,
            reflector_analysis=analysis_result,
            curation_results=curation_result,
            deduplication_results=dedup_results,
            playbook_changes=playbook_changes
        )

    except Exception as e:
        print(f"[ERROR] Analysis failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


@app.get("/playbook/stats")
async def get_playbook_stats():
    """Get current playbook statistics"""
    playbook = get_playbook()
    stats = playbook.get_statistics()

    return {
        "version": playbook_version,
        "statistics": stats,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/playbook/bullets")
async def get_playbook_bullets():
    """Get all playbook bullets with details"""
    playbook = get_playbook()
    bullets = playbook.get_all_bullets()

    return {
        "total_bullets": len(bullets),
        "bullets": [
            {
                "id": b.id,
                "content": b.content,
                "section": b.section,
                "helpful_count": b.helpful_count,
                "harmful_count": b.harmful_count,
                "confidence": b.get_confidence_score(),
                "created_at": b.created_at,
                "last_updated": b.last_updated
            }
            for b in bullets
        ]
    }


@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get conversation details and history"""
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")

    conv = conversations[conversation_id]

    return {
        "conversation_id": conversation_id,
        "model": conv["model"],
        "started_at": conv["started_at"],
        "ended_at": conv.get("ended_at"),
        "message_count": len([m for m in conv["history"] if m["role"] != "system"]),
        "history": conv["history"]
    }


@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation from memory"""
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")

    del conversations[conversation_id]

    return {
        "message": "Conversation deleted",
        "conversation_id": conversation_id
    }


@app.get("/conversations")
async def list_conversations():
    """List all active conversations"""
    return {
        "total_conversations": len(conversations),
        "conversations": [
            {
                "conversation_id": conv_id,
                "started_at": conv["started_at"],
                "ended_at": conv.get("ended_at"),
                "model": conv["model"],
                "message_count": len([m for m in conv["history"] if m["role"] != "system"])
            }
            for conv_id, conv in conversations.items()
        ]
    }


# ==================== Startup/Shutdown Events ====================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    print("=" * 70)
    print("ACE MEDICAL ASSISTANT - FastAPI Server")
    print("=" * 70)
    print(f"Ollama Cloud Base URL: {OLLAMA_BASE_URL}")
    print(f"API Key Set: {bool(OLLAMA_API_KEY)}")
    print(f"Playbook Version: {playbook_version}")

    # Create logs directory
    if not os.path.exists('logs'):
        os.makedirs('logs')

    print("=" * 70)
    print("Server ready! ")
    print("API Docs: http://localhost:5000/docs")
    print("=" * 70)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("\n" + "=" * 70)
    print("Shutting down ACE Medical Assistant API")
    print(f"Active conversations at shutdown: {len(conversations)}")
    print("=" * 70)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=5000,
        reload=True  # Enable auto-reload during development
    )
