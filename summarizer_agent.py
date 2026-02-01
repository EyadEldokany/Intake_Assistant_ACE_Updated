import ollama
import json
from ollama import Client
import os
OLLAMA_API_KEY = os.getenv('OLLAMA_API_KEY')
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'https://ollama.com')

ollama_client = Client(host=OLLAMA_BASE_URL)
def get_summarizer_playbook():
    """
    Defines the system prompt for the Summarizer Agent.
    Its job is to convert a raw conversation transcript into a structured clinical note.
    """
    return """You are a highly efficient medical scribe AI. Your task is to analyze a conversation transcript between a patient and an intake assistant and convert it into a structured, concise clinical summary.

The summary **MUST** be in Markdown format and follow the structure below. If a section has no relevant information, state "Not mentioned."

**Output Format:**

# Patient Intake Summary

## Chief Complaint
The primary reason the patient is seeking care, in their own words.

## History of Present Illness (HPI)

### Onset
When did the symptoms begin?

### Duration
How long do the symptoms last?

### Location
Where are the symptoms located?

### Character
How does the patient describe the symptom (e.g., sharp, dull, aching, burning)?

### Severity
Rate on a scale of 1-10 if mentioned, or describe qualitatively.

### Aggravating Factors
What makes the symptoms worse?

### Alleviating Factors
What makes the symptoms better?

### Associated Symptoms
Are there any other symptoms occurring at the same time?

### Timing/Pattern
Does the symptom come and go? Is it constant? Any pattern?

## Review of Systems
Any additional symptoms mentioned that weren't part of the chief complaint.

## Patient's Stated Concerns
Any specific worries, fears, or questions the patient expressed.

## Quality Assessment
Brief note on completeness of the intake (did the assistant gather sufficient information?).

---

**Instructions:**
- Extract information ONLY from what was actually discussed
- Use the patient's own words when possible (with quotes)
- Be concise but comprehensive
- If critical information is missing, note it in Quality Assessment
- Maintain clinical objectivity"""

def generate_summary(conversation_history: list, model='deepseek-v3.1:671b-cloud') -> str:
    """
    Runs the summarizer agent on a conversation.

    Args:
        conversation_history: The list of messages from the conversation.
        model: The Ollama model to use for summarization.

    Returns:
        A string containing the structured clinical summary.
    """
    summarizer_playbook = get_summarizer_playbook()
    
    # Format the conversation history as a single string for the summarizer
    conversation_text = "\n".join([
        f"{msg['role']}: {msg['content']}" 
        for msg in conversation_history 
        if msg['role'] != 'system'
    ])

    messages = [
        {'role': 'system', 'content': summarizer_playbook},
        {'role': 'user', 'content': f"Please generate a clinical summary from the following transcript:\n\n{conversation_text}"}
    ]

    try:
        print("[SUMMARIZER] Generating clinical summary...")
        response = ollama_client.chat(model=model, messages=messages)
        summary = response['message']['content']
        print("[SUMMARIZER] Summary generated successfully")
        return summary
    except Exception as e:
        error_msg = f"Error generating summary: {e}"
        print(f"[SUMMARIZER ERROR] {error_msg}")
        return f"""# Patient Intake Summary

**ERROR:** {error_msg}

The summary could not be generated automatically. Please review the conversation log manually."""


def generate_summary_json(conversation_history: list, model='deepseek-v3.1:671b-cloud') -> dict:
    """
    Alternative: Generate summary as structured JSON for programmatic use.
    
    Args:
        conversation_history: The list of messages from the conversation.
        model: The Ollama model to use for summarization.
    
    Returns:
        Dictionary containing structured clinical data
    """
    json_prompt = """You are a medical scribe AI. Extract clinical information from this conversation and output ONLY a JSON object:

{
    "chief_complaint": "string",
    "onset": "string",
    "duration": "string",
    "location": "string",
    "character": "string",
    "severity": "string",
    "aggravating_factors": ["string"],
    "alleviating_factors": ["string"],
    "associated_symptoms": ["string"],
    "timing_pattern": "string",
    "patient_concerns": ["string"],
    "completeness_score": "COMPLETE/PARTIAL/INCOMPLETE",
    "missing_information": ["string"]
}

If any field was not discussed, use null or an empty array."""

    conversation_text = "\n".join([
        f"{msg['role']}: {msg['content']}" 
        for msg in conversation_history 
        if msg['role'] != 'system'
    ])

    messages = [
        {'role': 'system', 'content': json_prompt},
        {'role': 'user', 'content': conversation_text}
    ]

    try:
        response = ollama_client.chat(model=model, messages=messages, format='json')
        return json.loads(response['message']['content'])
    except Exception as e:
        return {
            "error": f"Failed to generate JSON summary: {e}",
            "chief_complaint": None,
            "completeness_score": "ERROR"
        }
