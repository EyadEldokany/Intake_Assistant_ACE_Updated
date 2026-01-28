import ollama
from ollama import Client 
import json
import os
from datetime import datetime
from playbook import get_playbook, __version__ as playbook_version
from reflector_agent import run_reflector_analysis
from summarizer_agent import generate_summary
from curator_agent import run_curator, semantic_deduplication


OLLAMA_API_KEY = os.getenv('OLLAMA_API_KEY')
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'https://api.ollama.cloud')

if not OLLAMA_API_KEY:
    print("⚠️  WARNING: OLLAMA_API_KEY not set. Please set it in .env file")

# ✅ Create Ollama Cloud client
ollama_client = Client(host=OLLAMA_BASE_URL)
print(f"🌐 App initialized with Ollama Cloud at {OLLAMA_BASE_URL}")
class IntakeAssistant:
    """
    Manages the patient intake conversation with ACE-based learning.
    """
    def __init__(self, model='deepseek-v3.1:671b-cloud'):
        self.model = model
        self.playbook = get_playbook()
        
        # Use hybrid retrieval: get most relevant bullets for this task
        playbook_text = self.playbook.to_prompt_text(
            sections=['core_rules', 'communication_style', 'questioning_strategy', 'error_prevention'],
            top_k=20  # Limit context to top 20 most confident bullets
        )
        
        self.conversation_history = [{'role': 'system', 'content': playbook_text}]
        
        stats = self.playbook.get_statistics()
        print(f"Assistant initialized with Playbook v{playbook_version}")
        print(f"  • {stats['total_bullets']} bullets loaded")
        print(f"  • Average confidence: {stats['average_confidence']:.2f}")
        print(f"  • Sections: {', '.join(stats['sections'])}")

    def generator(self, user_input: str) -> str:
        """
        Generator Agent: Generates the next response in the conversation.
        Uses the curated playbook as context.
        """
        self.conversation_history.append({'role': 'user', 'content': user_input})
        try:
            response = ollama.chat(model=self.model, messages=self.conversation_history)
            assistant_response = response['message']['content']
            self.conversation_history.append({'role': 'assistant', 'content': assistant_response})
            return assistant_response
        except Exception as e:
            error_message = f"Error communicating with Ollama: {e}"
            print(error_message)
            return "I'm sorry, I'm having trouble connecting to my services right now. Please try again later."

def save_log(conversation: list, cycle_number: int = None):
    """Saves the conversation transcript to a timestamped file in the logs/ directory."""
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cycle_prefix = f"cycle{cycle_number}_" if cycle_number else ""
    log_filename = f"logs/{cycle_prefix}conversation_{timestamp}.json"
    
    try:
        with open(log_filename, 'w') as f:
            json.dump(conversation, f, indent=2)
        print(f"\n[LOG] Conversation saved to {log_filename}")
        return log_filename
    except IOError as e:
        print(f"\n[ERROR] Failed to save conversation log: {e}")
        return None

def save_ace_cycle_report(cycle_data: dict, cycle_number: int):
    """Saves a complete ACE cycle report."""
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"logs/cycle{cycle_number}_ace_report_{timestamp}.json"
    
    try:
        with open(report_filename, 'w') as f:
            json.dump(cycle_data, f, indent=2)
        print(f"[LOG] ACE cycle report saved to {report_filename}")
    except IOError as e:
        print(f"[ERROR] Failed to save ACE report: {e}")

def display_playbook_diff(before_stats: dict, after_stats: dict):
    """Display the changes to the playbook."""
    print("\n" + "="*50)
    print("Playbook Changes")
    print("="*50)
    print(f"Bullets: {before_stats['total_bullets']} → {after_stats['total_bullets']} "
          f"({after_stats['total_bullets'] - before_stats['total_bullets']:+d})")
    print(f"Avg Confidence: {before_stats['average_confidence']:.3f} → {after_stats['average_confidence']:.3f} "
          f"({after_stats['average_confidence'] - before_stats['average_confidence']:+.3f})")
    print(f"Total Helpful Feedback: {before_stats['total_helpful_feedback']} → {after_stats['total_helpful_feedback']} "
          f"({after_stats['total_helpful_feedback'] - before_stats['total_helpful_feedback']:+d})")
    print(f"Total Harmful Feedback: {before_stats['total_harmful_feedback']} → {after_stats['total_harmful_feedback']} "
          f"({after_stats['total_harmful_feedback'] - before_stats['total_harmful_feedback']:+d})")

def run_ace_cycle(cycle_number: int = 1):
    """
    Runs one complete ACE learning cycle.
    """
    print("\n" + "="*70)
    print(f"ACE LEARNING CYCLE #{cycle_number}")
    print("="*70)
    
    # --- PHASE 1: EXECUTION (Generator) ---
    print("\n" + "="*50)
    print("Phase 1: Conversation (Generator Agent)")
    print("="*50)
    print("Type 'exit' to end the conversation and begin analysis.")
    print("-"*50)
    
    assistant = IntakeAssistant()
    playbook_before = assistant.playbook.get_statistics()

    while True:
        user_message = input("You: ")
        if user_message.lower() == 'exit':
            print("\nAssistant: Thank you. Ending the conversation and starting analysis...")
            break
        assistant_response = assistant.generator(user_message)
        print(f"Assistant: {assistant_response}")

    conversation = assistant.conversation_history

    # --- PHASE 2: LOGGING ---
    log_file = save_log(conversation, cycle_number)

    # --- PHASE 3: SUMMARIZATION ---
    print("\n" + "="*50)
    print("Phase 2: Clinical Summary (Summarizer Agent)")
    print("="*50)
    summary = generate_summary(conversation)
    print(summary)

    # --- PHASE 4: REFLECTION ---
    print("\n" + "="*50)
    print("Phase 3: Performance Analysis (Reflector Agent)")
    print("="*50)
    analysis_result = run_reflector_analysis(conversation)
    print(json.dumps(analysis_result, indent=2))
    
    # --- PHASE 5: CURATION (Autonomous) ---
    print("\n" + "="*50)
    print("Phase 4: Autonomous Playbook Curation (Curator Agent)")
    print("="*50)
    
    curation_result = run_curator(assistant.playbook, analysis_result, conversation)
    
    print(f"\n[CURATOR RESULTS]")
    print(f"  • Operations proposed: {curation_result['total_operations']}")
    print(f"  • Operations successful: {curation_result['successful_operations']}")
    
    if curation_result['total_operations'] > 0:
        print("\n[OPERATIONS EXECUTED]")
        for result in curation_result['execution_results']:
            status = "✓" if result['success'] else "✗"
            print(f"  {status} {result['operation']}: {result['message']}")
            print(f"     Reasoning: {result['reasoning']}")
    
    # --- PHASE 6: SEMANTIC DEDUPLICATION ---
    print("\n" + "="*50)
    print("Phase 5: Semantic Deduplication")
    print("="*50)
    
    dedup_results = semantic_deduplication(assistant.playbook)
    if dedup_results:
        print(f"\n[DEDUPLICATION] Merged {len(dedup_results)} duplicate bullets")
    
    # --- PHASE 7: PLAYBOOK STATISTICS ---
    playbook_after = assistant.playbook.get_statistics()
    display_playbook_diff(playbook_before, playbook_after)
    
    # --- SAVE COMPLETE ACE CYCLE REPORT ---
    cycle_report = {
        "cycle_number": cycle_number,
        "timestamp": datetime.now().isoformat(),
        "conversation_log": log_file,
        "summary": summary,
        "reflector_analysis": analysis_result,
        "curation_results": curation_result,
        "deduplication_results": dedup_results,
        "playbook_before": playbook_before,
        "playbook_after": playbook_after
    }
    save_ace_cycle_report(cycle_report, cycle_number)
    
    print("\n" + "="*70)
    print(f"ACE CYCLE #{cycle_number} COMPLETE")
    print("="*70)
    print("\nThe playbook has been automatically updated based on this conversation.")
    print("Run another cycle to continue learning, or review the logs/ directory for details.")
    
    return cycle_report

def main():
    """
    Main function implementing the full ACE workflow with continuous learning.
    """
    print("="*70)
    print("AGENTIC CONTEXT ENGINEERING (ACE) - MEDICAL INTAKE ASSISTANT")
    print("="*70)
    print("\nThis system implements Stanford's ACE framework:")
    print("  • Generator Agent: Conducts patient intake conversations")
    print("  • Reflector Agent: Analyzes conversation quality")
    print("  • Curator Agent: Autonomously updates the playbook")
    print("  • Continuous Learning: Each conversation improves future performance")
    print("\n" + "="*70)
    
    cycle = 1
    
    while True:
        run_ace_cycle(cycle)
        
        print("\n" + "-"*70)
        choice = input("\nRun another learning cycle? (yes/no): ").strip().lower()
        if choice not in ['yes', 'y']:
            break
        cycle += 1
    
    print("\n" + "="*70)
    print("ACE SESSION COMPLETE")
    print("="*70)
    print("\nPlaybook has been updated through autonomous learning.")
    print("Check the logs/ directory for detailed reports of each cycle.")
    print("\nKey files:")
    print("  • playbook_data.json - Current playbook state")
    print("  • logs/ - All conversation logs and ACE cycle reports")

if __name__ == "__main__":
    main()