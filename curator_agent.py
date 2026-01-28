import ollama
from ollama import Client 
import json
from typing import Dict, List, Any
from playbook import Playbook
import os

OLLAMA_API_KEY = os.getenv('OLLAMA_API_KEY')
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'https://api.ollama.cloud')

ollama_client = Client(host=OLLAMA_BASE_URL)

def get_curator_playbook():
    """
    Defines the system prompt for the Curator Agent.
    Its job is to autonomously update the playbook based on reflector analysis.
    """
    return """You are an expert Knowledge Curator AI. Your task is to update a medical intake assistant's playbook based on performance analysis from a Reflector Agent.

You will receive:
1. The current playbook bullets (structured knowledge)
2. A reflector's analysis of a conversation
3. Suggested improvements

Your job is to generate **delta updates** to the playbook. You must output a JSON object with one or more of these operations:

**Operation Types:**
1. "add" - Add a new bullet
2. "remove" - Remove an existing bullet by ID
3. "modify" - Update an existing bullet's content
4. "mark_helpful" - Increment helpful counter for a bullet
5. "mark_harmful" - Increment harmful counter for a bullet

**Output Format:**
{
    "operations": [
        {
            "type": "add",
            "content": "The new strategy text",
            "section": "questioning_strategy",
            "reasoning": "Why this addition improves the playbook"
        },
        {
            "type": "modify",
            "bullet_id": "bullet_20250101_120000_123456",
            "new_content": "Updated strategy text",
            "reasoning": "Why this modification is needed"
        },
        {
            "type": "remove",
            "bullet_id": "bullet_20250101_120000_789012",
            "reasoning": "Why this bullet should be removed"
        },
        {
            "type": "mark_helpful",
            "bullet_id": "bullet_20250101_120000_345678",
            "reasoning": "This strategy worked well in the conversation"
        }
    ]
}

**Guidelines:**
- Only suggest operations that are clearly justified by the reflector's analysis
- Prefer modifying existing bullets over adding new ones when possible
- Remove bullets that consistently lead to harmful outcomes (harmful_count > helpful_count * 2)
- Mark bullets as helpful when they contributed to good outcomes
- Keep new bullet content concise and actionable (1-2 sentences)
- Use appropriate sections: "core_rules", "communication_style", "questioning_strategy", "error_prevention", "task_guidance"
- If no changes are needed, return an empty operations list: {"operations": []}

**Important:** Base your decisions solely on the evidence provided in the reflector's analysis. Do not make assumptions."""

def run_curator(playbook: Playbook, reflector_analysis: Dict[str, Any], 
                conversation_history: List[Dict], model='gpt-oss:120b-cloud') -> Dict[str, Any]:
    """
    Runs the curator agent to autonomously update the playbook.

    Args:
        playbook: The current Playbook instance
        reflector_analysis: The analysis from the reflector agent
        conversation_history: The conversation that was analyzed
        model: The Ollama model to use for curation

    Returns:
        A dictionary containing the curation operations and results
    """
    curator_playbook_text = get_curator_playbook()
    
    # Prepare playbook context
    current_bullets = [
        {
            "id": b.id,
            "content": b.content,
            "section": b.section,
            "helpful_count": b.helpful_count,
            "harmful_count": b.harmful_count,
            "confidence": b.get_confidence_score()
        }
        for b in playbook.get_all_bullets()
    ]
    
    # Create the curator prompt
    curator_input = f"""**Current Playbook Bullets:**
{json.dumps(current_bullets, indent=2)}

**Reflector Analysis:**
{json.dumps(reflector_analysis, indent=2)}

**Task:**
Based on the reflector's analysis, what delta updates should be made to the playbook? Generate operations that will improve the assistant's performance."""

    messages = [
        {'role': 'system', 'content': curator_playbook_text},
        {'role': 'user', 'content': curator_input}
    ]

    try:
        print("\n[CURATOR] Analyzing playbook updates...")
        response = ollama.chat(model=model, messages=messages, format='json')
        operations_data = json.loads(response['message']['content'])
        
        # Execute the operations
        execution_results = execute_operations(playbook, operations_data.get("operations", []))
        
        return {
            "operations_proposed": operations_data.get("operations", []),
            "execution_results": execution_results,
            "total_operations": len(operations_data.get("operations", [])),
            "successful_operations": sum(1 for r in execution_results if r.get("success"))
        }
        
    except Exception as e:
        return {
            "error": f"Curator failed: {e}",
            "operations_proposed": [],
            "execution_results": []
        }


def execute_operations(playbook: Playbook, operations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Execute the delta operations on the playbook.
    
    Args:
        playbook: The Playbook instance to update
        operations: List of operations to execute
    
    Returns:
        List of execution results
    """
    results = []
    
    for op in operations:
        op_type = op.get("type")
        reasoning = op.get("reasoning", "No reasoning provided")
        
        try:
            if op_type == "add":
                bullet = playbook.add_bullet(op["content"], op["section"])
                results.append({
                    "operation": "add",
                    "success": True,
                    "bullet_id": bullet.id,
                    "reasoning": reasoning,
                    "message": f"Added new bullet to section '{op['section']}'"
                })
                print(f"  ✓ Added: {op['content'][:60]}...")
                
            elif op_type == "remove":
                success = playbook.remove_bullet(op["bullet_id"])
                results.append({
                    "operation": "remove",
                    "success": success,
                    "bullet_id": op["bullet_id"],
                    "reasoning": reasoning,
                    "message": "Removed bullet" if success else "Bullet not found"
                })
                if success:
                    print(f"  ✓ Removed: {op['bullet_id']}")
                
            elif op_type == "modify":
                success = playbook.modify_bullet(op["bullet_id"], new_content=op.get("new_content"))
                results.append({
                    "operation": "modify",
                    "success": success,
                    "bullet_id": op["bullet_id"],
                    "reasoning": reasoning,
                    "message": "Modified bullet" if success else "Bullet not found"
                })
                if success:
                    print(f"  ✓ Modified: {op['bullet_id']}")
                
            elif op_type == "mark_helpful":
                success = playbook.modify_bullet(op["bullet_id"], mark_helpful=True)
                results.append({
                    "operation": "mark_helpful",
                    "success": success,
                    "bullet_id": op["bullet_id"],
                    "reasoning": reasoning,
                    "message": "Marked as helpful" if success else "Bullet not found"
                })
                if success:
                    print(f"  ✓ Marked helpful: {op['bullet_id']}")
                
            elif op_type == "mark_harmful":
                success = playbook.modify_bullet(op["bullet_id"], mark_harmful=True)
                results.append({
                    "operation": "mark_harmful",
                    "success": success,
                    "bullet_id": op["bullet_id"],
                    "reasoning": reasoning,
                    "message": "Marked as harmful" if success else "Bullet not found"
                })
                if success:
                    print(f"  ✓ Marked harmful: {op['bullet_id']}")
                    
            else:
                results.append({
                    "operation": op_type,
                    "success": False,
                    "reasoning": reasoning,
                    "message": f"Unknown operation type: {op_type}"
                })
                
        except Exception as e:
            results.append({
                "operation": op_type,
                "success": False,
                "reasoning": reasoning,
                "message": f"Error executing operation: {e}"
            })
    
    return results


def semantic_deduplication(playbook: Playbook, model='deepseek-v3.1:671b-cloud', threshold=0.85):
    """
    Identifies and merges semantically similar bullets to prevent playbook bloat.
    
    Args:
        playbook: The Playbook instance
        model: Model to use for semantic comparison
        threshold: Similarity threshold (0-1) for considering bullets duplicates
    
    Returns:
        List of merge operations performed
    """
    bullets = playbook.get_all_bullets()
    if len(bullets) < 2:
        return []
    
    merge_operations = []
    
    # Simple implementation: compare bullets pairwise
    # In production, this would use embeddings (e.g., sentence-transformers)
    dedup_prompt = """You are a semantic deduplication expert. Compare these two playbook bullets and determine if they convey essentially the same strategy or knowledge.

Output JSON:
{
    "are_duplicates": true/false,
    "similarity_score": 0.0-1.0,
    "reasoning": "Brief explanation"
}"""

    print("\n[DEDUPLICATION] Checking for semantic duplicates...")
    checked_pairs = set()
    
    for i, bullet1 in enumerate(bullets):
        for j, bullet2 in enumerate(bullets[i+1:], start=i+1):
            pair_key = f"{bullet1.id}_{bullet2.id}"
            if pair_key in checked_pairs:
                continue
            checked_pairs.add(pair_key)
            
            # Only check bullets in the same section
            if bullet1.section != bullet2.section:
                continue
            
            messages = [
                {'role': 'system', 'content': dedup_prompt},
                {'role': 'user', 'content': f"Bullet 1: {bullet1.content}\n\nBullet 2: {bullet2.content}"}
            ]
            
            try:
                response = ollama.chat(model=model, messages=messages, format='json')
                result = json.loads(response['message']['content'])
                
                if result.get("are_duplicates") and result.get("similarity_score", 0) >= threshold:
                    # Keep the bullet with higher confidence
                    if bullet1.get_confidence_score() >= bullet2.get_confidence_score():
                        # Merge bullet2 into bullet1
                        playbook.modify_bullet(
                            bullet1.id,
                            mark_helpful=bullet2.helpful_count > 0
                        )
                        playbook.remove_bullet(bullet2.id)
                        merge_operations.append({
                            "kept": bullet1.id,
                            "removed": bullet2.id,
                            "reasoning": result.get("reasoning")
                        })
                        print(f"  ✓ Merged duplicate: {bullet2.id} into {bullet1.id}")
                    else:
                        # Merge bullet1 into bullet2
                        playbook.modify_bullet(
                            bullet2.id,
                            mark_helpful=bullet1.helpful_count > 0
                        )
                        playbook.remove_bullet(bullet1.id)
                        merge_operations.append({
                            "kept": bullet2.id,
                            "removed": bullet1.id,
                            "reasoning": result.get("reasoning")
                        })
                        print(f"  ✓ Merged duplicate: {bullet1.id} into {bullet2.id}")
                    break  # Move to next bullet1
                    
            except Exception as e:
                print(f"  ! Error checking pair: {e}")
                continue
    
    if not merge_operations:
        print("  No duplicates found")
    
    return merge_operations