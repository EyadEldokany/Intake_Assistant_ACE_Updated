import json
import os
from datetime import datetime
from typing import List, Dict, Any

__version__ = "1.1"

PLAYBOOK_FILE = "playbook_data.json"

class PlaybookBullet:
    """Represents a single piece of learned knowledge in the playbook."""
    def __init__(self, content: str, section: str, bullet_id: str = None):
        self.id = bullet_id or self._generate_id()
        self.content = content
        self.helpful_count = 0
        self.harmful_count = 0
        self.section = section
        self.created_at = datetime.now().isoformat()
        self.last_updated = self.created_at
    
    def _generate_id(self) -> str:
        """Generate a unique ID for the bullet."""
        return f"bullet_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert bullet to dictionary for JSON storage."""
        return {
            "id": self.id,
            "content": self.content,
            "helpful_count": self.helpful_count,
            "harmful_count": self.harmful_count,
            "section": self.section,
            "created_at": self.created_at,
            "last_updated": self.last_updated
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PlaybookBullet':
        """Create a bullet from dictionary."""
        bullet = cls(data["content"], data["section"], data["id"])
        bullet.helpful_count = data.get("helpful_count", 0)
        bullet.harmful_count = data.get("harmful_count", 0)
        bullet.created_at = data.get("created_at", bullet.created_at)
        bullet.last_updated = data.get("last_updated", bullet.last_updated)
        return bullet
    
    def mark_helpful(self):
        """Increment helpful counter."""
        self.helpful_count += 1
        self.last_updated = datetime.now().isoformat()
    
    def mark_harmful(self):
        """Increment harmful counter."""
        self.harmful_count += 1
        self.last_updated = datetime.now().isoformat()
    
    def get_confidence_score(self) -> float:
        """Calculate confidence score based on feedback."""
        total = self.helpful_count + self.harmful_count
        if total == 0:
            return 0.5  # Neutral for new bullets
        return self.helpful_count / total


class Playbook:
    """Manages the structured playbook with bullets organized by sections."""
    
    def __init__(self):
        self.bullets: List[PlaybookBullet] = []
        self.version = __version__
        self.load_or_initialize()
    
    def load_or_initialize(self):
        """Load existing playbook or create initial one."""
        if os.path.exists(PLAYBOOK_FILE):
            self.load()
        else:
            self._initialize_default_playbook()
            self.save()
    
    def _initialize_default_playbook(self):
        """Create the initial playbook with baseline strategies."""
        initial_bullets = [
            # Core Rules
            PlaybookBullet(
                "NEVER provide a diagnosis, suggest treatments, or offer any form of medical advice. Your role is strictly for information gathering.",
                "core_rules"
            ),
            PlaybookBullet(
                "If the user asks for a diagnosis or advice, gently deflect: 'I am not qualified to provide a diagnosis, but I will make sure to document everything you've told me for the doctor.'",
                "core_rules"
            ),
            
            # Communication Style
            PlaybookBullet(
                "Always speak in a clear, simple, and reassuring tone. Use short sentences.",
                "communication_style"
            ),
            PlaybookBullet(
                "Acknowledge the patient's symptom before asking clarifying questions. Example: 'I'm sorry to hear you have a headache. Could you tell me more about where it hurts?'",
                "communication_style"
            ),
            
            # Questioning Strategy
            PlaybookBullet(
                "Start with a broad and welcoming question. Example: 'I'm here to listen. Could you please tell me what's been bothering you?'",
                "questioning_strategy"
            ),
            PlaybookBullet(
                "Ask about when the symptoms started. Example: 'When did you first start noticing this?'",
                "questioning_strategy"
            ),
            PlaybookBullet(
                "Ask what makes symptoms better or worse. Example: 'Is there anything that seems to make the symptom better or worse?'",
                "questioning_strategy"
            ),
            PlaybookBullet(
                "Use open-ended questions to gather more details. Example: 'Can you describe that feeling in more detail?'",
                "questioning_strategy"
            ),
            
            # Error Prevention
            PlaybookBullet(
                "If the patient describes an emergency symptom (chest pain, difficulty breathing, severe bleeding), immediately advise them to seek emergency care.",
                "error_prevention"
            ),
        ]
        self.bullets = initial_bullets
    
    def load(self):
        """Load playbook from JSON file."""
        try:
            with open(PLAYBOOK_FILE, 'r') as f:
                data = json.load(f)
                self.version = data.get("version", __version__)
                self.bullets = [PlaybookBullet.from_dict(b) for b in data.get("bullets", [])]
        except Exception as e:
            print(f"[ERROR] Failed to load playbook: {e}")
            self._initialize_default_playbook()
    
    def save(self):
        """Save playbook to JSON file."""
        data = {
            "version": self.version,
            "last_updated": datetime.now().isoformat(),
            "bullets": [b.to_dict() for b in self.bullets]
        }
        try:
            with open(PLAYBOOK_FILE, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[ERROR] Failed to save playbook: {e}")
    
    def add_bullet(self, content: str, section: str) -> PlaybookBullet:
        """Add a new bullet to the playbook (delta update)."""
        bullet = PlaybookBullet(content, section)
        self.bullets.append(bullet)
        self.save()
        return bullet
    
    def remove_bullet(self, bullet_id: str) -> bool:
        """Remove a bullet by ID (delta update)."""
        original_length = len(self.bullets)
        self.bullets = [b for b in self.bullets if b.id != bullet_id]
        if len(self.bullets) < original_length:
            self.save()
            return True
        return False
    
    def modify_bullet(self, bullet_id: str, new_content: str = None, 
                     mark_helpful: bool = False, mark_harmful: bool = False) -> bool:
        """Modify a bullet's content or feedback (delta update)."""
        for bullet in self.bullets:
            if bullet.id == bullet_id:
                if new_content:
                    bullet.content = new_content
                    bullet.last_updated = datetime.now().isoformat()
                if mark_helpful:
                    bullet.mark_helpful()
                if mark_harmful:
                    bullet.mark_harmful()
                self.save()
                return True
        return False
    
    def get_bullets_by_section(self, section: str) -> List[PlaybookBullet]:
        """Retrieve bullets from a specific section."""
        return [b for b in self.bullets if b.section == section]
    
    def get_all_bullets(self) -> List[PlaybookBullet]:
        """Retrieve all bullets sorted by confidence score."""
        return sorted(self.bullets, key=lambda b: b.get_confidence_score(), reverse=True)
    
    def to_prompt_text(self, sections: List[str] = None, top_k: int = None) -> str:
        """
        Convert playbook bullets to a formatted prompt text.
        Implements hybrid retrieval by filtering sections and top_k bullets.
        """
        bullets_to_use = self.bullets
        
        # Filter by sections if specified
        if sections:
            bullets_to_use = [b for b in bullets_to_use if b.section in sections]
        
        # Sort by confidence and take top_k if specified
        bullets_to_use = sorted(bullets_to_use, key=lambda b: b.get_confidence_score(), reverse=True)
        if top_k:
            bullets_to_use = bullets_to_use[:top_k]
        
        # Group by section
        sections_dict = {}
        for bullet in bullets_to_use:
            if bullet.section not in sections_dict:
                sections_dict[bullet.section] = []
            sections_dict[bullet.section].append(bullet)
        
        # Format as prompt
        prompt = "You are a friendly and empathetic medical intake assistant named 'HealthBot'.\n\n"
        
        for section, section_bullets in sections_dict.items():
            section_title = section.replace('_', ' ').title()
            prompt += f"**{section_title}:**\n"
            for bullet in section_bullets:
                confidence = bullet.get_confidence_score()
                prompt += f"- {bullet.content} [confidence: {confidence:.2f}]\n"
            prompt += "\n"
        
        return prompt
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get playbook statistics."""
        return {
            "version": self.version,
            "total_bullets": len(self.bullets),
            "sections": list(set(b.section for b in self.bullets)),
            "total_helpful_feedback": sum(b.helpful_count for b in self.bullets),
            "total_harmful_feedback": sum(b.harmful_count for b in self.bullets),
            "average_confidence": sum(b.get_confidence_score() for b in self.bullets) / len(self.bullets) if self.bullets else 0
        }


def get_playbook() -> Playbook:
    """Returns the playbook instance."""
    return Playbook()