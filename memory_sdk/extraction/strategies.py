"""
Default extraction strategies for Memory SDK.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Strategy:
    """Memory extraction strategy definition."""
    name: str
    description: str
    schema: Dict[str, Any]
    prompt_template: Optional[str] = None
    enabled: bool = True


# Default strategies
PROFILE_STRATEGY = Strategy(
    name="profile",
    description="User profile information like name, job, location",
    schema={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "User's name"},
            "job_title": {"type": "string", "description": "User's job title or profession"},
            "company": {"type": "string", "description": "User's company or organization"},
            "location": {"type": "string", "description": "User's location"},
            "expertise": {"type": "array", "items": {"type": "string"}, "description": "User's areas of expertise"},
        },
    },
)

PREFERENCES_STRATEGY = Strategy(
    name="preferences",
    description="User preferences and likes/dislikes",
    schema={
        "type": "object",
        "properties": {
            "communication_style": {"type": "string", "description": "Preferred communication style"},
            "interests": {"type": "array", "items": {"type": "string"}, "description": "User's interests and hobbies"},
            "favorite_foods": {"type": "array", "items": {"type": "string"}, "description": "Favorite foods"},
            "preferred_times": {"type": "string", "description": "Preferred meeting/work times"},
            "dislikes": {"type": "array", "items": {"type": "string"}, "description": "Things user dislikes"},
        },
    },
)

FACTS_STRATEGY = Strategy(
    name="facts",
    description="Important facts about the user",
    schema={
        "type": "object",
        "properties": {
            "facts": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of important facts about the user",
            },
        },
    },
)


DEFAULT_STRATEGIES: Dict[str, Strategy] = {
    "profile": PROFILE_STRATEGY,
    "preferences": PREFERENCES_STRATEGY,
    "facts": FACTS_STRATEGY,
}


def get_strategy(name: str) -> Optional[Strategy]:
    """Get a strategy by name."""
    return DEFAULT_STRATEGIES.get(name)


def register_strategy(strategy: Strategy) -> None:
    """Register a custom strategy."""
    DEFAULT_STRATEGIES[strategy.name] = strategy

