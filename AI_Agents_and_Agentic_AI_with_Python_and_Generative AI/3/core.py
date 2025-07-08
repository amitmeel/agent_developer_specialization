# agent loop core file
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, List

@dataclass(frozen=True)
class Goal:
    priority: str
    name: str
    description: str


class Action:
    def __init__(self,
            name: str,
            function: Callable,
            description: str,
            parameters: Dict,
            terminal: bool = False
    ):
        self.name = name
        self.function = function
        self.description = description
        self.terminal = terminal
        self.parameters = parameters

    def execute(self, **args):
        """Execute the action's function"""
        return self.function(**args)
    

class ActionRegistry:
    def __init__(self):
        self.action = {}

    def register(self, action: Action):
        self.action[action.name] = action
    
    def get_action(self, name:str) -> Optional[Action]:
        return self.action.get(name, None)

    def get_actions(self) -> List[Action]:
        """get all registered actions"""
        return list(self.action.values())
    
class Memory:
    def __init__(self):
        self.items = []  # Basic conversation histor

    def add_memory(self, memory: dict):
        """Add memory to working memory"""
        self.items.append(memory)

    def get_memories(self, limit: int = None) -> List[Dict]:
        """Get formatted conversation history for prompt"""
        return self.items[:limit]
    
class Environment:
    def execute_action(self, action: Action, args: dict) -> dict:
        """Execute an action and return the result."""
        try:
            result = action.execute(**args)
            return self.format_result(result)
        except Exception as e:
            return {
                "tool_executed": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }

    def format_result(self, result: Any) -> dict:
        """Format the result with metadata."""
        return {
            "tool_executed": True,
            "result": result,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")
        }
    

    
