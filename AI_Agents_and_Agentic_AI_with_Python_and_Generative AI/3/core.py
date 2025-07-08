from dataclasses import dataclass
from typing import Callable, Dict, Optional, List

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
    

    
