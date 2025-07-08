import inspect
from types import ModuleType
from core import Action, ActionRegistry

def register_llm_tools(module: ModuleType, registry: ActionRegistry):
    """
    Scan a module for @llm_tool decorated functions and register them as Actions.
    """
    for name, obj in inspect.getmembers(module, inspect.isfunction):
        if hasattr(obj, "tool_schema"):
            schema = obj.tool_schema
            action = Action(
                name=schema["name"],
                function=obj,
                description=schema["description"],
                parameters=schema["input_schema"],
                terminal=False,  # You can later support this via decorator too
            )
            registry.register(action)
