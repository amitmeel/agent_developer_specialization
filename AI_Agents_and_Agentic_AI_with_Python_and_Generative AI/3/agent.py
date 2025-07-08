import tools

from core import ActionRegistry
from register_tools import register_llm_tools


# register all the tools as action
registry = ActionRegistry()
register_llm_tools(tools, registry)
# print([action.name for action in registry.get_actions()])
