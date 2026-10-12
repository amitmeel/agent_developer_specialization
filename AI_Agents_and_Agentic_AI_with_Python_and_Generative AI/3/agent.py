### Hack so that we can run this as standalone script if required#####
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
#####################################################
import os
from typing import List, Dict, Any, Union, Optional

import litellm
from litellm import completion

import tools
from core import ActionRegistry
from register_tools import register_llm_tools
from utils import config

os.environ['GEMINI_API_KEY'] = config.GEMINI_API_KEY
assert litellm.supports_function_calling(model="gemini/gemini-2.5-flash") == True

def generate_response(messages: List[Dict], tools=None) -> str:
    """Call LLM to get response"""
    response = completion(
        model="gemini/gemini-2.5-flash",
        messages=messages,
        tools=tools,
        max_tokens=4096
    )
    return response.choices[0].message

# register all the tools as action
registry = ActionRegistry()
register_llm_tools(tools, registry)
# print([action.name for action in registry.get_actions()])
