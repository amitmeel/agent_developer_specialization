### Hack so that we can run this as standalone script if required#####
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
#####################################################
import inspect
import json
import os

from functools import wraps
from pathlib import Path
from typing import List, Dict, Any, Union, get_type_hints, get_origin, get_args

from utils import config
from litellm import completion

os.environ['GEMINI_API_KEY'] = config.GEMINI_API_KEY

def generate_response(messages: List[Dict]) -> str:
    """Call LLM to get response"""
    response = completion(
        model="gemini/gemini-2.5-flash",
        messages=messages,
        max_tokens=4096
    )
    return response.choices[0].message.content

############ Decrator to convert any function to llm too schema #############

def llm_tool(func):
    """
    Decorator that converts a Python function into an LLM tool schema.
    
    The function should have:
    - Type hints for parameters
    - A docstring with parameter descriptions
    - Optional default values
    
    Example:
        @llm_tool
        def search_papers(topic: str, max_results: int = 5) -> dict:
            '''Search for papers on arXiv based on a topic and store their information.
            
            Args:
                topic: The topic to search for
                max_results: Maximum number of results to retrieve
            '''
            # Function implementation here
            pass
    """
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    
    # Get function signature and type hints
    sig = inspect.signature(func)
    type_hints = get_type_hints(func)
    
    # Parse docstring for descriptions
    doc = inspect.getdoc(func) or ""
    description = ""
    param_descriptions = {}
    
    if doc:
        lines = doc.split('\n')
        # Extract main description (everything before Args:)
        desc_lines = []
        in_args = False
        current_param = None
        
        for line in lines:
            line = line.strip()
            if line.lower().startswith('args:') or line.lower().startswith('parameters:'):
                in_args = True
                continue
            elif line.lower().startswith('returns:') or line.lower().startswith('raises:'):
                break
            elif not in_args:
                if line:
                    desc_lines.append(line)
            else:
                # Parse parameter descriptions
                if ':' in line and not line.startswith(' '):
                    # New parameter
                    param, desc = line.split(':', 1)
                    current_param = param.strip()
                    param_descriptions[current_param] = desc.strip()
                elif current_param and line:
                    # Continue previous parameter description
                    param_descriptions[current_param] += ' ' + line
        
        description = ' '.join(desc_lines)
    
    # Build properties for input schema
    properties = {}
    required = []
    
    for param_name, param in sig.parameters.items():
        if param_name == 'self':  # Skip self parameter for methods
            continue
            
        param_type = type_hints.get(param_name, str)
        param_schema = _get_json_schema_type(param_type)
        
        # Add description if available
        if param_name in param_descriptions:
            param_schema["description"] = param_descriptions[param_name]
        
        # Add default value if present
        if param.default != inspect.Parameter.empty:
            param_schema["default"] = param.default
        else:
            required.append(param_name)
        
        properties[param_name] = param_schema
    
    # Build the complete tool schema
    tool_schema = {
        "name": func.__name__,
        "description": description or f"Execute the {func.__name__} function",
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required
        }
    }
    
    # Attach the schema to the function
    wrapper.tool_schema = tool_schema
    return wrapper


def _get_json_schema_type(python_type) -> Dict[str, Any]:
    """Convert Python type hints to JSON schema types."""
    
    # Handle None type
    if python_type is type(None):
        return {"type": "null"}
    
    # Handle basic types
    if python_type == str:
        return {"type": "string"}
    elif python_type == int:
        return {"type": "integer"}
    elif python_type == float:
        return {"type": "number"}
    elif python_type == bool:
        return {"type": "boolean"}
    elif python_type == list:
        return {"type": "array"}
    elif python_type == dict:
        return {"type": "object"}
    
    # Handle generic types (List, Dict, etc.)
    origin = get_origin(python_type)
    args = get_args(python_type)
    
    if origin is list:
        schema = {"type": "array"}
        if args:
            schema["items"] = _get_json_schema_type(args[0])
        return schema
    elif origin is dict:
        schema = {"type": "object"}
        if len(args) >= 2:
            schema["additionalProperties"] = _get_json_schema_type(args[1])
        return schema
    elif origin is tuple:
        return {"type": "array"}
    
    # Handle Union types (Optional)
    if hasattr(python_type, '__origin__') and python_type.__origin__ is Union:
        union_args = python_type.__args__
        if len(union_args) == 2 and type(None) in union_args:
            # This is Optional[T]
            non_none_type = next(arg for arg in union_args if arg is not type(None))
            return _get_json_schema_type(non_none_type)
    
    # Default to string for unknown types
    return {"type": "string"}


def get_tools_schema(functions: List) -> List[Dict]:
    """
    Extract tool schemas from a list of decorated functions.
    
    Args:
        functions: List of functions decorated with @llm_tool
        
    Returns:
        List of tool schema dictionaries
    """
    tools = []
    for func in functions:
        if hasattr(func, 'tool_schema'):
            tools.append(func.tool_schema)
    return tools

#################function which will work as tools for llm##############

@llm_tool
def list_files(directory='.'):
    """
    Lists all files in the specified directory.
    
    Args:
        directory: Path to the directory
    
    Returns
        List[str]: List of file names (not including directories)
    """
    try:
        return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    except FileNotFoundError:
        print(f"Directory not found: {directory}")
        return []
    except Exception as e:
        print(f"Error reading directory: {e}")
        return []

@llm_tool
def read_file(filepath):
    """
    Reads and returns the content of a file.
    
    Args:
        filepath: Path to the file
    
    Returns:
        str: Content of the file as a string
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return ""
    except Exception as e:
        print(f"Error reading file: {e}")
        return ""

def parse_action(response):
    parsed_response = response.split('```')[1]
    if "action" in parsed_response:
        parsed_response = parsed_response[6:]
    return parsed_response

# Create your tool registry
# This maps the string name of the tool to its actual function object
available_tools = {
    "list_files": list_files,
    "read_file": read_file,
    # Add any other tools you create here
}
# Generate tool schemas
functions = [read_file, list_files]
tools = get_tools_schema(functions)

tools = json.dumps(tools, indent=2)


agent_system_message = f"""
You are an AI agent that performs tasks using available tools:

{tools}

Instructions:
- Always show the available tools when the user asks about capabilities.
- If the user asks about files, list them **before** reading any file.
- Every response MUST include an action.

Respond in the following format only if an action is needed else ask user what action he wants, if he want to end / terminate chat, respond with terminate action.:

```action
{{
    "tool_name": "name_of_tool",
    "args": {{
        "arg1": "value1",
        "arg2": "value2"
    }}
}}
"""
system_message = [{'role': 'system', 'content': agent_system_message}]
max_iteration=8
iteration=0


memory = [{"role": "user", "content": "What files are in this directory?"}]
while iteration < max_iteration:
    prompt =  system_message + memory
    response = generate_response(prompt)
    if "```action" in response:
        action = parse_action(response)
        action = json.loads(action)
        
        if action["tool_name"]=="terminate":
            break
        else:
            tool_name_from_llm = action["tool_name"]

            # Look up the actual function from the registry
            if tool_name_from_llm in available_tools:
                tool_function = available_tools[tool_name_from_llm]
                try:
                    # Now call the actual function object
                    result = tool_function(**action["args"])
                except TypeError as e:
                    # Handle cases where the LLM might generate incorrect args
                    result = {"error": f"Tool '{tool_name_from_llm}' called with incorrect arguments: {e}"}
                except Exception as e:
                    # Catch any other runtime errors during tool execution
                    result = {"error": f"Error executing tool '{tool_name_from_llm}': {e}"}
            else:
                # If the LLM hallucinates a tool name that doesn't exist
                result = {"error": "Unknown tool: " + tool_name_from_llm}
        
        memory.extend([
            {"role": "assistant", "content": response},
            {"role": "user", "content": json.dumps(result)}
        ])
    else:
        memory.extend([
            {"role": "assistant", "content": response},
            {"role": "user", "content": input("response to assistance messgage")}
        ])

    iteration += 1
    
