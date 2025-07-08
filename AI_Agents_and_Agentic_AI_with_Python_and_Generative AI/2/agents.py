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
from typing import List, Dict, Any, Union, Optional, get_type_hints, get_origin, get_args

import litellm
from utils import config
from litellm import completion

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

########################### tool functions ################################
@llm_tool
def list_python_files(directory: Optional[str] = None) -> List[str]:
    """
    Returns a list of all Python files in the specified directory.

    Args:
        directory (str, optional): The path to the directory to search. 
            Defaults to the current working directory if not provided.

    Returns:
        List[str]: A list of file paths to Python (.py) files in the directory.

    Raises:
        FileNotFoundError: If the specified directory does not exist.
        NotADirectoryError: If the provided path is not a directory.
    """
    dir_to_search = directory or os.getcwd()

    if not os.path.exists(dir_to_search):
        raise FileNotFoundError(f"The directory '{dir_to_search}' does not exist.")
    if not os.path.isdir(dir_to_search):
        raise NotADirectoryError(f"'{dir_to_search}' is not a directory.")

    return [
        os.path.join(dir_to_search, f)
        for f in os.listdir(dir_to_search)
        if f.endswith(".py") and os.path.isfile(os.path.join(dir_to_search, f))
    ]

@llm_tool
def read_file(file_path: str) -> str:
    """
    Reads the content of a specified file.

    Args:
        file_path (str): The path to the file to be read.

    Returns:
        str: The content of the file.

    Raises:
        FileNotFoundError: If the file does not exist.
        OSError: If there is an issue reading the file.
    """
    if not file_path:
        raise ValueError("The 'file_path' must be provided.")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")
    except OSError as e:
        raise OSError(f"Failed to read file '{file_path}': {e}")

@llm_tool
def write_doc_file(file_name: str, content: str) -> None:
    """
    Writes a documentation file to the docs/ directory.

    Args:
        file_name (str): The name of the file to write (e.g., 'intro.md').
        content (str): The content to write into the file.

    Raises:
        ValueError: If file_name or content is empty.
        OSError: If there is an error creating the directory or writing the file.
    """
    if not file_name or not content:
        raise ValueError("Both 'file_name' and 'content' must be provided.")

    docs_dir = "docs"
    os.makedirs(docs_dir, exist_ok=True)

    file_path = os.path.join(docs_dir, file_name)

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Documentation written to {file_path}")
    except OSError as e:
        raise OSError(f"Failed to write documentation file: {e}")

def terminate(message: str) -> None:
    """Terminate the agent loop and provide a summary message."""
    print(f"Termination message: {message}")

tool_functions = {
    "list_python_files": list_python_files,
    "read_file": read_file,
    "write_doc_file": write_doc_file,
    "terminate": terminate
}

# Generate tool schemas
functions = [list_python_files, read_file, write_doc_file, terminate]
tools = get_tools_schema(functions)
# print(tools)

# Our rules are simplified since we don't have to worry about getting a specific output format
agent_rules = [{
    "role": "system",
    "content": """
You are an AI agent that can perform tasks by using available tools. 

If a user asks about files, documents, or content, first list the files before reading them.

When you are done, terminate the conversation by using the "terminate" tool and I will provide the results to the user.
"""
}]

# Initialize agent parameters
iterations = 0
max_iterations = 10

user_task = input("What would you like me to do? ")

memory = [{"role": "user", "content": user_task}]

# The Agent Loop
while iterations < max_iterations:

    messages = agent_rules + memory

    response = completion(
        model="openai/gpt-4o",
        messages=messages,
        tools=tools,
        max_tokens=1024
    )

    # No More Custom Parsing Logic
    # Dynamic Execution
    # Automated Function Execution
    if response.choices[0].message.tool_calls:
        tool = response.choices[0].message.tool_calls[0]
        tool_name = tool.function.name
        tool_args = json.loads(tool.function.arguments)

        action = {
            "tool_name": tool_name,
            "args": tool_args
        }

        if tool_name == "terminate":
            print(f"Termination message: {tool_args['message']}")
            break
        elif tool_name in tool_functions:
            try:
                result = {"result": tool_functions[tool_name](**tool_args)}
            except Exception as e:
                result = {"error":f"Error executing {tool_name}: {str(e)}"}
        else:
            result = {"error": f"Unknown tool: {tool_name}"}

        print(f"Executing: {tool_name} with args {tool_args}")
        print(f"Result: {result}")
        memory.extend([
            {"role": "assistant", "content": json.dumps(action)},
            {"role": "user", "content": json.dumps(result)}
        ])
    else:
        result = response.choices[0].message.content
        print(f"Response: {result}")
        break