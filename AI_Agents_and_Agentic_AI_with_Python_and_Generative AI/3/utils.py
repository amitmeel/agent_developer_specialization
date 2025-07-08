import inspect 
from functools import wraps
from typing import Any, Dict, get_type_hints, get_args, get_origin, List, Union

############ Decorator to convert any function to llm tool schema #############
def llm_tool(_func=None, *, terminal: bool = False):
    """
    Decorator or decorator factory that attaches tool_schema to a function.
    Can be used with or without arguments:
        @llm_tool
        @llm_tool(terminal=True)
    """
    def decorator(func):
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
            },
            "terminal": terminal
        }
    
        # Attach the schema to the wrapper function (this is what gets returned)
        wrapper.tool_schema = tool_schema
        return wrapper
    
    # Handle both @llm_tool and @llm_tool(...) syntax
    if _func is None:
        # Called as @llm_tool(terminal=True)
        return decorator
    else:
        # Called as @llm_tool
        return decorator(_func)


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


# Test the decorator
if __name__ == "__main__":
    @llm_tool
    def search_papers(topic: str, max_results: int = 5) -> dict:
        '''Search for papers on arXiv based on a topic and store their information.
        
        Args:
            topic: The topic to search for
            max_results: Maximum number of results to retrieve
        '''
        return {"results": f"Found papers about {topic}"}
    
    @llm_tool(terminal=True)
    def calculate_sum(a: int, b: int) -> int:
        '''Calculate the sum of two numbers.
        
        Args:
            a: First number
            b: Second number
        '''
        return a + b
    
    # Test that the schema is attached
    print("search_papers has tool_schema:", hasattr(search_papers, 'tool_schema'))
    print("calculate_sum has tool_schema:", hasattr(calculate_sum, 'tool_schema'))
    
    if hasattr(search_papers, 'tool_schema'):
        print("\nSearch papers schema:")
        print(search_papers.tool_schema)
    
    if hasattr(calculate_sum, 'tool_schema'):
        print("\nCalculate sum schema:")
        print(calculate_sum.tool_schema)
    
    # Test get_tools_schema function
    tools = get_tools_schema([search_papers, calculate_sum])
    print(f"\nFound {len(tools)} tools")
    for tool in tools:
        print(f"Tool: {tool['name']}")