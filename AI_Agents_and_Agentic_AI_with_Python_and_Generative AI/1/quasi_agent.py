### Hack so that we can run this as standalone script if required#####
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
#####################################################
"""
or this exercise, you should write a program that uses sequential prompts to generate any Python
function based on user input. The program should:

First Prompt:

- Ask the user what function they want to create
- Ask the LLM to write a basic Python function based on the user’s description
- Store the response for use in subsequent prompts
- Parse the response to separate the code from the commentary by the LLM

Second Prompt:

- Pass the code generated from the first prompt
- Ask the LLM to add comprehensive documentation including:
- Function description
- Parameter descriptions
- Return value description
- Example usage
- Edge cases

Third Prompt:

- Pass the documented code generated from the second prompt
- Ask the LLM to add test cases using Python’s unittest framework
- Tests should cover:
- Basic functionality
- Edge cases
- Error cases
- Various input scenarios

Requirements:

- Use the LiteLLM library
- Maintain conversation context between prompts
- Print each step of the development process
- Save the final version to a Python file

If you want to practice further, try using the system message to force the LLM to always output 
code that has a specific style or uses particular libraries.
"""



import os
import re

from pathlib import Path
from typing import List, Dict

from litellm import completion

from utils import config

os.environ['GEMINI_API_KEY'] = config.GEMINI_API_KEY

def generate_response(messages: List[Dict]) -> str:
    """Call LLM to get response"""
    response = completion(
        model="gemini/gemini-2.5-flash-lite-preview-06-17",
        messages=messages,
        max_tokens=4096
    )
    return response.choices[0].message.content


def extract_python_code(llm_response: str) -> List[str]:
    """
    Extract Python code blocks from LLM response text.
    
    Args:
        llm_response (str): The response text from an LLM that may contain code blocks
        
    Returns:
        List[str]: List of extracted Python code blocks (without the markdown markers)
    """
    # Pattern to match code blocks with optional 'python' language specifier
    # Matches both ```python and ``` code blocks, handling various whitespace scenarios
    pattern = r'```(?:python)?\s*(.*?)```'
    
    # Find all matches using DOTALL flag to match across newlines
    matches = re.findall(pattern, llm_response, re.DOTALL)
    
    # Clean up the extracted code (remove extra whitespace)
    # cleaned_code = []
    cleaned_code = """"""
    for match in matches:
        # Strip leading/trailing whitespace but preserve internal formatting
        code = match.strip()
        if code:  # Only add non-empty code blocks
            # cleaned_code.append(code)
            cleaned_code += code + '\n\n'
    
    return cleaned_code


code_generate_prompt = """you are an expert in python programming. user is going to provide you some requirements to write
python code. you need to think step by step what user is asking, what can be the edge scenario in users requirement and 
based on it you need to write/implement the python code along with type hints, error handling and edge case scenarios. python code
should be enclosed within ``` (backticks).
"""

code_document_generator_prompt = """write the documentation in google style for the python code.
 documentation should include the folloing:
- class description
- Function/method description
- Parameter descriptions
- Return value description
- Example usage
- Edge cases
"""

unit_test_case_generator_prompt = """you need to write the unit test cases for the user provided function. you don;t need to repeat the same function ,
just write unittest using unittest framework while covering the following:
- Basic functionality
- Edge cases
- Error cases
- Various input scenarios
"""


user_requirement = input("Please enter your requirement for which you want to generate the python code: ")

messages = [{'role': 'system', 'content': code_generate_prompt},
            {'role': 'user', 'content': user_requirement}]

code_generator_response = generate_response(messages)

parsed_code = extract_python_code(code_generator_response)
print(f'parsed code {"="*20}')
print(parsed_code)

messages.append(
    {'role': 'assistant', 'content': f"\`\`\`python\n\n"+ parsed_code+ "\n\n\`\`\`"}
)
messages.append(
    {'role': 'user', 'content': code_document_generator_prompt}
)

document_generator_response = generate_response(messages)
parsed_documented_code = extract_python_code(code_generator_response)
print(f'parsed documented code {"="*20}')
print(parsed_documented_code)

messages.append(
    {'role': 'assistant', 'content': f"\`\`\`python\n\n"+ parsed_documented_code+"\n\n\`\`\`"}
)
messages.append(
    {'role': 'user', 'content': unit_test_case_generator_prompt}
)

unittest_generator_response = generate_response(messages)
parsed_unittest_code = extract_python_code(unittest_generator_response)
print(f'parsed unittest  code {"="*20}')
print(parsed_unittest_code)

 # Generate filename from function description
filename = user_requirement.lower()
filename = ''.join(c for c in filename if c.isalnum() or c.isspace())
filename = filename.replace(' ', '_')[:30] + '.py'

# Save final version
with open(filename, 'w') as f:
    f.write(parsed_documented_code + '\n\n' + parsed_unittest_code)
