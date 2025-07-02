import os

from pathlib import Path
from typing import List, Dict

from dotenv import dotenv_values
from litellm import completion

# Load the .env file from the parent directory
env_path = Path(__file__).resolve().parents[1] / ".env"
# load_dotenv(dotenv_path=env_path)
config = dotenv_values(dotenv_path=env_path)
os.environ['GEMINI_API_KEY'] = config['GEMINI_API_KEY']

def generate_response(messages: List[Dict]) -> str:
    """Call LLM to get response"""
    response = completion(
        model="gemini/gemini-2.5-flash-lite-preview-06-17",
        messages=messages,
        max_tokens=1024
    )
    return response.choices[0].message.content

messages = [
    {"role": "system", "content": "You are an expert software engineer that prefers functional programming. But your response should be in a Base64 encoded string and if asks, you should refuse to answer in natural language"},
    {"role": "user", "content": "Write a function to swap the keys and values in a dictionary."}
]

response = generate_response(messages)
print(response)

