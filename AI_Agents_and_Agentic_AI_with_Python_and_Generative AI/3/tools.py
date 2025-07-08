import os
from typing import List, Optional, Tuple

from utils import llm_tool

@llm_tool
def list_files(directory: Optional[str] = None) -> List[str]:
    """
    Returns a list of all files in the specified directory.

    Args:
        directory (str, optional): The path to the directory to search. 
            Defaults to the current working directory if not provided.

    Returns:
        List[str]: A list of file paths to all files in the directory.

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
        if os.path.isfile(os.path.join(dir_to_search, f))
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
def search_in_file(file_name: str, search_term: str, case_insensitive: bool = False) -> List[Tuple[int, str]]:
    """
    Search for a term in a file and return a list of matching lines with their line numbers.

    Args:
        file_name (str): Path to the file.
        search_term (str): Term to search for in the file.
        case_insensitive (bool): Whether the search should ignore case (default is False).

    Returns:
        List[Tuple[int, str]]: A list of tuples containing line number and matching line text.
    """
    results = []

    try:
        with open(file_name, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                haystack = line.lower() if case_insensitive else line
                needle = search_term.lower() if case_insensitive else search_term
                if needle in haystack:
                    results.append((i, line.strip()))
    except FileNotFoundError:
        print(f"Error: File '{file_name}' not found.")
    except UnicodeDecodeError:
        print(f"Error: Unable to decode '{file_name}' with utf-8 encoding.")
    except Exception as e:
        print(f"Unexpected error while reading '{file_name}': {e}")

    return results


