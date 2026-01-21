import os
from pathlib import Path

def get_prompt(text: str) -> str:
    """
    Write text to prompt.txt file in the project root directory.
    
    Args:
        text: The text content to write to the file
        
    Returns:
        The path to the created file
    """
    # Get project root directory (scripts/ folder is one level down from project root)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    file_path = project_root / "prompt.txt"
    
    # Write the text to the file
    with open(file_path, 'w') as file:
        file.write(text)
    
    return str(file_path)