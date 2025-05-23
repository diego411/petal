from pathlib import Path
from typing import Optional

def find_file_path(file_name: str, directory: Path) -> Optional[Path]:
    """
    Searches for a file with the given name in the specified directory and its subdirectories.

    Args:
        file_name (str): The name of the file to search for.
        directory (str): The path to the directory where the search should begin.

    Returns:
        Path | None: The full path to the file if found, otherwise None.
    """
    for path in directory.rglob(file_name):
        if path.is_file():
            return path
    return None
