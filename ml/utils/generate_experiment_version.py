from pathlib import Path
import os
import sys


def get_latest_version(path_str: str):
    if not os.path.exists(path_str):
        return 'version_0'

    path: Path = Path(path_str)
    directories = [dir for dir in path.iterdir() if dir.is_dir() and dir.stem.startswith('v')]
    return f'version_{len(directories) - 1}'
    
def get_experiment_version(path_str: str):
    if not os.path.exists(path_str):
        return 'version_0'

    path: Path = Path(path_str)
    directories = [dir for dir in path.iterdir() if dir.is_dir() and dir.stem.startswith('v')]
    return f'version_{len(directories)}'


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: generate_experiment_version.py <path>")
        sys.exit(1)

    version = get_experiment_version(sys.argv[1])
    print(version)