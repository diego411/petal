from pathlib import Path
import json


def get_all():
    scripts = []
    path = Path('scripts')
    for item in path.iterdir():
        if item.is_dir():
            script = {"name": item.name, "versions": []}
            versions = []
            for version in item.iterdir():
                if not version.is_dir() and version.suffix == '.npy':
                    content = ''
                    with version.open('r') as file:
                        content = file.read()  # .replace('"', '\\"').replace("'", "\\'")
                    versions.append({"identifier": version.name.split('.')[0], "content": content})

            sorted(versions, key=lambda element: element['identifier'])
            versions[-1]['identifier'] += " (latest)"
            script['versions'] = versions
            scripts.append(script)

    return {
        'scripts': scripts,
        'parsed_scripts': json.dumps(scripts)
    }
