#!/usr/bin/env python3

import sys
import yaml
import json
import hashlib

def hash_yaml(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)  # Parse YAML into a dictionary

    json_data = json.dumps(data, sort_keys=True, separators=(",", ":"))  # Convert to JSON, sorted
    hash_value = hashlib.md5(json_data.encode("utf-8")).hexdigest()  # Compute SHA-256 hash
    print(hash_value)  # Output the hash
    return hash_value

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: generate_experiment_name.py <yaml_file>")
        sys.exit(1)
    hash_yaml(sys.argv[1])
