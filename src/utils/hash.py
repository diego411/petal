import hashlib


def hash_file_name(file_name: str) -> str:
    hash_object = hashlib.md5(file_name.encode())
    return hash_object.hexdigest()