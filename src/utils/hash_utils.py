# src/utils/hash_utils.py

import hashlib

def generate_query_hash(query: str, length: int = 10) -> str:
    return hashlib.sha256(query.encode()).hexdigest()[:length]
