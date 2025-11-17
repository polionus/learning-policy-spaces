from typing import List
import json
import hashlib

def hash_list(d: List) -> str:
    # Serialize the dict to a JSON string with sorted keys
    serialized = json.dumps(d, sort_keys=True).encode('utf-8')
    return hashlib.sha256(serialized).hexdigest()