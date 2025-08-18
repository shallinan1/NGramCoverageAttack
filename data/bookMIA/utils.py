import re

# Function to clean the 'snippet' by replacing '\x00' with an empty string for huggingface BookMIA dataset
def clean_snippet(snippet):
    if isinstance(snippet, bytes):  # Decode if snippet is in bytes
        snippet = snippet.decode('utf-8', errors='ignore')
    return re.sub(r'\x14', '',re.sub(r'\x00', '', snippet))
