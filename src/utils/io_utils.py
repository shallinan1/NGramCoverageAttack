"""
Input/Output utilities for JSON and JSONL file operations.

This module provides functions for loading and saving JSON and JSONL files,
with progress bars for large datasets.
"""
import json
from tqdm import tqdm

def load_jsonl(file_path):
    """
    Load data from a JSONL (JSON Lines) file.
    
    Each line in the file should contain a valid JSON object.
    Shows a progress bar while loading.
    
    Args:
        file_path (str): Path to the JSONL file to load
        
    Returns:
        list: List of dictionaries, one per line in the file
        
    Example:
        >>> data = load_jsonl('dataset.jsonl')
        >>> print(len(data))
        1000
    """
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in tqdm(f, "Loading jsonl")]
    return data

def load_json(file_path):
    """
    Load data from a standard JSON file.
    
    Args:
        file_path (str): Path to the JSON file to load
        
    Returns:
        dict or list: The loaded JSON data
        
    Example:
        >>> config = load_json('config.json')
        >>> print(config['model_name'])
        'gpt-3.5-turbo'
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def save_to_jsonl(data, file_path):
    """
    Save a list of dictionaries to a JSONL file.
    
    Each dictionary is written as a separate JSON object on its own line.
    
    Args:
        data (list): List of dictionaries to save
        file_path (str): Path where the JSONL file will be saved
        
    Returns:
        None
        
    Example:
        >>> data = [{'id': 1, 'text': 'hello'}, {'id': 2, 'text': 'world'}]
        >>> save_to_jsonl(data, 'output.jsonl')
        Data saved to output.jsonl in JSONL format.
    """
    with open(file_path, "w") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")
    print(f"Data saved to {file_path} in JSONL format.")