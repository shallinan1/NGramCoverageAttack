"""
Task-specific utilities for text formatting and processing.

This module provides functions for converting conversation formats
and performing text preprocessing tasks specific to the membership
inference pipeline.
"""
import re

def convert_to_tulu_v1_format(messages, turns=1):
    """
    Convert conversation messages to Tulu v1 chat format.
    
    Formats a conversation with user and assistant messages using
    the Tulu v1 template format with role tags.
    
    Args:
        messages (list[dict]): List of message dictionaries with 'role' and 'content' keys
        turns (int, optional): Number of conversation turns to include. Defaults to 1.
                              Each turn consists of 2 messages (user + assistant)
    
    Returns:
        str: Formatted conversation string with Tulu v1 role tags
        
    Example:
        >>> messages = [
        ...     {"role": "user", "content": "Hello"},
        ...     {"role": "assistant", "content": "Hi there!"}
        ... ]
        >>> convert_to_tulu_v1_format(messages)
        '<|user|>\nHello\n<|assistant|>\nHi there!'
    """
    messages = messages[:2*turns]  # Limit to specified number of turns
    final_text = ""
    for c in messages:
        if c["role"] == "user":
            final_text += f"<|user|>\n"
        else:
            final_text += f"<|assistant|>\n"

        final_text += c["content"] + "\n"

    return final_text.strip()

def convert_to_tulu_v1_open(text):
    """
    Convert text to an open Tulu v1 format for completion.
    
    Wraps the input text in user tags and adds an assistant tag
    for model completion/generation tasks.
    
    Args:
        text (str): Input text to format as a user message
        
    Returns:
        str: Formatted string ready for model completion
        
    Example:
        >>> convert_to_tulu_v1_open("What is 2+2?")
        '<|user|>\nWhat is 2+2?<|assistant|>\n'
    """
    return f"<|user|>\n{text}<|assistant|>\n"

def remove_first_sentence_if_needed(text):
    """
    Remove the first sentence from text if it doesn't start properly.
    
    Removes the first sentence if it doesn't begin with an uppercase letter
    or quotation mark, which may indicate it's a continuation or fragment.
    
    Args:
        text (str): Input text to process
        
    Returns:
        str: Text with first sentence removed if criteria met, otherwise original text
        
    Example:
        >>> remove_first_sentence_if_needed("this is lowercase. This is proper.")
        'This is proper.'
        >>> remove_first_sentence_if_needed("This is proper. More text.")
        'This is proper. More text.'
    """
    # Match the first sentence using regex
    match = re.match(r"([^.!?]+[.!?])\s*(.*)", text, re.DOTALL)
    
    if match:
        first_sentence, rest = match.groups()
        # Check if the first character is not uppercase or a quotation mark
        if not first_sentence.lstrip()[0].isupper() and not first_sentence.lstrip().startswith(("'", '"')):
            return rest.strip()  # Return the rest without the first sentence
    return text  # Return the original text if no change is needed