"""
Utilities for experiments: plotting ROC curves, simple list operations, and
text tokenization helpers (including removing last n words using HuggingFace or
`tiktoken`). Includes a small __main__ demo for quick testing.
"""

import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Any

def bool_to_first_upper(value: bool) -> str:
    return str(value)[0].upper()

# Function to plot ROC curve
def plot_roc_curve(fpr, tpr, roc_auc, strategy_title, save_path):
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')  # Diagonal line for random guess
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{strategy_title} ROC Curve')
    plt.grid(alpha=0.15)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

def zigzag_append(lists):
    """
    Interleaves elements from multiple lists in a zigzag pattern.
    
    This function takes a collection of lists and returns a new list where
    elements are arranged by taking the first element from each list, then
    the second element from each list, and so on.
    
    Parameters:
    -----------
    lists : list of lists
        A collection of lists to be interleaved
    
    Returns:
    --------
    list
        A single list containing all elements from the input lists in zigzag order
    
    Examples:
    ---------
    >>> zigzag_append([[1, 2, 3], ['a', 'b', 'c']])
    [1, 'a', 2, 'b', 3, 'c']
    
    >>> zigzag_append([[1, 2], ['a', 'b', 'c'], [True, False]])
    [1, 'a', True, 2, 'b', False]
    
    Notes:
    ------
    If the input lists have different lengths, elements from shorter lists
    will not be included after they're exhausted.
    """
    return [item for group in zip(*lists) for item in group]

def chunk_list(lst, n):
    """
    Splits a list into chunks of specified size.
    
    This function divides the input list into smaller sublists, each containing
    at most 'n' elements. The last chunk may contain fewer than 'n' elements
    if the length of the list is not divisible by 'n'.
    
    Parameters:
    -----------
    lst : list
        The list to be chunked
    n : int
        The size of each chunk
        
    Returns:
    --------
    list of lists
        A list containing sublists of up to 'n' elements each
        
    Examples:
    ---------
    >>> chunk_list([1, 2, 3, 4, 5, 6, 7], 3)
    [[1, 2, 3], [4, 5, 6], [7]]
    
    >>> chunk_list(['a', 'b', 'c', 'd'], 2)
    [['a', 'b'], ['c', 'd']]
    
    >>> chunk_list([1, 2, 3], 5)
    [[1, 2, 3]]
    
    Notes:
    ------
    If n <= 0, a ValueError should be raised (not handled by this function).
    Empty lists return an empty list of chunks.
    """
    return [lst[i:i + n] for i in range(0, len(lst), n)]

def remove_last_n_words(
        tokenizer,
        sentence: str, 
        n: int = 1,
        openai: bool = False
        ) -> Tuple[List[str], List[str], int]:
    """
    Tokenizes a sentence, removes the last n words using offset mapping to identify
    word boundaries, and returns the original text split at that boundary.
    
    Punctuation is currently grouped with the adjacent word. Words are identified by
    transitions from whitespace to non-whitespace characters.
    
    Args:
        tokenizer: The HuggingFace tokenizer to use
        sentence (str): The input sentence
        n (int): Number of words to remove from the end (default: 1)
            Will always leave at least the first word
        openai: Will use tiktoken, not huggingface tokenizer
            
    Returns:
        Tuple containing:
            - String of the remaining text (original text minus the last n words)
            - String of the removed text (just the last n words)
            - Number of tokens removed
    """
    if n == 0:
        return sentence, "", 0

    # TODO - currently punctuation is grouped with word - maybe remove this in the future (separate)

    if openai:
        tokenized = tokenize_with_offsets(sentence, tokenizer)
    else:
        tokenized = tokenizer(sentence, return_offsets_mapping=True, add_special_tokens=False)
        
    # Identify word boundaries based on spaces in the original text
    word_boundaries = []
    in_whitespace = False
    for i in range(len(sentence)):
        is_whitespace = sentence[i] in [' ', '\n', '\r', '\t']
        # If we're moving from whitespace to a non-whitespace character,
        # we're at the start of a new word
        if in_whitespace and not is_whitespace:
            word_boundaries.append(i)
        in_whitespace = is_whitespace
    
    # Add the start of the first word (if the text doesn't begin with whitespace)
    if not (sentence and sentence[0] in [' ', '\n', '\r', '\t']):
        word_boundaries.insert(0, 0)
    
    assert len(word_boundaries) >= 1
    n = min(n, len(word_boundaries)-1) # Leave at least the first word always
    
    last_words_start = word_boundaries[-n] if n > 0 else len(sentence)
    # Find the index of the first token that belongs to the removed part
    split_index = 0
    for i, (start, end) in enumerate(tokenized["offset_mapping"]):
        if end <= last_words_start:
            split_index = i + 1

    return (
        tokenizer.decode(tokenized["input_ids"][:split_index]),
        tokenizer.decode(tokenized["input_ids"][split_index:]),
        len(tokenized["input_ids"]) - split_index
    )

def tokenize_with_offsets(text, encoding):
    """
    Generate token IDs and offset mappings for a given text using tiktoken.
    
    Args:
        text: The input text string
        encoding: A tiktoken encoding object
    
    Returns:
        Dictionary containing:
        - input_ids: List of token IDs
        - offset_mapping: List of tuples with (start, end) character positions
    """
    token_ids = encoding.encode(text) # Get the token IDs

    # Get the byte offsets for each token
    byte_offsets = []
    byte_pos = 0
    for token_id in token_ids:
        # Get the bytes for this token
        token_bytes = encoding.decode_single_token_bytes(token_id)
        token_byte_len = len(token_bytes)
        
        # Record the byte position
        byte_offsets.append((byte_pos, byte_pos + token_byte_len))
        byte_pos += token_byte_len
    
    # Convert byte offsets to character offsets
    char_offsets = []
    text_bytes = text.encode('utf-8')
    for start_byte, end_byte in byte_offsets:
        # Find the character position corresponding to this byte position
        start_char = len(text_bytes[:start_byte].decode('utf-8', errors='ignore'))
        end_char = len(text_bytes[:end_byte].decode('utf-8', errors='ignore'))
        char_offsets.append((start_char, end_char))

    return {
        "input_ids": token_ids,
        "offset_mapping": char_offsets
    }

# Example usage
if __name__ == "__main__":

    import tiktoken
    # Example usage:
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    import nltk
    import os
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    CACHE_PATH = os.getenv("CACHE_PATH")
    
    os.environ["HF_HOME"] = CACHE_PATH
    os.environ["HF_DATASETS_PATH"] = CACHE_PATH
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
    tokenize_func = lambda x: nltk.tokenize.casual.casual_tokenize(x)    
    sentences = [
        "Oh no!!!!",
        "Natural language processing is an exciting field of artificial intelligence",
        "   Natural language processing is an exciting field of artificial intelligence",
        "We went to the store. Then we went fishing!",
        "ARTICLE\n\nMen go to the store often.\n\nBut do women?\tWe are not sure about that."
    ]
    
    for sentence in sentences:
        output = tokenize_with_offsets(sentence, encoding)
        print(output)

        for i in [1, 3, 5, 20]:
            # Remove last word
            remaining, removed, num_removed = remove_last_n_words(tokenizer, sentence, i)
            print(f"Original: {sentence}")
            print(f"Removing {i} words")
            print(f"Remaining tokens: {remaining}")
            print(f"Removed tokens: {removed}")
            print(f"Number of tokens removed: {num_removed}")

            remaining, removed, num_removed = remove_last_n_words(encoding, sentence, i, openai=True)
            print("\nTiktokenizer")
            print(f"Remaining tokens: {remaining}")
            print(f"Removed tokens: {removed}")
            print(f"Number of tokens removed: {num_removed}")


"""
python3 -m code.experiments.utils
"""