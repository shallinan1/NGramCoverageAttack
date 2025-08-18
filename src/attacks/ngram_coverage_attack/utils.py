"""
Utilities for text processing and chunking in the n-gram coverage attack.

This module provides functions for extracting specific portions of text based on
sentence boundaries while preserving the original formatting (newlines).

Requirements:
    Before using this module, you must download the NLTK punkt tokenizer:
    >>> import nltk
    >>> nltk.download('punkt')
"""

from nltk import sent_tokenize
from typing import List, Tuple, Optional

def split_text_and_newlines(text: str) -> Tuple[List[str], List[int]]:
    """
    Split text by newlines while tracking the number of consecutive newlines.
    
    This function separates text into non-empty segments and keeps track of how
    many newlines appeared between each segment. This is useful for preserving
    the original formatting when reconstructing text.
    
    Args:
        text: The input text containing potential newline characters.
    
    Returns:
        A tuple containing:
            - texts: List of non-empty text segments (stripped of whitespace)
            - newline_counts: List of integers indicating the number of newlines
                            after each text segment (except the last one)
    
    Example:
        >>> text = "Hello\\n\\nWorld\\nPython"
        >>> texts, counts = split_text_and_newlines(text)
        >>> texts
        ['Hello', 'World', 'Python']
        >>> counts
        [2, 1]
    """
    texts = []
    newline_counts = []
    parts = text.split("\n")
    current_text = parts[0]
    newline_count = 0

    texts.append(current_text.strip())
    
    for part in parts[1:]:
        newline_count += 1
        
        if part.strip() != "":
            texts.append(part.strip())
            newline_counts.append(newline_count)
            newline_count = 0

    return texts, newline_counts

def extract_chunk_words(text, start_word, num_words):
    pass # TODO fill

def extract_chunk_sentence(text: str, start_sentence: int, num_sentences: int) -> Tuple[Optional[str], str]:
    """
    Extract a chunk of text based on sentence boundaries while preserving formatting.
    
    This function extracts a specified number of sentences starting from a given
    position, preserving the original newline structure of the text. It's designed
    for creating prompts from text while maintaining the distinction between what's
    used as prompt and what remains.
    
    Args:
        text: The input text to extract sentences from.
        start_sentence: The index of the first sentence to extract (0-indexed).
        num_sentences: The maximum number of sentences to extract.
    
    Returns:
        A tuple containing:
            - prompt_text: The extracted sentences as a single string, or None if
                         extraction fails (e.g., invalid indices).
            - rest_of_text: The remaining text after the extracted sentences.
    
    Behavior:
        - Preserves newline formatting between paragraphs
        - Joins sentences within paragraphs with spaces
        - Adjusts num_sentences if it would exceed available sentences
        - Returns at least one sentence if possible (for generation)
    
    Example:
        >>> text = "First sentence. Second sentence.\\n\\nThird sentence."
        >>> prompt, rest = extract_chunk_sentence(text, 0, 2)
        >>> prompt
        'First sentence. Second sentence.'
        >>> rest
        '\\n\\nThird sentence.'
    
    Edge Cases:
        - If start_sentence is beyond available sentences, returns None for prompt
        - If num_sentences would go past the last sentence, it's capped at the
          second-to-last sentence (leaving at least one for generation)
    """

    # Split the text by new lines and remember the spacing
    split_texts, newline_counts = split_text_and_newlines(text)

    text_sentence_count = 0
    for cur_split_text in split_texts:
        text_sentences = sent_tokenize(cur_split_text)
        text_sentence_count += len(text_sentences)

    # Make it so we at least have the last sentence to generate
    num_sentences = min(text_sentence_count - start_sentence - 1, num_sentences)
    num_sentence_copy = num_sentences

    if text_sentence_count == 1:
        try:
            prompt_text = text_sentences[start_sentence]
        except:
            # embed()
            print("Something wrong")
            return None
          
    cur_sentence_count, total_sents = 0, 0
    prompt_text, rest_of_text = "", ""

    for i, cur_split_text in enumerate(split_texts):
        text_sentences = sent_tokenize(cur_split_text)
        cur_split_text_length = len(text_sentences)
        assert num_sentences >= 0
        assert cur_sentence_count <= num_sentence_copy

        if cur_sentence_count < num_sentence_copy: # Need to build the prompt
          if cur_sentence_count == 0: # Edge case - starting
            if total_sents + cur_split_text_length < start_sentence + 1:
              total_sents += cur_split_text_length
              continue # Continue if we haven't yet reached the start idx
            else:
              start_idx = start_sentence - total_sents
              assert start_idx >= 0 and start_idx < cur_split_text_length
          else:
            start_idx = 0

          sents_to_add = min(num_sentences, cur_split_text_length - start_idx)
          final_idx = start_idx + sents_to_add
          assert final_idx <= cur_split_text_length

          prompt_text += " ".join(text_sentences[start_idx:final_idx])
          num_sentences -= sents_to_add

          cur_sentence_count += sents_to_add
          total_sents += cur_split_text_length

          if final_idx == sents_to_add and num_sentences > 0:
              prompt_text += newline_counts[i] * "\n"
          else:
            assert num_sentences == 0
            rest_of_text += " ".join(text_sentences[final_idx:])
            if i < len(newline_counts):
              rest_of_text += newline_counts[i] * "\n"
            else: 
              assert i == len(newline_counts) 

        else: # We're done building the prompt - build rest_of_text
          rest_of_text += " ".join(text_sentences)
          if i < len(newline_counts):
            rest_of_text += newline_counts[i] * "\n"
          else: 
            assert i == len(newline_counts) 

        assert num_sentences + cur_sentence_count == num_sentence_copy

    return prompt_text, rest_of_text


def run_tests():
    """Run comprehensive tests for the text extraction functions."""
    
    # Test data with various formatting scenarios
    test_cases = [
        ("Multi-paragraph with mixed formatting", 
         "Hello\n\n\nWorld. The world is bright \nhahaha...wohoo\nPython\n\nis\nGreat"),
        ("Short text with newlines", 
         "hey\n\hah\n\nhoh\n\nhehe\nhar\n\ntest"),
        ("Math text with incomplete sentence", 
         "Math\n\nThe eigenvector associating zero eigenvalue of is . Remember what mean? If and is collinear,"),
        ("Complete math sentence", 
         "The eigenvector associating zero eigenvalue of is . Remember what mean? If and is collinear.")
    ]
    
    # Test parameters: (start_sentence, num_sentences, description)
    test_params = [
        (0, 4, "Extract first 4 sentences"),
        (2, 1, "Extract 1 sentence starting from index 2"),
        (1, 1, "Extract 1 sentence starting from index 1"), 
        (3, 10, "Extract 10 sentences starting from index 3 (boundary test)"),
        (0, 10, "Extract 10 sentences from beginning (boundary test)"),
        (0, 2, "Extract first 2 sentences"),
        (0, 1, "Extract first sentence only")
    ]
    
    print("Testing extract_chunk_sentence function")
    print("=" * 50)
    
    for case_name, text in test_cases:
        print(f"\nTest Case: {case_name}")
        print(f"Input text: {repr(text)}")
        print("-" * 30)
        
        for start_idx, num_sents, description in test_params:
            try:
                prompt, rest = extract_chunk_sentence(text, start_idx, num_sents)
                print(f"{description}:")
                print(f"  Prompt: {repr(prompt)}")
                print(f"  Rest: {repr(rest)}")
            except Exception as e:
                print(f"{description}: ERROR - {e}")
        
        print()


if __name__ == "__main__":
    run_tests()