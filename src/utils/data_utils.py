"""
Utilities for combining and manipulating data structures.

This module provides functions for combining lists and dictionaries,
particularly useful for merging data from multiple sources while
preserving structure and validating consistency.
"""

def combine_lists(list1, list2):
    """
    Combine two lists element-wise by concatenating corresponding elements.
    
    Args:
        list1 (list): First list of sequences (strings, lists, etc.)
        list2 (list): Second list of sequences to combine with first
        
    Returns:
        list: A new list where each element is the concatenation of
              corresponding elements from list1 and list2
              
    Example:
        >>> combine_lists(['a', 'b'], ['1', '2'])
        ['a1', 'b2']
    """
    output_list = []
    for l1, l2 in zip(list1, list2):
        output_list.append(l1 + l2)
    return output_list

def combine_dicts(dict1, dict2):
    """
    Combine two dictionaries by merging their values.
    
    This function removes 'logprobs' and 'model' keys from both dictionaries,
    then combines the remaining keys. List values are concatenated, while
    non-list values must be equal between dictionaries.
    
    Args:
        dict1 (dict): First dictionary to combine
        dict2 (dict): Second dictionary to combine
        
    Returns:
        dict: Combined dictionary with merged values
        
    Raises:
        AssertionError: If dictionaries have different keys or non-list values differ
        
    Example:
        >>> d1 = {'texts': ['a'], 'count': 5}
        >>> d2 = {'texts': ['b'], 'count': 5}
        >>> combine_dicts(d1, d2)
        {'texts': ['a', 'b'], 'count': 5}
    """
    # Remove ML-specific keys that shouldn't be combined
    for cur_key in ['logprobs', 'model']:
        if cur_key in dict1:
            dict1.pop(cur_key)
        if cur_key in dict2:
            dict2.pop(cur_key)

    # Validate that both dictionaries have the same keys
    assert dict1.keys() == dict2.keys(), f"Keys do not match: {dict1.keys()} != {dict2.keys()}"
    
    combined_dict = {}
    
    for key in dict1.keys():
        if isinstance(dict1[key], list):
            # Combine list values by concatenation
            combined_dict[key] = dict1[key] + dict2[key]
        else:
            # Ensure non-list values are equal
            assert dict1[key] == dict2[key], f"Values for '{key}' do not match: {dict1[key]} != {dict2[key]}"
            combined_dict[key] = dict1[key]
    
    return combined_dict
    
def combine_list_of_dicts(list1, list2):
    """
    Combine two lists of dictionaries element-wise.
    
    Each dictionary at position i in list1 is combined with the dictionary
    at position i in list2 using the combine_dicts function.
    
    Args:
        list1 (list[dict]): First list of dictionaries
        list2 (list[dict]): Second list of dictionaries
        
    Returns:
        list[dict]: List of combined dictionaries
        
    Raises:
        AssertionError: If lists have different lengths
        
    Example:
        >>> l1 = [{'texts': ['a'], 'id': 1}]
        >>> l2 = [{'texts': ['b'], 'id': 1}]
        >>> combine_list_of_dicts(l1, l2)
        [{'texts': ['a', 'b'], 'id': 1}]
    """
    # Ensure both lists have the same length
    assert len(list1) == len(list2), f"Lists must have the same length: {len(list1)} != {len(list2)}"
    
    combined_list = []

    # Iterate through both lists simultaneously and combine each pair of dictionaries
    for dict1, dict2 in zip(list1, list2):
        combined_dict = combine_dicts(dict1, dict2)
        combined_list.append(combined_dict)
    
    return combined_list