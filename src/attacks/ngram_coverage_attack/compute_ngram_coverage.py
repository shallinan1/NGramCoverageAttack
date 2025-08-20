"""
N-gram Coverage Computation for Membership Inference Attack

This module implements the coverage computation component of the N-Gram Coverage Attack 
method for membership inference attacks against language models.

The module analyzes generated text continuations by computing exact n-gram matches
against source documents and calculating coverage statistics for membership inference.
It uses dynamic programming for efficient longest common substring/subsequence computation
and supports parallel processing for large-scale analysis.

Pipeline:
    1. Load generated text continuations from JSONL file
    2. Tokenize both generated texts and source documents using NLTK
    3. Find exact n-gram matches using sliding window approach
    4. Calculate coverage percentage (portion of text covered by matching n-grams)
    5. Compute longest common substring (character-level) and subsequence (word-level) metrics
    6. Save results with metadata for downstream analysis

Outputs:
    JSONL file containing:
    - Coverage statistics (percentage of text matching source)
    - Matched span information (start/end indices, text, occurrence count)
    - Length metrics (character and word counts)
    - Longest common substring/subsequence lengths

Hardcoded Configuration:
    - Tokenization: NLTK casual tokenizer for word-level processing
    - Detokenization: Moses detokenizer for English text reconstruction
    - Parallel processing: Configurable worker count (default max 4 CPUs)

Dependencies:
    - Environment variable CACHE_PATH for HuggingFace cache location
    - NLTK tokenizers and Moses detokenizer
    - pylcs for efficient longest common substring computation

Usage:
    python -m src.attacks.ngram_coverage_attack.compute_ngram_coverage \\
        --gen_data PATH_TO_GENERATIONS.jsonl \\
        --output_dir OUTPUT_DIRECTORY \\
        --min_ngram 3 \\
        [--subset 1000] \\
        [--generation_field "generation"] \\
        [--parallel]
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
CACHE_PATH = os.getenv("CACHE_PATH")

# Set up environment variables
os.environ["HF_HOME"] = CACHE_PATH
os.environ["HF_DATASETS_PATH"] = CACHE_PATH

import nltk
import json
import argparse
from src.utils.io_utils import save_to_jsonl
import numpy as np
from tqdm import tqdm
from typing import Callable
from unidecode import unidecode
from sacremoses import MosesDetokenizer
from datasets import Dataset
import datasets
from dataclasses import dataclass
from datasets.utils.logging import disable_progress_bar
from typing import List
from multiprocessing import Pool, cpu_count
import time
import logging
from pylcs import lcs_string_length
from src.attacks.ngram_coverage_attack.utils import (
    find_max_common_sublist_length,
    longest_sublist,
    longest_substring,
    process_single_sublist_pair,
    process_single_substring_pair
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

datasets.logging.set_verbosity_error()
disable_progress_bar() # Disable filter progress bar
md = MosesDetokenizer(lang='en')

# Define global tokenization and detokenization functions
tokenize_func = lambda x: nltk.tokenize.casual.casual_tokenize(x)
detokenize = lambda x: md.detokenize(x)

@dataclass
class Document:
    """Represents a tokenized document for n-gram analysis.
    
    Attributes:
        doc_id: Unique identifier for the document
        tokens: List of string tokens from tokenization
    """
    doc_id: str
    tokens: List[str]  # [num_tokens]

@dataclass
class Span:
    """Represents a matched text span in the source document.
    
    Attributes:
        start_index: Token index where the span begins (inclusive)
        end_index: Token index where the span ends (exclusive)
        span_text: The actual text content of the span
        occurrence: Number of times this span appears in source documents
    """
    start_index: int
    end_index: int
    span_text: str
    occurrence: int

class Hypothesis:
    """Tracks n-gram coverage hypothesis for a target document.
    
    Maintains a collection of matched spans and computes coverage statistics
    for membership inference analysis.
    
    Attributes:
        target_doc: The document being analyzed
        min_ngram: Minimum n-gram size for matching
        spans: List of matched Span objects
        finished: Whether processing has reached document end
    """
    def __init__(self, target_doc: Document, min_ngram: int) -> None:
        self.target_doc = target_doc
        self.min_ngram = min_ngram
        self.spans = []
        self.finished = False

    def add_span(self, new_span: Span) -> None:
        self.spans.append(new_span)
        if new_span.end_index >= len(self.target_doc.tokens):
            self.finished = True

    def replace_span(self, new_span: Span) -> None:
        self.spans = self.spans[:-1] + [new_span]
        if new_span.end_index >= len(self.target_doc.tokens):
            self.finished = True

    def get_score(self) -> float:
        if not self.spans:
            return 0
        progress_len = self.spans[-1].end_index if not self.finished else len(self.target_doc.tokens)
        flags = [False for _ in range(progress_len)]
        for span in self.spans:
            span_length = span.end_index - span.start_index
            flags[span.start_index: span.end_index] = [True] * span_length
        coverage = len([fa for fa in flags if fa]) / len(flags)
        return coverage

    def format_span(self) -> str:
        return ' | '.join([s.span_text for s in self.spans])

    def __hash__(self) -> int:
        return hash(self.format_span())

    def __eq__(self, other) -> bool:
        if isinstance(other, Hypothesis):
            return self.format_span() == other.format_span()
        return NotImplemented

    def get_avg_span_len(self) -> float:
        if not self.spans:
            return 0
        span_len = [s.end_index - s.start_index for s in self.spans]
        return sum(span_len) / len(span_len)

    def export_json(self) -> dict:
        matched_spans = [{'start_index': s.start_index,
                          'end_index': s.end_index,
                          'span_text': s.span_text,
                          'occurrence': s.occurrence} for s in self.spans]
        return {
            'matched_spans': matched_spans,
            'coverage': self.get_score(),
            'avg_span_len': self.get_avg_span_len(),
        }

def find_exact_match(detokenize: Callable[[List[str]], str], 
                    doc: Document, 
                    min_ngram: int, 
                    source_data: Dataset, 
                    num_cpus: int) -> dict:
    """Find all exact n-gram matches between document and source data.
    
    Uses a sliding window approach to identify matching n-grams and builds
    a coverage hypothesis by extending or replacing spans.
    
    Args:
        detokenize: Function to convert tokens back to text
        doc: Target document to analyze
        min_ngram: Minimum n-gram size for matching
        source_data: HuggingFace Dataset containing source documents
        num_cpus: Number of CPUs to use for parallel filtering
    
    Returns:
        Dictionary containing:
            - matched_spans: List of span dictionaries with position and text
            - coverage: Percentage of document covered by matches (0.0 to 1.0)
            - avg_span_len: Average length of matched spans in tokens
    
    Raises:
        ValueError: If pointer arithmetic enters invalid state
    """
    hypothesis = Hypothesis(doc, min_ngram)

    first_pointer, second_pointer = 0, min_ngram
    while second_pointer <= len(doc.tokens):
        span_text = detokenize(doc.tokens[first_pointer: second_pointer])
        hit_data = source_data.filter(lambda x: span_text in x['text'], num_proc=num_cpus)
        occurrence = len(hit_data)

        if occurrence:
            matched_span = Span(start_index=first_pointer,
                                end_index=second_pointer,
                                span_text=span_text,
                                occurrence=occurrence)
            if not hypothesis.spans:
                hypothesis.add_span(matched_span)
            else:
                last_span = hypothesis.spans[-1]
                if matched_span.start_index <= last_span.start_index and last_span.end_index <= matched_span.end_index:
                    hypothesis.replace_span(matched_span)
                else:
                    hypothesis.add_span(matched_span)
            second_pointer += 1
        else:
            if second_pointer - first_pointer > min_ngram:
                first_pointer += 1
            elif second_pointer - first_pointer == min_ngram:
                first_pointer += 1
                second_pointer += 1
            else:
                raise ValueError

    hypothesis.finished = True
    return hypothesis.export_json()

def process_single_doc(t_idx: int, 
                      all_gens: List[dict], 
                      min_ngram: int, 
                      source_docs: Dataset) -> List[dict]:
    """Process all generations for a single target document.
    
    Args:
        t_idx: Index of the target document
        all_gens: List of generation dictionaries containing "text" field
        min_ngram: Minimum n-gram size for matching
        source_docs: Dataset containing source documents for comparison
    
    Returns:
        List of dictionaries with original generation data plus coverage metrics
    """
    outputs = []
    for t_doc in tqdm(all_gens, leave=False, position=1):
        tokenized_text = tokenize_func(unidecode(t_doc["text"]))
        tgt_doc = Document(f'tgt_{t_idx}', tokenized_text)
        if len(tgt_doc.tokens) < min_ngram: # Edge case if gen is too short
            output = {'matched_spans': [], 'coverage': 0, 'avg_span_len': 0, 'too_short': True}
        else:
            output = find_exact_match(detokenize, tgt_doc, min_ngram, source_docs[t_idx], 1) # Hardcode 1 CPU here
        # Add it to the existing dict which stores the text
        t_doc.update(output)
        outputs.append(t_doc)

    return outputs

def dj_search(generation_texts_list: List[List[str]],
              source_docs: List[Dataset], 
              min_ngram: int, 
              subset: int = None, 
              num_cpus: int = 1) -> List[List[dict]]:
    """Perform document search with n-gram matching across multiple generations.
    
    Coordinates parallel or sequential processing of generated texts against
    source documents to compute coverage statistics.
    
    Args:
        generation_texts_list: Nested list of generated text strings
        source_docs: List of HuggingFace Datasets containing source documents
        min_ngram: Minimum n-gram size for matching
        subset: Maximum number of items to process (None for all)
        num_cpus: Number of CPUs for parallel processing (1 for sequential)
    
    Returns:
        Nested list of dictionaries containing coverage analysis results
    """
    data = [[{"text": g} for g in generation_texts] for generation_texts in generation_texts_list]
    data = data[:subset] if subset is not None else data

    all_outputs = []

    if num_cpus > 1:
        combinations = [(t_idx, all_gens, min_ngram, source_docs) for t_idx, all_gens in enumerate(data)]
        logger.info(f"Launching search in parallel with {num_cpus} on {len(combinations)} inputs")
        with Pool(num_cpus) as pool:
            all_outputs = list(pool.starmap(process_single_doc, tqdm(combinations, total=len(combinations), position=0)))
    else:
        logger.info("Launching search iteratively")
        for t_idx, all_gens in tqdm(enumerate(data), desc='target gens', total=len(data)):         
            outputs = process_single_doc(t_idx, all_gens, min_ngram, source_docs)
            all_outputs.append(outputs)

    return all_outputs

def main(args: argparse.Namespace) -> None:
    """Main entry point for n-gram coverage computation.
    
    Loads generated texts, computes coverage statistics against source documents,
    calculates longest common substring/subsequence metrics, and saves results
    to JSONL format.
    
    Args:
        args: Command-line arguments containing:
            - gen_data: Path to input JSONL file
            - output_dir: Directory for output files
            - min_ngram: Minimum n-gram size
            - subset: Max examples to process
            - generation_field: Field name containing generations
            - parallel: Whether to use multiprocessing
    
    Side Effects:
        - Creates output directory if it doesn't exist
        - Writes results to JSONL file in output directory
        - Prints execution time and output filename
    """
    start_time = time.time()
    os.makedirs(args.output_dir, exist_ok=True)  # Ensure the output directory exists

    # Make the name of the output file the same as the input file but in the args.output_dir
    input_filename = os.path.basename(args.gen_data)
    args.output_file = os.path.join(args.output_dir, input_filename).replace(".jsonl", f"_{args.min_ngram}.jsonl")

    generations = []
    with open(args.gen_data, 'r') as f:  # Load the input data from a JSONL file
        for line in f:
            generations.append(json.loads(line.strip()))
    generation_texts = [g[args.generation_field] for g in generations]

    # Merge the list of lists
    num_sequences = len(generation_texts[0][0])
    generation_texts = [sum(g, []) for g in generation_texts]

    # We only use the original snippet for the source docs, but we could use more documents as well (need to modify source_docs to do so)
    source_docs = [Dataset.from_dict({"text": [unidecode(g["snippet_no_prompt"])]}) for g in generations]
    args.output_file = args.output_file.replace(".jsonl", "_onedoc.jsonl")
    logger.info(f"Output file: {args.output_file}")

    num_workers = min(cpu_count(), 4) if args.parallel else 1 # Set to 4 since it will get killed with too many cpus
    # Run the n-gram coverage computation
    all_outputs = dj_search(generation_texts, source_docs, args.min_ngram, args.subset, num_workers)

    execution_time = time.time() - start_time
    minutes, seconds = int(execution_time // 60), int(execution_time % 60)
    logger.info(f"Program executed in {minutes}:{seconds:02d} ({execution_time:.4f} seconds)")

    assert len(all_outputs) == len(generations)
    for cur_generation, cur_all_output in zip(generations, all_outputs):
        cur_generation["coverage"] = cur_all_output

    generation_texts_word_tokenized = [[tokenize_func(g) for g in g_list] for g_list in generation_texts] 
    source_docs_word_tokenized = [[tokenize_func(s) for s in source["text"]] for source in source_docs]
    
    # Length (number of words) and characters
    generation_texts_length_chars = [[len(g) for g in g_list] for g_list in generation_texts]
    generation_texts_length_words = [[len(g) for g in g_list] for g_list in generation_texts_word_tokenized]

    if num_workers > 1: # Multi-processing
        combinations_substring = [(g_list, source["text"]) for g_list, source in zip(generation_texts, source_docs)]
        with Pool(num_workers) as pool:
            longest_substring_chars = list(pool.starmap(process_single_substring_pair, tqdm(combinations_substring, total=len(combinations_substring), desc="maximum substring", position=0))) 
    
        combinations_sublist = list(zip(generation_texts_word_tokenized, source_docs_word_tokenized))

        with Pool(num_workers) as pool:
            longest_sublist_words = list(pool.starmap(process_single_sublist_pair, tqdm(combinations_sublist,total=len(combinations_sublist), desc="maximum sublist", position=0)))
    else: # Single thread
        longest_substring_chars = [[longest_substring(g,source["text"]) for g in g_list] for g_list,source in tqdm(zip(generation_texts, source_docs))]
        longest_sublist_words = [[longest_sublist(g, source_doc_word_tokenized) for g in g_list_tokenized] 
                                 for g_list_tokenized, source_doc_word_tokenized in tqdm(zip(generation_texts_word_tokenized, source_docs_word_tokenized))]

    # Add computed metrics to each generation entry
    metrics = {
        "gen_text_length_char": generation_texts_length_chars,
        "gen_text_length_word": generation_texts_length_words,
        "longest_substring_char": longest_substring_chars,
        "longest_sublist_word": longest_sublist_words
    }
    
    for cur_generation, *metric_values in zip(generations, *metrics.values()):
        for field_name, value in zip(metrics.keys(), metric_values):
            cur_generation[field_name] = value

    save_to_jsonl(generations, args.output_file)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="N-gram coverage attack tool for analyzing text generation and detecting potential membership inference vulnerabilities by comparing generated texts against source documents using exact n-gram matching and longest common subsequence analysis.")
    parser.add_argument('--gen_data', type=str, required=True,
                        help="Path to JSONL file containing generated text data to analyze for potential membership inference vulnerabilities")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Directory where analysis results will be saved. Output filename will be derived from input filename with ngram size suffix")
    parser.add_argument("--min_ngram", type=int, default=3,
                        help="Minimum n-gram size for exact matching analysis (default: 3). Larger values require longer exact matches")
    parser.add_argument("--subset", type=int, default=int(1e6),
                        help="Maximum number of examples to process from the input data (default: 1,000,000). Use smaller values for testing")
    parser.add_argument("--generation_field", type=str, default="generation",
                        help="Name of the field in the input JSONL containing the generated text sequences to analyze")
    parser.add_argument('--parallel', action="store_true",
                        help="Enable parallel processing using multiple CPU cores for faster analysis of large datasets")

    args = parser.parse_args()
    main(args)