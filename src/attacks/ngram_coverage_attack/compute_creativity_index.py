"""
Creativity Index Computation for N-Gram Coverage Attack

This module implements the creativity index computation component of the N-Gram Coverage 
Attack method for membership inference attacks against language models.

The module processes coverage analysis results by computing creativity indices across
multiple n-gram ranges. Note: This implementation computes creativity as the sum of coverage
values (not 1-coverage for all coverages), meaning HIGHER values indicate MORE copying from source text.
This consistent directionality is intentional: higher coverage values indicate a higher likelihood of membership, 
which is the standard convention in membership inference attacks (MIA).

Pipeline:
    1. Load coverage analysis results from JSONL file (output of compute_ngram_coverage.py)
    2. Compute additional coverage metrics in two variants:
       - Standard: All matching spans counted (including overlapping/duplicate spans)
       - Unique: Each unique span text counted only once per generation (deduplicated by text content)
    3. Calculate modified creativity indices by summing coverage across n-gram ranges for both variants
    4. Save enriched results with both standard and unique-filtered metrics to JSONL

Outputs:
    JSONL file containing:
    - Original coverage data from previous pipeline stage
    - Additional coverage metrics (gen_length, ref_length, total_length variants)
    - Creativity indices across configurable n-gram ranges
    - All metrics provided in two variants:
      * Standard fields: "coverages_*", "creativities_*" (all matching spans)
      * Unique fields: "coverages_*_unique", "creativities_*_unique" (deduplicated spans)

Hardcoded Configuration:
    - Default n-gram bounds: LOW_CI_BOUND=2, HIGH_CI_BOUND=12 for creativity index calculation
    - Tokenization: NLTK casual tokenizer for word-level processing
    - Min-gram filtering: Uses min_gram=1 for comprehensive coverage in additional metrics
    - Output naming: Appends "_CI{min_ngram}-{max_ngram}" suffix to input filename

Usage:
    python -m src.attacks.ngram_coverage_attack.compute_creativity_index \
        --coverage_path PATH_TO_COVERAGE.jsonl \
        --output_dir OUTPUT_DIRECTORY \
        --min_ngram 1 \
        --max_ngram 12
"""

from unidecode import unidecode
from nltk.tokenize.casual import casual_tokenize
import os
from tqdm import tqdm
import argparse
from typing import List, Dict, Any
from src.utils.io_utils import load_jsonl, save_to_jsonl

# Define constants for the argparser
LOW_CI_BOUND: int = 2
HIGH_CI_BOUND: int = 12

# Define global tokenization function
tokenize_func = lambda x: nltk.casual_tokenize(x)

def get_ngram_coverage(
    text: str, 
    spans: List[Dict[str, Any]], 
    min_gram: int, 
    ref_length: int, 
    unique_coverages: bool = False
) -> Dict[str, float]:
    """
    Compute n-gram coverage statistics for a given text and its matched spans.
    
    Args:
        text: The generated text to analyze
        spans: List of matched span dictionaries with start_index, end_index, and span_text
        min_gram: Minimum n-gram length to consider for coverage calculation
        ref_length: Reference text length for normalization
        unique_coverages: If True, each unique span text is counted only once (deduplication)
        
    Returns:
        Dictionary containing coverage percentages:
        - coverages_gen_length: Coverage relative to generated text length
        - coverages_ref_length: Coverage relative to reference text length  
        - coverages_total_length: Harmonic mean of generated and reference lengths
    """
    tokens = casual_tokenize(unidecode(text))
    flags = [False for _ in tokens]
    seen = set()

    for span in spans:
        span_text = span['span_text']
        span_length = span['end_index'] - span['start_index']
        if span_length >= min_gram:
            if not unique_coverages or span_text not in seen:
                flags[span['start_index']: span['end_index']] = [True] * span_length
                
        seen.add(span_text)

    if len([f for f in flags if f]) == 0:
        coverage_gen, coverage_ref, coverage_total = 0, 0, 0
    else: 
        ref_adjusted_length = max(len(flags), ref_length)
        coverage_gen = len([f for f in flags if f]) / len(flags)
        coverage_ref = len([f for f in flags if f]) / ref_adjusted_length
        coverage_total = (2 * len([f for f in flags if f])) / (len(flags) + ref_adjusted_length)
    return {
        "coverages_gen_length": coverage_gen,
        "coverages_ref_length": coverage_ref, 
        "coverages_total_length": coverage_total
        }

def compute_modified_creativity_index(
    outputs: List[Dict[str, Any]], 
    min_ngram: int, 
    max_ngram: int, 
    ref_length: int, 
    unique_coverages: bool = False
) -> List[Dict[str, float]]:
    """
    The creativity index is calculated by summing n-gram coverage scores across
    a range from min_ngram to max_ngram. Higher coverage indicates more copying
    from source text, while lower coverage suggests more creative generation.

    This function computes a modified creativity index statistic by summing coverage across n-gram ranges.
    This is to ensure that higher values correspond to more copying from a source text (needed to maintain consistency with other attacks).

    Args:
        outputs: List of generation outputs with 'text' and 'matched_spans' fields
        min_ngram: Starting n-gram size for creativity index calculation
        max_ngram: Ending n-gram size for creativity index calculation
        ref_length: Reference text length for coverage normalization
        unique_coverages: If True, only count unique spans once
        
    Returns:
        List of dictionaries containing creativity indices:
        - creativities_gen_length: Sum of coverage relative to generated text lengths (higher = more copying)
        - creativities_ref_length: Sum of coverage relative to reference text lengths (higher = more copying)
        - creativities_total_length: Sum of harmonic mean coverages (higher = more copying)
    """
    total_coverages = []
    ngram_list = list(range(min_ngram, max_ngram + 1))
    for output in outputs:
        coverages = []
        for cur_ngram in ngram_list:
            coverage_dict = get_ngram_coverage(output['text'], output['matched_spans'], cur_ngram, ref_length, unique_coverages)
            coverages.append(coverage_dict)

        total_coverages.append({key.replace("coverages", "creativities"): sum(d[key] for d in coverages) for key in coverages[0]})

    return total_coverages

def main(args: argparse.Namespace) -> None:
    """
    Main function to compute creativity indices from n-gram coverage analysis.
    
    Loads coverage analysis results, computes additional coverage metrics with
    unique span filtering, calculates modified creativity indices across n-gram ranges,
    and saves enriched results to JSONL format.
    
    Args:
        args: Command-line arguments containing coverage_path, output_dir,
              min_ngram, and max_ngram parameters
    """
    data = load_jsonl(args.coverage_path)
    os.makedirs(args.output_dir, exist_ok=True)

    # Compute the coverages
    for cur_data in tqdm(data, desc="Adding more coverages"):
        length_ref = len(tokenize_func(unidecode(cur_data["snippet_no_prompt"])))
        cur_coverages = []
        cur_coverages_unique = []
        for coverage in cur_data["coverage"]:
            cur_coverages.append(
                get_ngram_coverage(coverage["text"], 
                                   coverage["matched_spans"], 
                                   min_gram=1, 
                                   ref_length=length_ref,
                                   unique_coverages=False))
            cur_coverages_unique.append(
                get_ngram_coverage(coverage["text"], 
                                   coverage["matched_spans"], 
                                   min_gram=1, 
                                   ref_length=length_ref,
                                   unique_coverages=True))
        for coverage_key in ["coverages_gen_length", "coverages_ref_length", "coverages_total_length"]:
            cur_data[coverage_key] = [c[coverage_key] for c in cur_coverages]
            cur_data[coverage_key + "_unique"] = [c[coverage_key] for c in cur_coverages_unique]
    
    # Compute the modified creativity indices
    for cur_data in tqdm(data, desc = "Iterating through original data"):
        length_ref = len(tokenize_func(unidecode(cur_data["snippet_no_prompt"])))
        cur_creativity = compute_modified_creativity_index(cur_data["coverage"], args.min_ngram, args.max_ngram, length_ref, unique_coverages=False)
        cur_creativity_unique = compute_modified_creativity_index(cur_data["coverage"], args.min_ngram, args.max_ngram, length_ref, unique_coverages=True)
        
        for creativity_key in ["creativities_gen_length", "creativities_ref_length", "creativities_total_length"]:
            cur_data[creativity_key] = [c[creativity_key] for c in cur_creativity]
            cur_data[creativity_key + "_unique"] = [c[creativity_key] for c in cur_creativity_unique]

    # Save the outputs
    output_file = os.path.join(args.output_dir, os.path.basename(args.coverage_path).replace('.jsonl', f"_CI{args.min_ngram}-{args.max_ngram}.jsonl"))
    save_to_jsonl(data, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute modified creativity indices from n-gram coverage analysis results for "
                   "membership inference attack evaluation."
    )
    parser.add_argument(
        "--coverage_path", 
        type=str, 
        required=True,
        help="Path to the input JSONL file containing coverage analysis results from compute_ngram_coverage.py"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="experiments/ours/outputs/bookMIA/cis/",
        help="Directory where creativity index results will be saved (default: %(default)s)"
    )
    parser.add_argument(
        "--min_ngram", 
        type=int, 
        default=LOW_CI_BOUND,
        help="Minimum n-gram size for creativity index calculation (default: %(default)s)"
    )
    parser.add_argument(
        "--max_ngram", 
        type=int, 
        default=HIGH_CI_BOUND,
        help="Maximum n-gram size for creativity index calculation (default: %(default)s)"
    )
    
    args = parser.parse_args()
    main(args)
