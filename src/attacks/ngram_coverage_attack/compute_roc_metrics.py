"""
ROC Scoring and Analysis for N-Gram Coverage Attack

This module implements the final scoring component of the N-Gram Coverage Attack 
method for membership inference attacks against language models.

The module computes ROC curves and performance metrics from creativity indices
and coverage statistics produced by the previous pipeline stages. It evaluates
multiple aggregation strategies (min, max, median, mean) across different
metric types to identify the most effective membership inference signals.

Pipeline:
    1. Load creativity index results from JSONL file (output of compute_creativity_index.py)
    2. Apply various aggregation strategies to coverage and creativity metrics
    3. Compute ROC curves and AUC scores for each strategy
    4. Calculate TPR at specific FPR thresholds (0.1%, 0.5%, 1%, 5%)
    5. Save comprehensive scoring results to JSON format
    6. Optionally generate ROC curve visualizations

Outputs:
    JSON file containing:
    - ROC AUC scores for each aggregation strategy
    - TPR values at predefined FPR thresholds
    - Strategy rankings and performance comparisons
    
Supported Aggregation Strategies:
    - Coverage metrics: gen_length, ref_length, total_length
    - Creativity indices: Sums across n-gram ranges for all coverage types
    - Text length metrics: Character and word counts for generated text
    - Longest match metrics: Character-level substring and word-level subsequence lengths
    - Aggregation methods: Min, Max, Median, Mean for each metric type

Hardcoded Configuration:
    - FPR thresholds: 0.1%, 0.5%, 1%, 5% for TPR evaluation
    - Output directory structure: Derived from input path by replacing 'creativities' with 'scores'
    - File naming: Uses target model name extracted from input filename

Usage:
    python -m src.attacks.ngram_coverage_attack.compute_roc_metrics \
        --creativity_file PATH_TO_CREATIVITY_INDICES.jsonl
"""

# Standard library imports
import argparse
import json
import os
from typing import Dict, List, Any, Callable

# Third-party imports
import numpy as np
from dotenv import load_dotenv
from sklearn.metrics import roc_curve, auc

# Local imports
from src.utils.io_utils import load_jsonl
from src.experiments.utils import plot_roc_curve

# Load environment variables
load_dotenv()
CACHE_PATH = os.getenv("CACHE_PATH")

# Set up environment variables
os.environ["HF_HOME"] = CACHE_PATH
os.environ["HF_DATASETS_PATH"] = CACHE_PATH

FPR_THRESHOLDS = [0.001, 0.005, 0.01, 0.05]  # 0.1%, 0.5%, 1%, 5% FPR thresholds

# Define aggregation strategies for computing membership inference scores
strategies: Dict[str, Dict[str, Callable[[Dict[str, Any]], float]]] = {
    "Min_Coverage_Gen_Length": {"func": lambda x: np.min(x["coverages_gen_length"])},
    "Max_Coverage_Gen_Length": {"func": lambda x: np.max(x["coverages_gen_length"])},
    "Median_Coverage_Gen_Length": {"func": lambda x: np.median(x["coverages_gen_length"])},
    "Mean_Coverage_Gen_Length": {"func": lambda x: np.mean(x["coverages_gen_length"])},
    
    "Min_Coverage_Ref_Length": {"func": lambda x: np.min(x["coverages_ref_length"])},
    "Max_Coverage_Ref_Length": {"func": lambda x: np.max(x["coverages_ref_length"])},
    "Median_Coverage_Ref_Length": {"func": lambda x: np.median(x["coverages_ref_length"])},
    "Mean_Coverage_Ref_Length": {"func": lambda x: np.mean(x["coverages_ref_length"])},
    
    "Min_Coverage_Total_Length": {"func": lambda x: np.min(x["coverages_total_length"])},
    "Max_Coverage_Total_Length": {"func": lambda x: np.max(x["coverages_total_length"])},
    "Median_Coverage_Total_Length": {"func": lambda x: np.median(x["coverages_total_length"])},
    "Mean_Coverage_Total_Length": {"func": lambda x: np.mean(x["coverages_total_length"])},

    "Min_Creativity_Gen_Length": {"func": lambda x: np.min(x["creativities_gen_length"])},
    "Max_Creativity_Gen_Length": {"func": lambda x: np.max(x["creativities_gen_length"])},
    "Median_Creativity_Gen_Length": {"func": lambda x: np.median(x["creativities_gen_length"])},
    "Mean_Creativity_Gen_Length": {"func": lambda x: np.mean(x["creativities_gen_length"])},

    "Min_Creativity_Ref_Length": {"func": lambda x: np.min(x["creativities_ref_length"])},
    "Max_Creativity_Ref_Length": {"func": lambda x: np.max(x["creativities_ref_length"])},
    "Median_Creativity_Ref_Length": {"func": lambda x: np.median(x["creativities_ref_length"])},
    "Mean_Creativity_Ref_Length": {"func": lambda x: np.mean(x["creativities_ref_length"])},
    
    "Min_Creativity_Total_Length": {"func": lambda x: np.min(x["creativities_total_length"])},
    "Max_Creativity_Total_Length": {"func": lambda x: np.max(x["creativities_total_length"])},
    "Median_Creativity_Total_Length": {"func": lambda x: np.median(x["creativities_total_length"])},
    "Mean_Creativity_Total_Length": {"func": lambda x: np.mean(x["creativities_total_length"])},

    "Min_GenTextLengthChar": {"func": lambda x: np.min(x["gen_text_length_char"])},
    "Max_GenTextLengthChar": {"func": lambda x: np.max(x["gen_text_length_char"])},
    "Median_GenTextLengthChar": {"func": lambda x: np.median(x["gen_text_length_char"])},
    "Mean_GenTextLengthChar": {"func": lambda x: np.mean(x["gen_text_length_char"])},
    
    "Min_GenTextLengthWord": {"func": lambda x: np.min(x["gen_text_length_word"])},
    "Max_GenTextLengthWord": {"func": lambda x: np.max(x["gen_text_length_word"])},
    "Median_GenTextLengthWord": {"func": lambda x: np.median(x["gen_text_length_word"])},
    "Mean_GenTextLengthWord": {"func": lambda x: np.mean(x["gen_text_length_word"])},
    
    "Min_LongestSubstringChar": {"func": lambda x: np.min(x["longest_substring_char"])},
    "Max_LongestSubstringChar": {"func": lambda x: np.max(x["longest_substring_char"])},
    "Median_LongestSubstringChar": {"func": lambda x: np.median(x["longest_substring_char"])},
    "Mean_LongestSubstringChar": {"func": lambda x: np.mean(x["longest_substring_char"])},
    
    "Min_LongestSublistWord": {"func": lambda x: np.min(x["longest_sublist_word"])},
    "Max_LongestSublistWord": {"func": lambda x: np.max(x["longest_sublist_word"])},
    "Median_LongestSublistWord": {"func": lambda x: np.median(x["longest_sublist_word"])},
    "Mean_LongestSublistWord": {"func": lambda x: np.mean(x["longest_sublist_word"])},
}

def main(args: argparse.Namespace) -> None:
    """
    Execute ROC analysis on creativity indices with multiple aggregation strategies.
    
    Args:
        args: Command-line arguments with creativity_file path to creativity indices
    
    Side Effects:
        Creates output directory and writes scores.json with ROC metrics
    """
    target_model_name = args.creativity_file.split(os.sep)[-1][:-6]

    base_dir = os.path.dirname(args.creativity_file).replace("creativities", "scores")  # Up one level from 'probs'
    output_dir = os.path.join(base_dir, target_model_name)
    plot_dir = os.path.join(output_dir, 'plots')
    print(f"Saving to {output_dir}")

    results = load_jsonl(args.creativity_file)  # Load creativity indices from file
    gen_labels = [g["label"] for g in results]  # Extract membership labels (1=member, 0=non-member)

    all_scores = {}
    for strategy in strategies:
        strategy_values = strategies[strategy]

        scores = [strategy_values["func"](r) for r in results]

        fpr, tpr, thresholds = roc_curve(gen_labels, scores)
        roc_auc = auc(fpr, tpr)
        all_scores[strategy] = {}
        all_scores[strategy]["roc_auc"] = roc_auc
        
        # Find the TPR at different FPR thresholds
        for target_fpr in FPR_THRESHOLDS:
            valid_indices = np.where(fpr <= target_fpr)[0]
            if len(valid_indices) > 0:
                idx = valid_indices[-1]  # Get the last (highest) FPR that's <= target
                tpr_at_fpr = tpr[idx]
            else:
                tpr_at_fpr = 0.0  # If no FPR <= target, set TPR to 0
            all_scores[strategy][f"tpr_at_{target_fpr*100:.1f}_fpr"] = tpr_at_fpr

        # Uncomment if you want to generate roc_curves for different metrics
        # plot_title=f"{dataset} ({split}): {strategy}, {target_model_name}"
        # plot_roc_curve(fpr, tpr, roc_auc, plot_title, os.path.join(plot_dir, f"{strategy}.png"))

    output_file_path = os.path.join(output_dir, f"scores.json")
    os.makedirs(plot_dir, exist_ok=True)
    with open(output_file_path, 'w') as f:
        json.dump(all_scores, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Compute ROC curves and performance metrics for membership inference attack evaluation "
                   "using multiple aggregation strategies on creativity indices and coverage statistics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--creativity_file', 
        type=str, 
        required=True,
        help="Path to JSONL file containing creativity indices and coverage statistics from compute_creativity_index.py. "
             "Each line should contain membership labels and computed metrics for ROC analysis."
    )
    main(parser.parse_args())