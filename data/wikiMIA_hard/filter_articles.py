"""
Filters scraped Wikipedia articles for the WikiMIA-2024 Hard dataset.
Applies quality filters, removes bad content, and creates paired examples
where old summaries (2016) are labeled as members (1) and new summaries as non-members (0).
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
CACHE_PATH = os.getenv("CACHE_PATH")

# Set up environment variables
os.environ["HF_HOME"] = CACHE_PATH
os.environ["HF_DATASETS_PATH"] = CACHE_PATH

import random
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from code.utils import load_jsonl, save_to_jsonl
import re

def remove_thumb(summary):
    """Remove thumbnail markup artifacts from Wikipedia summaries"""
    while summary.startswith("thumb"):
        index = summary.find("\n")
        if index != -1:
            summary = summary[index+1:].strip()
    
    return summary

def has_period_followed_by_capital(text):
    """Check for malformed sentences (period directly followed by capital letter)"""
    pattern = r"\.[A-Z]"
    return bool(re.search(pattern, text))

# List of strings that indicate low-quality or problematic content
bad_strings = ["ISBN", "United States Patent Office", "Decreto", "CASE STUDY", "the source of this method's name", 
               "\mathbf", "Mairie de Montrea", "FCCdata.org/CBY", "Today National Correspondent"]

def main(args):  
    # Load scraped Wikipedia article data
    raw_data = load_jsonl("data/wikiMIA_hard/scraped/scraped.jsonl")
    
    filtered_data = []
    for r in raw_data:
        # Clean thumbnail artifacts
        r["old_summary"] = remove_thumb(r["old_summary"])
        r["new_summary"] = remove_thumb(r["new_summary"])

        # Clean empty parentheses and limit to 256 words
        r["old_summary"] = " ".join(re.sub(r'\(\s*[\W_]*\s*\)', '', r["old_summary"]).split()[:256])
        r["new_summary"] = " ".join(re.sub(r'\(\s*[\W_]*\s*\)', '', r["new_summary"]).split()[:256])
        old_sum = r["old_summary"]
        new_sum = r["new_summary"]

        # Filter out articles with problematic content
        bad = False
        for b in bad_strings:
            if b in old_sum or b in new_sum:
                bad = True
                break
            if has_period_followed_by_capital(old_sum) or has_period_followed_by_capital(new_sum):
                bad = True
                break
        if bad:
            continue

        # Filter by length constraints
        len_old = len(old_sum)
        len_new = len(new_sum)
        shorter = min(len_old, len_new)
        longer = max(len_old, len_new)

        # Require at least 25 words in each summary
        if len(r["old_summary"].split()) < 25 or len(r["new_summary"].split()) < 25:
            continue

        # Keep only articles with similar lengths (80%+) and significant differences (50%+)
        if shorter / longer >= 0.8 and r["percent_diff"] >= 0.5:
            filtered_data.append(r)

    # Deduplicate by title - keep only the first occurrence of each title
    seen_titles = set()
    deduplicated_data = []
    for item in filtered_data:
        if item["title"] not in seen_titles:
            seen_titles.add(item["title"])
            deduplicated_data.append(item)
    
    print(f"Filtered data: {len(filtered_data)} entries")
    print(f"After deduplication: {len(deduplicated_data)} entries")
    
    # Sample from deduplicated data
    filtered_data = random.sample(deduplicated_data, min(1000, len(deduplicated_data)))

    val_split_adjusted = args.val_split / (args.val_split + args.test_split)
    # Split nonmember and member data
    train, temp = train_test_split(filtered_data, test_size=args.val_split + args.test_split, random_state=args.seed)
    val, test = train_test_split(temp, test_size=1 - val_split_adjusted, random_state=args.seed)

    train_data = []
    for i, t in enumerate(train):
        train_data.append({"title": t["title"], "date": t["first_retrieved_date"], "input": t["old_summary"], "id": i, "diff": t["char_difference"], "label": 1})
        train_data.append({"title": t["title"], "date": t["last_edit_date"], "input": t["new_summary"], "id": i, "diff": t["char_difference"], "label": 0})
    
    val_data = []
    for i, t in enumerate(val):
        val_data.append({"title": t["title"], "date": t["first_retrieved_date"], "input": t["old_summary"], "id": i, "diff": t["char_difference"], "label": 1})
        val_data.append({"title": t["title"], "date": t["last_edit_date"], "input": t["new_summary"], "id": i, "diff": t["char_difference"], "label": 0})

    test_data = []
    for i, t in enumerate(test):
        test_data.append({"title": t["title"], "date": t["first_retrieved_date"], "input": t["old_summary"], "id": i, "diff": t["char_difference"], "label": 1})
        test_data.append({"title": t["title"], "date": t["last_edit_date"], "input": t["new_summary"], "id": i, "diff": t["char_difference"], "label": 0})


    # Recombine and shuffle
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)

    save_folder = os.path.join("data", "wikiMIA_hard", "split-random-overall")
    os.makedirs(save_folder, exist_ok=True)

    # Save train, val, and test splits
    save_to_jsonl(train_data, os.path.join(save_folder, "test.jsonl"))
    save_to_jsonl(test_data, os.path.join(save_folder, "train.jsonl")) # Make the "train set" the test set, since we want it to be the majority
    save_to_jsonl(val_data, os.path.join(save_folder, "val.jsonl"))

    print("Data splits saved in folder:", save_folder)        
    print(train_data[0])
    print(len(train_data), len(test_data), len(val_data))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter and split Wikipedia articles")

    parser.add_argument("--val_split", type=float, default=0.05)
    parser.add_argument("--test_split", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    main(args)

    """
    # To run this file, use:
    python3 -m data.wikiMIA_hard.filter_articles
    """
