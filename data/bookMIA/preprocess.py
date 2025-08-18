"""
Preprocesses the BookMIA dataset.
Downloads from HuggingFace, cleans problematic text snippets with null characters,
and creates train/val/test splits.
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
CACHE_PATH = os.getenv("CACHE_PATH")

# Set up environment variables
os.environ["HF_HOME"] = CACHE_PATH
os.environ["HF_DATASETS_PATH"] = CACHE_PATH

from datasets import load_dataset
import random
import numpy as np
import argparse
import pandas as pd
from data.bookMIA.utils import clean_snippet
from sklearn.model_selection import train_test_split

def main(args):
    # Load dataset
    ds = load_dataset("swj0419/BookMIA")
    df = ds["train"].to_pandas()

    # Clean problematic snippets with null characters (e.g., Jane Eyre data corruption)
    problematic_rows = df[df['snippet'].str.contains('\x00', regex=False)].index
    for idx in problematic_rows:
        df.at[idx, 'snippet'] = clean_snippet(df.at[idx, 'snippet'])

    # Make train, val, and test split
    train_df, temp_df = train_test_split(df, test_size=args.val_split + args.test_split, random_state=args.seed)
    val_split_adjusted = args.val_split / (args.val_split + args.test_split)
    val_df, test_df = train_test_split(temp_df, test_size=1 - val_split_adjusted, random_state=args.seed)

    save_folder = os.path.join("data", "bookMIA", "split-random-overall")
    os.makedirs(save_folder, exist_ok=True)
    
    # Save train, val, and test splits
    train_df.to_json(os.path.join(save_folder, "test.jsonl"), lines=True, orient='records')
    test_df.to_json(os.path.join(save_folder, "train.jsonl"), lines=True, orient='records') # Make the "train set" the test set, since we want it to be the majority
    val_df.to_json(os.path.join(save_folder, "val.jsonl"), lines=True, orient='records')

    print("Data splits saved in folder:", save_folder)        
    print(train_df.loc[0])
    print(len(train_df), len(test_df), len(val_df))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make data splits")

    parser.add_argument("--val_split", type=float, default=0.05)
    parser.add_argument("--test_split", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    main(args)

    """
    # To run this file, use:
    python3 -m data.bookMIA.preprocess
    """
