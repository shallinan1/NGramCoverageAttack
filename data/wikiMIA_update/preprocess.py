"""
Preprocesses the WikiMIA-24 dataset to match the size and format of the original WikiMIA dataset.
Downloads from HuggingFace, samples equal numbers from each label (271 each for 542 total),
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
from sklearn.model_selection import train_test_split

def main(args):
    # Load dataset
    ds = load_dataset("wjfu99/WikiMIA-24")

    df = ds["WikiMIA_length64"].to_pandas()
    
    # Match original WikiMIA dataset size: 542 total examples (271 per label)
    sample_size = 542 // 2
    
    # Sample equal number of examples from each label
    label_0 = df[df['label'] == 0].sample(n=sample_size, random_state=args.seed)
    label_1 = df[df['label'] == 1].sample(n=sample_size, random_state=args.seed)
    df = pd.concat([label_0, label_1])
    
    # Make train, val, and test split
    train_df, temp_df = train_test_split(df, test_size=args.val_split + args.test_split, random_state=args.seed)
    val_split_adjusted = args.val_split / (args.val_split + args.test_split)
    val_df, test_df = train_test_split(temp_df, test_size=1 - val_split_adjusted, random_state=args.seed)

    save_folder = os.path.join("data", "wikiMIA_update", "split-random-overall")
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
    python3 -m data.wikiMIA_update.preprocess
    """
