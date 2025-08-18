"""
Preprocesses the Dolma v1.7 dataset with sophisticated length distribution matching.
Downloads member data from a 0.01% subsample and nonmember data from Paloma evaluation set.
Uses length-based bucketing to ensure similar text length distributions between member/nonmember sets.
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
from code.utils import save_to_jsonl
from datasets import load_dataset

def main(args):   
    dataset_member = load_dataset("emozilla/dolma-v1_7-3B")["train"] # DOLMA 1.7 0.01% subsample, so we don't need to process ourselves
    dataset_nonmember = load_dataset("allenai/paloma", "dolma-v1_5")["test"] # Paloma evaluation set
    random.seed(args.seed)

    dataset_member = dataset_member.add_column("label", [1] * len(dataset_member))
    dataset_member = dataset_member.rename_column("text", "snippet")    
    dataset_nonmember = dataset_nonmember.add_column("label", [0] * len(dataset_nonmember))
    dataset_nonmember = dataset_nonmember.rename_column("text", "snippet")

    # Precompute text lengths for efficiency
    dataset_member = dataset_member.map(lambda x: {"length": len(x["snippet"].split())})
    dataset_nonmember = dataset_nonmember.map(lambda x: {"length": len(x["snippet"].split())})
    lower_m, upper_m = np.percentile(dataset_member["length"], [args.min_percentile, 100-args.min_percentile])
    lower_nm, upper_nm = np.percentile(dataset_nonmember["length"], [args.min_percentile, 100-args.min_percentile])

    # Determine common range
    common_lower = max(lower_m, lower_nm)
    common_upper = min(upper_m, upper_nm)
    print(f"Filtering texts with lengths between {common_lower} and {common_upper}")
    dataset_member = dataset_member.filter(lambda x: common_lower <= x["length"] <= common_upper).shuffle(seed=args.seed)
    dataset_nonmember = dataset_nonmember.filter(lambda x: common_lower <= x["length"] <= common_upper).shuffle(seed=args.seed)

    # Group texts by length buckets to ensure similar length distributions
    # This prevents bias where member/nonmember sets have different length characteristics
    def create_length_buckets(dataset, num_buckets=10):
        lengths = dataset["length"]
        min_len, max_len = min(lengths), max(lengths)
        bucket_size = (max_len - min_len) / num_buckets
        
        # Assign each example to a bucket based on word count
        # E.g., if range is 10-100 words with 10 buckets: bucket 0 = 10-19 words, bucket 1 = 20-29 words, etc.
        return dataset.map(lambda x: {"length_bucket": int((x["length"] - min_len) // bucket_size)})

    # Create buckets in both datasets and find common buckets
    dataset_member = create_length_buckets(dataset_member)
    dataset_nonmember = create_length_buckets(dataset_nonmember)
    member_buckets = set(dataset_member["length_bucket"])
    nonmember_buckets = set(dataset_nonmember["length_bucket"])
    common_buckets = member_buckets.intersection(nonmember_buckets)

    # Count examples in each bucket across both datasets
    # This determines how many samples to take from each bucket proportionally
    bucket_counts = {}
    for bucket in common_buckets:
        member_count = sum(1 for b in dataset_member["length_bucket"] if b == bucket)
        nonmember_count = sum(1 for b in dataset_nonmember["length_bucket"] if b == bucket)
        bucket_counts[bucket] = member_count + nonmember_count

    # Calculate total examples in common buckets
    total_examples = sum(bucket_counts.values())

    # Sample proportionally to maintain length distribution (target: 500 examples each)
    target_samples = 500
    sampled_member = []
    sampled_nonmember = []

    for bucket in common_buckets:
        # Calculate proportional samples for this bucket
        proportion = bucket_counts[bucket] / total_examples
        bucket_samples = int(target_samples * proportion)
        
        # Ensure at least 1 sample from each bucket if possible
        bucket_samples = max(1, bucket_samples)
        
        # Get indices of examples in this bucket
        member_indices = [i for i, b in enumerate(dataset_member["length_bucket"]) if b == bucket]
        nonmember_indices = [i for i, b in enumerate(dataset_nonmember["length_bucket"]) if b == bucket]
        
        # Sample from each bucket (or take all if fewer than needed)
        bucket_member = random.sample(member_indices, min(bucket_samples, len(member_indices)))
        bucket_nonmember = random.sample(nonmember_indices, min(bucket_samples, len(nonmember_indices)))
        
        sampled_member.extend(bucket_member)
        sampled_nonmember.extend(bucket_nonmember)

    # Adjust if we don't have exactly target_samples
    if len(sampled_member) < target_samples:
        remaining_indices = [i for i in range(len(dataset_member)) if i not in sampled_member]
        additional = random.sample(remaining_indices, min(target_samples - len(sampled_member), len(remaining_indices)))
        sampled_member.extend(additional)
    elif len(sampled_member) > target_samples:
        sampled_member = sampled_member[:target_samples]

    if len(sampled_nonmember) < target_samples:
        remaining_indices = [i for i in range(len(dataset_nonmember)) if i not in sampled_nonmember]
        additional = random.sample(remaining_indices, min(target_samples - len(sampled_nonmember), len(remaining_indices)))
        sampled_nonmember.extend(additional)
    elif len(sampled_nonmember) > target_samples:
        sampled_nonmember = sampled_nonmember[:target_samples]

    data_member = dataset_member.select(sampled_member).to_list()
    data_nonmember = dataset_nonmember.select(sampled_nonmember).to_list()

    val_split_adjusted = args.val_split / (args.val_split + args.test_split)
    # Split nonmember and member data
    nonmember_train, nonmember_temp = train_test_split(data_nonmember, test_size=args.val_split + args.test_split, random_state=args.seed)
    nonmember_val, nonmember_test = train_test_split(nonmember_temp, test_size=1 - val_split_adjusted, random_state=args.seed)

    member_train, member_temp = train_test_split(data_member, test_size=args.val_split + args.test_split, random_state=args.seed)
    member_val, member_test = train_test_split(member_temp, test_size=1 - val_split_adjusted, random_state=args.seed)

    # Recombine and shuffle
    train_data = nonmember_train + member_train
    val_data = nonmember_val + member_val
    test_data = nonmember_test + member_test

    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)

    save_folder = os.path.join("data", "dolma_v17", "split-random-overall")
    os.makedirs(save_folder, exist_ok=True)

    # Save train, val, and test splits
    save_to_jsonl(train_data, os.path.join(save_folder, "test.jsonl"))
    save_to_jsonl(test_data, os.path.join(save_folder, "train.jsonl")) # Make the "train set" the test set, since we want it to be the majority
    save_to_jsonl(val_data, os.path.join(save_folder, "val.jsonl"))

    print("Data splits saved in folder:", save_folder)        
    print(train_data[0])
    print(len(train_data), len(test_data), len(val_data))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make data splits")

    parser.add_argument("--val_split", type=float, default=0.05)
    parser.add_argument("--test_split", type=float, default=0.05)
    parser.add_argument("--min_percentile", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    main(args)

    """
    # To run this file, use:
    python3 -m data.dolma_v17.preprocess
    """
