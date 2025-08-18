"""
Preprocesses the TULU v1 instruction-following dataset with sophisticated length distribution matching.
Uses 2D binning (user_length x response_length) and proportional sampling to ensure balanced
length distributions between member (core TULU) and non-member (complementary) datasets.
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
from code.utils import save_to_jsonl, load_jsonl
from tqdm import tqdm
import matplotlib.pyplot as plt

def calculate_word_counts(sample):
    """Calculate word counts for user prompt and assistant response"""
    user_message = sample["messages"][0]["content"]
    response_message = sample["messages"][1]["content"]

    user_length, response_length = len(user_message.split()), len(response_message.split())

    return {
        "user_length": user_length, 
        "response_length": response_length, 
        "total_length": user_length + response_length
    }

def print_length_percentiles(sampled_member, sampled_nonmember):
    """
    Print the mean values for user length, response length, and total length 
    for both member and nonmember samples.
    
    Args:
        sampled_member (list of dict): List of dictionaries containing 'user_length', 'response_length', and 'total_length'.
        sampled_nonmember (list of dict): List of dictionaries containing 'user_length', 'response_length', and 'total_length'.
    """
    sampled_member_lengths = [(s["user_length"], s["response_length"], s["total_length"]) for s in sampled_member]
    sampled_nonmember_lengths = [(s["user_length"], s["response_length"], s["total_length"]) for s in sampled_nonmember]
    
    for member_mean, nonmember_mean in zip(np.mean(sampled_member_lengths, axis=0), np.mean(sampled_nonmember_lengths, axis=0)):
        print(f"Member: {member_mean:.2f} Nonmember: {nonmember_mean:.2f}")


def plot_length_histogram(data_member, data_nonmember, field, title):
    """
    Plot histogram of lengths for a given field (e.g., "user_length", "response_length", "total_length").
    The upper and lower ends are cut off at the 95th percentile.
    Histograms for Member and Nonmember are displayed in separate subplots.
    """
    # Extract lengths from both datasets
    member_lengths = [sample[field] for sample in data_member]
    nonmember_lengths = [sample[field] for sample in data_nonmember]
    
    # Combine lengths to calculate percentiles
    all_lengths = member_lengths + nonmember_lengths
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    
    # Plot Member histogram
    ax1.hist(member_lengths, bins=10, alpha=0.5, label="Member", color="blue",zorder=3)
    ax1.set_ylabel("Frequency")
    ax1.set_title(f"Member: {title}")
    ax1.legend()
    ax1.grid(alpha=0.2, zorder=-1)
    
    # Plot Nonmember histogram
    ax2.hist(nonmember_lengths, bins=10, alpha=0.5, label="Nonmember", color="orange",zorder=3)
    ax2.set_xlabel(field)
    ax2.set_ylabel("Frequency")
    ax2.set_title(f"Nonmember: {title}")
    ax2.legend()
    ax2.grid(alpha=0.3, zorder=-1)
    
    # Adjust layout and save figure
    plt.tight_layout()
    plt.savefig(f"{title}.png", bbox_inches="tight", dpi=200)
    
def main(args):   
    n_bins = 4

    random.seed(args.seed)
    np.random.seed(args.seed)

    data_member = load_jsonl("data/tulu_v1/processed/tulu_v1/tulu_v1_data.jsonl")
    data_nonmember = load_jsonl("data/tulu_v1/processed/tulu_v1/inverse_tulu_v1_data.jsonl")
    random.shuffle(data_member)
    random.shuffle(data_nonmember)

    data_member_classes, data_nonmember_classes = set(), set()
    for d in data_member:
        d["label"] = 1
        data_member_classes.add(d["dataset"])
    for d in data_nonmember:
        d["label"] = 0
        data_nonmember_classes.add(d["dataset"])

    for sample in tqdm(data_member):
        sample.update(calculate_word_counts(sample))
    for sample in tqdm(data_nonmember):
        sample.update(calculate_word_counts(sample))

    # Remove data that is too short or too long
    member_user_lengths = [sample["user_length"] for sample in data_member]
    nonmember_user_lengths = [sample["user_length"] for sample in data_nonmember]
    member_response_lengths = [sample["response_length"] for sample in data_member]
    nonmember_response_lengths = [sample["response_length"] for sample in data_nonmember]

    
    tail_amount = 5
    # Prune extreme length responses
    user_lower_cutoff = max(np.percentile(member_user_lengths, tail_amount), np.percentile(nonmember_user_lengths, tail_amount))
    user_upper_cutoff = min(np.percentile(member_user_lengths, 100-tail_amount), np.percentile(nonmember_user_lengths, 100-tail_amount))
    
    response_lower_cutoff = max(np.percentile(member_response_lengths, tail_amount), np.percentile(nonmember_response_lengths, tail_amount))
    response_upper_cutoff = min(np.percentile(member_response_lengths, 100-tail_amount),np.percentile(nonmember_response_lengths, 100-tail_amount))

    data_member_new, data_nonmember_new = [], []
    for d in data_member:
        if (
            user_lower_cutoff < d["user_length"] < user_upper_cutoff
            and response_lower_cutoff < d["response_length"] < response_upper_cutoff
        ):
            data_member_new.append(d)
    for d in data_nonmember:
        if (
            user_lower_cutoff < d["user_length"] < user_upper_cutoff
            and response_lower_cutoff < d["response_length"] < response_upper_cutoff
        ):
            data_nonmember_new.append(d)

    print("Old/new member lengths:", len(data_member), len(data_member_new))
    print("Old/new nonmember lengths:", len(data_nonmember), len(data_nonmember_new))
    data_member = data_member_new
    data_nonmember = data_nonmember_new

    # Plot lengths - before pruning
    plot_length_histogram(data_member, data_nonmember, "user_length", "Histogram of User Lengths - Original")
    plot_length_histogram(data_member, data_nonmember, "response_length", "Histogram of Response Lengths - Original")
    plot_length_histogram(data_member, data_nonmember, "total_length", "Histogram of Total Lengths - Original")
    print_length_percentiles(data_member, data_nonmember)

    def sample_uniformly_by_length(data_member, data_nonmember, n_bins=4):
        """
        Sample data using 2D binning to match length distributions.
        
        Creates a grid of bins based on user_length × response_length, then samples
        proportionally from each bin to ensure member/nonmember sets have similar
        length characteristics across multiple source datasets.
        """
        # Combine data to calculate percentiles for bin edges
        all_user_lengths = [sample["user_length"] for sample in data_member + data_nonmember]
        all_response_lengths = [sample["response_length"] for sample in data_member + data_nonmember]

        plt.figure()
        plt.hist(all_user_lengths,bins=100)
        plt.tight_layout()
        plt.savefig("all_user_lengths.png", dpi=200,bbox_inches="tight")
        plt.figure()
        plt.hist(all_response_lengths,bins=100)
        plt.tight_layout()
        plt.savefig("all_response_lengths.png", dpi=200,bbox_inches="tight")

        # Create bins for user_length and response_length
        user_bin_edges = np.percentile(all_user_lengths, np.linspace(0, 100, n_bins + 1))
        response_bin_edges = np.percentile(all_response_lengths, np.linspace(0, 100, n_bins + 1))
        
        user_bin_edges[-1] += 1 # Fix bug for longest sequence in assign_to_bins
        response_bin_edges[-1] += 1

        # Assign samples to bins
        def assign_to_bins(data, user_bin_edges, response_bin_edges):
            binned_data = [[[] for _ in range(n_bins)] for _ in range(n_bins)]
            for sample in data:
                user_bin = np.digitize(sample["user_length"], user_bin_edges) - 1
                response_bin = np.digitize(sample["response_length"], response_bin_edges) - 1
                assert user_bin >= 0 and user_bin < n_bins and response_bin >= 0 and response_bin < n_bins
                binned_data[user_bin][response_bin].append(sample)
            return binned_data
        
        binned_member = assign_to_bins(data_member, user_bin_edges, response_bin_edges)
        binned_nonmember = assign_to_bins(data_nonmember, user_bin_edges, response_bin_edges)
        
        bin_counts_dict_total = {i: {j: 0 for j in range(n_bins)} for i in range(n_bins)}
        for i in range(n_bins):
            for j in range(n_bins):
                bin_counts_dict_total[i][j] += len(binned_member[i][j]) + len(binned_nonmember[i][j])
        total_sum = sum(sum(inner.values()) for inner in bin_counts_dict_total.values())
        for i in bin_counts_dict_total:
            for j in bin_counts_dict_total[i]:
                bin_counts_dict_total[i][j] /= total_sum

        # Sample from each bin according to bin_counts_dict_total
        sampled_member, sampled_nonmember = [], []
        for i in range(n_bins):
            for j in range(n_bins):
                member_samples = binned_member[i][j]
                nonmember_samples = binned_nonmember[i][j]

                member_samples_by_dataset = {}
                nonmember_samples_by_dataset = {}
                for m in member_samples:
                    cur_dataset = m["dataset"]
                    if cur_dataset not in member_samples_by_dataset:
                        member_samples_by_dataset[cur_dataset] = []
                    member_samples_by_dataset[cur_dataset].append(m)

                for nm in nonmember_samples:
                    cur_dataset = nm["dataset"]
                    if cur_dataset not in nonmember_samples_by_dataset:
                        nonmember_samples_by_dataset[cur_dataset] = []
                    nonmember_samples_by_dataset[cur_dataset].append(nm)
                
                # Target 30 samples per dataset per bin (balanced across source datasets)
                samples_per_dataset = 30
                sample_amount_member = int(samples_per_dataset * bin_counts_dict_total[i][j] * len(nonmember_samples_by_dataset))
                sample_amount_nonmember = int(samples_per_dataset * bin_counts_dict_total[i][j] * len(member_samples_by_dataset))
                                
                for m in member_samples_by_dataset:
                    cur_data = member_samples_by_dataset[m]
                    if len(cur_data) < sample_amount_member:
                        print(m, len(cur_data))
                    sampled_member.extend(random.sample(cur_data, min(len(cur_data), sample_amount_member)))
                
                for nm in nonmember_samples_by_dataset:
                    cur_data = nonmember_samples_by_dataset[nm]
                    if len(cur_data) < sample_amount_nonmember:
                        print(nm, len(cur_data))
                    sampled_nonmember.extend(random.sample(cur_data, min(len(cur_data), sample_amount_nonmember)))
        
        return sampled_member, sampled_nonmember

    sampled_member, sampled_nonmember = sample_uniformly_by_length(data_member, data_nonmember, n_bins)
    
    # Summarize dataset splits
    plot_length_histogram(sampled_member, sampled_nonmember, "user_length", "Histogram of User Lengths - Sampled")
    plot_length_histogram(sampled_member, sampled_nonmember, "response_length", "Histogram of Response Lengths - Sampled")
    plot_length_histogram(sampled_member, sampled_nonmember, "total_length", "Histogram of Total Lengths - Sampled")
    print_length_percentiles(sampled_member, sampled_nonmember)
    dataset_counts = {"member": {}, "nonmember": {}}
    for sm, snm in zip(sampled_member, sampled_nonmember):
        if sm["dataset"] not in dataset_counts["member"]:
            dataset_counts["member"][sm["dataset"]] = 0
        dataset_counts["member"][sm["dataset"]] += 1

        if snm["dataset"] not in dataset_counts["nonmember"]:
            dataset_counts["nonmember"][snm["dataset"]] = 0
        dataset_counts["nonmember"][snm["dataset"]] += 1
    print(dataset_counts["member"], "\n", dataset_counts["nonmember"])
    print(len(sampled_member), len(sampled_nonmember))

    # Make the train test splits
    member_train, member_temp = train_test_split(sampled_member, test_size=args.val_split + args.test_split, random_state=args.seed)
    val_split_adjusted = args.val_split / (args.val_split + args.test_split)
    member_val, member_test = train_test_split(member_temp, test_size=1 - val_split_adjusted, random_state=args.seed)

    # Split nonmember and member data
    nonmember_train, nonmember_temp = train_test_split(sampled_nonmember, test_size=args.val_split + args.test_split, random_state=args.seed)
    val_split_adjusted = args.val_split / (args.val_split + args.test_split)
    nonmember_val, nonmember_test = train_test_split(nonmember_temp, test_size=1 - val_split_adjusted, random_state=args.seed)

    # Recombine and shuffle
    train_data = nonmember_train + member_train
    val_data = nonmember_val + member_val
    test_data = nonmember_test + member_test

    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)

    save_folder = os.path.join("data", "tulu_v1", "split-random-overall")
    os.makedirs(save_folder, exist_ok=True)

    # Save train, val, and test splits
    save_to_jsonl(train_data, os.path.join(save_folder, "test.jsonl"))
    save_to_jsonl(test_data, os.path.join(save_folder, "train.jsonl")) # Make the "train set" the test set, since we want it to be the majority
    save_to_jsonl(val_data, os.path.join(save_folder, "val.jsonl"))

    print("Data splits saved in folder:", save_folder)        
    print(train_data[0])
    print(len(train_data), len(test_data), len(val_data))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process TULU v1 dataset with length distribution matching")
    parser.add_argument("--val_split", type=float, default=0.05)
    parser.add_argument("--test_split", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    args = parser.parse_args()

    main(args)

    """
    # To run this file, use:
    python3 -m data.tulu_v1.preprocess
    """

