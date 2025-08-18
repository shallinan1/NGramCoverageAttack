"""
Preprocesses the WikiMIA-2024 Hard dataset.
Directly downloads the pre-split JSONL files from HuggingFace and saves train/val/test splits locally.
This dataset contains temporal Wikipedia articles with different versions based on date cutoffs.
"""
from dotenv import load_dotenv
load_dotenv()
import os
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

from huggingface_hub import hf_hub_download

def main():
    # Create output directory
    save_folder = os.path.join("data", "wikiMIA_hard", "split-random-overall")
    os.makedirs(save_folder, exist_ok=True)
    
    # Download files directly from HuggingFace
    files = ["train.jsonl", "val.jsonl", "test.jsonl"]
    
    for filename in files:
        print(f"Downloading {filename}...")
        file_path = hf_hub_download(
            repo_id="hallisky/wikiMIA-2024-hard",
            filename=filename,
            repo_type="dataset",
            local_dir=save_folder,
            local_dir_use_symlinks=False
        )
    
    print("Data splits downloaded to folder:", save_folder)

if __name__ == "__main__":
    main()

    """
    # To run this file, use:
    python3 -m data.wikiMIA_hard.preprocess
    """