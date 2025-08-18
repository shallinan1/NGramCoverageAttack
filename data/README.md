# Dataset Preprocessing

This directory contains preprocessing scripts to reproduce the datasets for experiments in the paper. Each dataset is processed and saved to `split-random-overall/` subdirectories with train/val/test splits.

## Dataset Preprocessing Scripts

### WikiMIA
This is the original WikiMIA benchmark dataset ([paper](https://arxiv.org/abs/2310.16789)), specifically the length-64 set. 

To download the dataset, run:
```bash
python3 -m data.wikiMIA.preprocess
```

---

### WikiMIA_2024 Hard
The WikiMIA_2024 Hard dataset is introduced in our paper, and consists of Wikipedia articles with different versions based on date cutoffs. Requires `HF_TOKEN` to be set in your `.env` file.

To download the dataset, run:
```bash
python3 -m data.wikiMIA_hard.preprocess
```

#### Recreating the Dataset
To instead create the dataset from scratch (scrape articles and filter data), run:
```bash
python3 -m data.wikiMIA_hard.scrape_articles
python3 -m data.wikiMIA_hard.filter_articles
```

---

### WikiMIA-24
This is the WikiMIA-24 benchmark dataset ([paper](https://arxiv.org/abs/2408.08661)), specifically the length-64 set.

To download the dataset, run:
```bash
python3 -m data.wikiMIA_update.preprocess
```

---

### BookMIA
This is a book membership inference dataset ([paper](https://arxiv.org/abs/2310.16789)).

To download the dataset, run:
```bash
python3 -m data.bookMIA.preprocess
```

---

### TULU v1
This is the TULU v1 instruction-following dataset.

#### Recreating the Dataset
These are the steps to download, reformat, and generate the processed dataset,

##### Prerequisites

This part requires a HuggingFace token for accessing the LIMA dataset (part of TULU). Request access at https://huggingface.co/datasets/GAIR/lima and ensure your `HF_TOKEN` is set in your `.env` file.

##### Creation

Now, run:
```bash
# Download raw datasets (requires HF_TOKEN)
./data/tulu_v1/download_data.sh

# Reformat datasets into standardized format
python3 data/tulu_v1/reformat_datasets.py --raw_data_dir data/tulu_v1/raw_train/ --output_dir data/tulu_v1/processed/ --dataset tulu_v1

# Create final train/val/test splits
python3 -m data.tulu_v1.preprocess
```

---

### Dolma (v1.7)
This is the Dolma v1.7 pre-training corpus ([paper](https://arxiv.org/abs/2402.00159)).

To download the dataset, run:
```bash
python3 -m data.dolma_v17.preprocess
```

---

### Pile (MIMIR)
This is the Pile dataset from MIMIR ([paper](https://arxiv.org/abs/2402.07841)).

To download the dataset, run:
```bash
python3 -m data.pile_external.preprocess
```

## Output Format

All datasets are processed into a standardized format:

**Directory Structure:**
```
data/[dataset_name]/split-random-overall/
├── train.jsonl
├── test.jsonl  
└── val.jsonl
```

**JSONL Schema:**
Each line contains a JSON object with fields like:
```json
{
  "label": 0,           // 0 = non-member, 1 = member
  "snippet": "text...", // Text content (varies by dataset)
  "dataset": "name"     // Source dataset identifier
}
```

