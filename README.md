# The Surprising Effectiveness of Membership Inference with Simple N-Gram Coverage

This is the repository for the CoLM 2025 paper ["The Surprising Effectiveness of Membership Inference with Simple N-Gram Coverage"](https://arxiv.org/abs/2508.09603).

<p align="center">
<img src="fig1.png" alt="drawing" width="99%"/>
</p>

## 🚀 Quick Start

Our N-Gram Coverage Attack is a simple yet effective method for membership inference. See the complete pipeline:

```bash
# 1. Generate text continuations
python -m src.attacks.ngram_coverage_attack.generate \
    --model gpt-3.5-turbo-0125 --task bookMIA --data_split train \
    --num_sentences 3 --num_sequences 20 --max_tokens 512 --openai

# 2. Compute n-gram coverage
python -m src.attacks.ngram_coverage_attack.compute_ngram_coverage \
    --gen_data outputs/ours/bookMIA/generations/train/[FILE].jsonl \
    --output_dir outputs/ours/bookMIA/coverages/train/ --min_ngram 3 --parallel

# 3. Calculate creativity indices
python -m src.attacks.ngram_coverage_attack.compute_creativity_index \
    --coverage_path outputs/ours/bookMIA/coverages/train/[FILE]_3_onedoc.jsonl \
    --output_dir outputs/ours/bookMIA/creativities/train/

# 4. Evaluate performance
python -m src.attacks.ngram_coverage_attack.compute_roc_metrics \
    --creativity_file outputs/ours/bookMIA/creativities/train/[FILE]_CI2-12.jsonl
```

For detailed usage and parameters, see [`src/attacks/ngram_coverage_attack/README.md`](src/attacks/ngram_coverage_attack/README.md).

## 🔧 Setup

Create conda environment:
```bash
conda create -n mia python=3.10
conda activate mia
pip install -r requirements.txt
```

Create a `.env` file in the root directory with the following variables:
```
OPENAI_API_KEY=your_openai_api_key_here
CACHE_PATH=/path/to/your/cache
HF_TOKEN=your_huggingface_token_here
```

## 📊 Repository Structure

```
├── src/
│   ├── attacks/
│   │   └── ngram_coverage_attack/    # Main attack implementation
│   ├── generation/                   # Text generation utilities
│   └── utils/                        # Helper functions
├── data/                             # Dataset files
├── outputs/                          # Generated results
└── requirements.txt
```

## 📚 Released Datasets

**WikiMIA 2024 Hard**: Available at https://huggingface.co/datasets/hallisky/wikiMIA-2024-hard

WikiMIA 2024 Hard is the challenging benchmark we introduce for membership inference attacks, designed to test attack methods on difficult examples.

## 📄 Citation

If you find this work useful, please cite:
```bibtex
@inproceedings{hallinan2025surprising,
  title={The Surprising Effectiveness of Membership Inference with Simple N-Gram Coverage},
  author={Hallinan, Skyler and Jung, Jaehun and Sclar, Melanie and Lu, Ximing and Ravichander, Abhilasha and Ramnath, Sahana and Choi, Yejin and Karimireddy, Sai Praneeth and Mireshghallah, Niloofar and Ren, Xiang},
  booktitle={Conference on Language Modeling (CoLM)},
  year={2025}
}
```

## 📧 Contact

For questions/issues about this repository or the paper, please email skyler.r.hallinan@gmail.com or raise an issue on this repository.