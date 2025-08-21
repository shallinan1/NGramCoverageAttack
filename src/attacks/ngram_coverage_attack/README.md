# N-Gram Coverage Attack

Implementation of the **N-Gram Coverage Attack** method for membership inference attacks against language models, as described in our paper.

## 📊 Method Overview

The N-Gram Coverage Attack detects training data membership by analyzing how language models reproduce exact n-gram sequences from source documents. The attack exploits the observation that models tend to memorize and reproduce training data more faithfully than unseen text.

### Attack Pipeline
1. **Text Generation** (`generate.py`): Generate multiple continuations from partial text prompts
2. **N-gram Coverage Analysis** (`compute_ngram_coverage.py`): Find exact n-gram overlaps and compute coverage statistics using sliding window approach
3. **Creativity Scoring** (`compute_creativity_index.py`): Calculate modified creativity indices (sum of coverages across n-gram ranges)
4. **Membership Classification** (`compute_roc_metrics.py`): Evaluate using ROC-AUC and TPR@FPR metrics

## 🚀 Complete Pipeline Example

```bash
# Step 1: Generate text continuations
python -m src.attacks.ngram_coverage_attack.generate \
    --model gpt-3.5-turbo-0125 \
    --task bookMIA \
    --data_split train \
    --num_sentences 3 \
    --num_sequences 20 \
    --max_tokens 512 \
    --temperature 1.0 \
    --openai

# Step 2: Compute n-gram coverage
python -m src.attacks.ngram_coverage_attack.compute_ngram_coverage \
    --gen_data outputs/ours/bookMIA/generations/train/[GENERATED_FILE].jsonl \
    --output_dir outputs/ours/bookMIA/coverages/train/ \
    --min_ngram 3 \
    --parallel

# Step 3: Calculate creativity indices
python -m src.attacks.ngram_coverage_attack.compute_creativity_index \
    --coverage_path outputs/ours/bookMIA/coverages/train/[COVERAGE_FILE]_3_onedoc.jsonl \
    --output_dir outputs/ours/bookMIA/creativities/train/ \
    --min_ngram 2 \
    --max_ngram 12

# Step 4: Evaluate ROC metrics
python -m src.attacks.ngram_coverage_attack.compute_roc_metrics \
    --creativity_file outputs/ours/bookMIA/creativities/train/[CREATIVITY_FILE]_CI2-12.jsonl
```

## 📝 Detailed Usage

### Step 1: Text Generation (`generate.py`)

Generates text continuations using either OpenAI API or vLLM models.

#### OpenAI Models
```bash
python -m src.attacks.ngram_coverage_attack.generate \
    --openai \
    --model gpt-3.5-turbo-0125 \
    --task bookMIA \
    --data_split train \
    --start_sentence 1 \
    --num_sentences 5 \
    --num_sequences 20 \
    --max_tokens 512 \
    --temperature 1.0 \
    --top_p 0.95 \
    --task_prompt_idx 0 1 2  # Multiple prompt templates
```

#### vLLM Models
```bash
python -m src.attacks.ngram_coverage_attack.generate \
    --model meta-llama/Llama-2-7b-hf \
    --task bookMIA \
    --data_split train \
    --num_sentences 3 \
    --num_sequences 10 \
    --max_tokens 512 \
    --temperature 0.8 \
    --hf_token YOUR_TOKEN  # For gated models
```

#### Word-based Prompting (Alternative to Sentence-based)
```bash
# Remove last 50 words
python -m src.attacks.ngram_coverage_attack.generate \
    --model gpt-3.5-turbo \
    --task bookMIA \
    --data_split train \
    --prompt_with_words_not_sent \
    --num_words_from_end 50 \
    --num_sequences 20

# Remove last 30% of words
python -m src.attacks.ngram_coverage_attack.generate \
    --model gpt-3.5-turbo \
    --task bookMIA \
    --data_split train \
    --prompt_with_words_not_sent \
    --num_proportion_from_end 0.3 \
    --num_sequences 20
```

**Key Parameters:**
- `--max_length_to_sequence_length`: Match generation length to remaining text
- `--task_prompt_idx`: Select specific prompt templates (can specify multiple)
- `--temperature 0`: Use greedy decoding (automatically sets num_sequences=1)
- `--remove_bad_first`: Clean malformed first sentences

### Step 2: N-Gram Coverage Analysis (`compute_ngram_coverage.py`)

Computes exact n-gram matches and coverage statistics using dynamic programming.

```bash
python -m src.attacks.ngram_coverage_attack.compute_ngram_coverage \
    --gen_data path/to/generations.jsonl \
    --output_dir outputs/coverages/ \
    --min_ngram 3 \
    --subset 1000  # Process first 1000 examples \
    --generation_field "generation" \
    --parallel  # Use multiprocessing (max 4 CPUs)
```

**Features:**
- Sliding window approach for n-gram matching
- Longest common substring (character-level) computation
- Longest common subsequence (word-level) computation
- Efficient parallel processing with configurable workers

**Output includes:**
- `matched_spans`: List of matched text spans with positions
- `coverage`: Percentage of text covered by matches
- `avg_span_len`: Average length of matched spans
- `longest_substring_char`: Character-level longest match
- `longest_sublist_word`: Word-level longest match

### Step 3: Creativity Index Computation (`compute_creativity_index.py`)

Calculates modified creativity indices by summing coverage across n-gram ranges.

```bash
python -m src.attacks.ngram_coverage_attack.compute_creativity_index \
    --coverage_path path/to/coverage_3_onedoc.jsonl \
    --output_dir outputs/creativities/ \
    --min_ngram 2 \
    --max_ngram 12
```

**Important Note:** Higher creativity values indicate MORE copying (not less), maintaining consistency with MIA conventions where higher scores suggest membership.

**Metrics Computed:**
- **Standard metrics**: Include all matching spans (with duplicates)
- **Unique metrics**: Deduplicate spans by text content (suffix `_unique`)
- **Coverage types**:
  - `gen_length`: Normalized by generated text length
  - `ref_length`: Normalized by reference text length
  - `total_length`: Harmonic mean of both lengths

### Step 4: ROC Evaluation (`compute_roc_metrics.py`)

Evaluates membership inference performance using multiple aggregation strategies.

```bash
python -m src.attacks.ngram_coverage_attack.compute_roc_metrics \
    --creativity_file path/to/creativity_CI2-12.jsonl
```

**Aggregation Strategies:**
- Min, Max, Median, Mean for each metric type
- Coverage metrics (gen_length, ref_length, total_length)
- Creativity indices (modified sums across n-gram ranges)
- Text length metrics (character and word counts)
- Longest match metrics (substring and subsequence)

**Output Metrics:**
- ROC-AUC scores for each strategy
- TPR at FPR thresholds: 0.1%, 0.5%, 1%, 5%
- Strategy rankings and comparisons

## 📁 Output Structure

### Directory Layout
```
outputs/ours/
├── {task}/
│   ├── generations/
│   │   └── {split}/
│   │       └── {model}_{params}_{timestamp}.jsonl
│   ├── coverages/
│   │   └── {split}/
│   │       └── {model}_{params}_{ngram}_onedoc.jsonl
│   ├── creativities/
│   │   └── {split}/
│   │       └── {model}_{params}_CI{min}-{max}.jsonl
│   └── scores/
│       └── {model}_{params}/
│           ├── scores.json
│           └── plots/
```

### File Naming Convention
Files include detailed metadata in filenames:
```
{model}_maxTok{512}_minTokStr{0}_numSeq{20}_topP{0.95}_temp{1.0}_numSent{3}_startSent{1}_numWordFromEnd{-1}-{-1}_maxLenSeqFalse_useSentTrue-rmvBadFalse_promptIdx{0-1-2}_len{1000}_{timestamp}.jsonl
```

## 🔧 Configuration

### Hardcoded Defaults
- **Tokenization**: NLTK casual tokenizer for word processing
- **Detokenization**: Moses detokenizer for English
- **OpenAI tokenizer**: "gpt-3.5-turbo" encoding for all OpenAI models
- **vLLM formatting**: "lightest" prompt key for minimal overhead
- **Parallel processing**: Max 4 CPUs for n-gram computation
- **FPR thresholds**: [0.001, 0.005, 0.01, 0.05] for evaluation

### Rate Limits (OpenAI)
Rate limits are automatically applied for OpenAI models based on `src/generation/rate_limits.py`. Please see `src/generation/README.md` for more details.

## 📊 Data Requirements

The attack expects data files in the following structure:
- Path format: `data/{task}/split-random-overall/{split}.jsonl`
- Each JSONL line should contain a text field (default: "snippet", configurable via `--key_name`)
- Special handling for `tulu_v1`: Processes conversational format with user/assistant turns

## ⚠️ Important Notes

1. **Coverage Direction**: Higher coverage/creativity values indicate MORE copying (membership likelihood)
2. **File Formats**: All intermediate outputs use JSONL format for streaming processing


## 🐛 Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce `--parallel` workers or `--subset` size
2. **API Rate Limits**: Adjust rate limits in `requests_limits_dict`
3. **Tokenization Initialization Errors**: Ensure NLTK punkt data is downloaded


## Files Included
* `generate.py`: Generates text continuations from partial prompts using OpenAI API or vLLM models
* `compute_ngram_coverage.py`: Finds exact n-gram matches and computes coverage statistics
* `compute_creativity_index.py`: Calculates creativity indices from coverage scores across n-gram ranges
* `compute_roc_metrics.py`: Evaluates membership inference performance with ROC-AUC and TPR metrics
* `utils.py`: Text processing and similarity computation utilities
* `README.md`: Documentation and usage instructions
