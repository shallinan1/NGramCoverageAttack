"""
Text Generation Module for N-Gram Coverage Attack

This module implements the generation component of the N-Gram Coverage Attack method
for membership inference attacks, as described in our paper.

The module generates text continuations using either OpenAI API or vLLM models for
evaluating membership inference vulnerabilities. It supports two generation pipelines:

1. OpenAI API: Async parallel generation with rate limiting for OpenAI models
2. vLLM: Local inference for open-source models with advanced sampling control

Pipeline:
    1. Load and validate input data from JSONL files
    2. Extract prompts from text snippets (sentence-based or word-based splitting)
    3. Format prompts using task-specific templates
    4. Generate text continuations using specified model
    5. Save results with comprehensive metadata for downstream analysis

Outputs:
    JSONL file containing original snippets, prompts, generations, and metadata
    saved to outputs/ours/{task}/generations/{data_split}/

Hardcoded Configuration:
    - OpenAI tokenizer: Uses "gpt-3.5-turbo" encoding for all OpenAI models
    - vLLM prompt formatting: Uses "lightest" prompt_key for minimal formatting overhead
    - Data paths: Expects data in "data/{task}/split-random-overall/{split}.jsonl" format
    - Output paths: Saves to "outputs/ours/{task}/generations/{split}/" directory

Usage:
    python -m src.attacks.ngram_coverage_attack.generate \
        --model MODEL_NAME \
        --task TASK_NAME \
        --data_split SPLIT \
        --start_sentence 1 \
        --num_sentences 3 \
        --num_sequences 20 \
        --max_tokens 512 \
        --task_prompt_idx 0 \
        [additional options]
"""

import argparse
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
CACHE_PATH = os.getenv("CACHE_PATH")

# Set up environment variables
os.environ["HF_HOME"] = CACHE_PATH
os.environ["HF_DATASETS_PATH"] = CACHE_PATH

import pandas as pd
from src.generation.vllm_generate import ModelGenerator 
from src.generation.prompt_formatting import task_prompts_dict_book, make_prompts
import random
from datetime import datetime
from src.attacks.ngram_coverage_attack.utils import extract_chunk_sentence
import asyncio
from src.generation.openai_parallel_generate import openai_parallel_generate, requests_limits_dict, requests_url_dict
from src.utils import remove_first_sentence_if_needed
from src.experiments.utils import zigzag_append, chunk_list, remove_last_n_words, bool_to_first_upper
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_openai(prompts, chunk_size, args, model_str, token_lengths):
    """
    Generate text using OpenAI API with async parallel processing and rate limiting.
    
    This function handles the OpenAI-specific generation pipeline including:
    - Building API requests with proper formatting for different model types
    - Managing rate limits and parallel execution
    - Handling request ID tracking and recovery
    - Extracting generated text from API responses
    
    Args:
        prompts (List[str]): List of formatted prompts ready for generation
        chunk_size (int): Number of prompt templates used (for organizing results)
        args: Command line arguments containing generation parameters
        model_str (str): Model name for API requests
        token_lengths (List[int]): Token lengths for each text snippet, used when
                                 args.max_length_to_sequence_length is True
    
    Returns:
        Tuple[List[str], List[List[str]]]: 
            - prompts: Original prompts passed to the API
            - all_text_outputs: Generated text sequences, organized as list of lists
                              where each inner list contains multiple generations per prompt
    
    Note:
        - Automatically detects model type (instruct/davinci vs chat) for proper formatting
        - Uses async parallel generation with configurable rate limits
        - Includes recovery mechanism for requests with missing IDs
    """
    requests = []
    for i, prompt in enumerate(prompts):
        if args.max_length_to_sequence_length:
            cur_max_tokens = token_lengths[i // chunk_size]
        else:
            assert args.max_tokens >= 1
            cur_max_tokens = args.max_tokens
        request_id = i
        cur_request = {
            "model": args.model,
            "max_tokens": cur_max_tokens,
            "temperature": args.temperature,
            "seed": args.seed,
            "top_p": args.top_p,
            "n": args.num_sequences, # Hardcode this
            "metadata": {"request_id": request_id},
        }
        if "instruct" in args.model or "davinci" in args.model:
            cur_request = cur_request | {
                "prompt": prompt,
                # "logprobs": 10 # Uncomment to get logprobs (not needed for this attack)
            }
        else:
            cur_request = cur_request | {
                "messages": [{"role": "user", "content": prompt}],
                # "logprobs": True, "top_logprobs": 10 # Uncomment to get logprobs (not needed for this attack)
            }
        requests.append(cur_request)
    if "instruct" not in args.model:
        logger.info(f"Example prompt:\n{requests[0]['messages'][0]['content']}")
    else:
        logger.info(f"Example prompt:\n{requests[0]['prompt']}")

    max_requests_per_minute = requests_limits_dict[args.model]["max_requests_per_minute"]
    max_tokens_per_minute = requests_limits_dict[args.model]["max_tokens_per_minute"]
    request_url = requests_url_dict[args.model]
    logger.info(f"Using OpenAI rate limits - Max requests/min: {max_requests_per_minute}, Max tokens/min: {max_tokens_per_minute}")

    full_generations = asyncio.run(openai_parallel_generate(
            requests,
            max_requests_per_minute=max_requests_per_minute, 
            max_tokens_per_minute=max_tokens_per_minute,
            request_url=request_url
            ))
        
    indexed_results = {}
    unknown_id_generations = [] # Special case where the request_id is not returned
    for result in full_generations:
        try:
            request_id = result[2]["request_id"] # Extract request_id from metadata
            indexed_results[request_id] = result[1]  # API response is the second element
        except:
            unknown_id_generations.append(result[1])

    if len(unknown_id_generations) != 0:
        len_unknown = len(unknown_id_generations)
        logger.warning(f"Failed to extract request IDs for {len_unknown} generations, attempting recovery")
        for i in range(len(requests)):
            if i not in indexed_results:
                indexed_results[i] = unknown_id_generations.pop()

    all_text_outputs = []
    for i in range(len(full_generations)):
        cur_results = indexed_results[i]
        if "instruct" in args.model or "davinci" in args.model:
            all_text_outputs.append([cur["text"] for cur in cur_results["choices"]])
        else:
            all_text_outputs.append([cur["message"]["content"] for cur in cur_results["choices"]])

    return prompts, all_text_outputs

def process_prompts(final_subset, args, openai=False, tokenizer_or_encoder=None):
    """
    Extract prompt texts and compute token lengths for generation.
    
    This function handles text preprocessing for both OpenAI and vLLM pipelines,
    supporting both sentence-based and word-based prompt extraction strategies.
    It also handles special processing for the tulu_v1 conversational dataset.
    
    Args:
        final_subset (pd.DataFrame): Dataset containing text snippets or conversations
        args: Command line arguments with extraction parameters
        openai (bool): Whether processing for OpenAI (affects tokenization)
        tokenizer_or_encoder: Tokenizer (vLLM) or encoding (OpenAI) for token counting
    
    Returns:
        Tuple[List[str], List[str], List[int]]:
            - prompt_texts: Extracted text portions to use as prompts
            - rest_of_texts: Remaining text portions (ground truth continuations)
            - token_lengths: Token counts for each text, used for max_length_to_sequence_length
    
    Extraction Methods:
        - Sentence-based: Extract first N sentences using extract_chunk_sentence()
        - Word-based: Remove last N words or proportion using remove_last_n_words()
        - tulu_v1: Extract user/assistant turns from conversational format
    
    Note:
        - For OpenAI: Uses tiktoken encoding without special tokens
        - For vLLM: Uses model tokenizer with add_special_tokens=False
        - Word-based extraction supports both absolute counts and proportions
    """
    if args.task == "tulu_v1": # Special processing for tulu dataset
        assert openai is False # We should only be using this for vLLM models

        # Num turns = 1 by default for now
        prompt_texts = final_subset.messages.apply(lambda x: x[0]['content']).tolist()
        rest_of_texts = final_subset.messages.apply(lambda x: x[1]['content']).tolist()
        
        # Check that the first message is from the user and the second is from the assistant
        assert all(x[0]['role'] == 'user' and x[1]['role'] == 'assistant' for x in final_subset.messages), \
            "Expected first message to be from 'user' and second from 'assistant' in each conversation."

        final_subset["snippet"] = [p + "\n" + r for p, r in zip(prompt_texts, rest_of_texts)]
        args.start_sentence, args.num_sentence = -1, -1

        # tokenizer_or_encoder should be a tokenizer from vllm here
        token_lengths = [len(tokenizer_or_encoder.encode(text, add_special_tokens=False)) for text in rest_of_texts]
    else:
        passages = final_subset.snippet.tolist()
        if not args.prompt_with_words_not_sent: # Use sentences
            prompt_outputs = [extract_chunk_sentence(text, args.start_sentence, args.num_sentences) for text in passages]        
            prompt_texts, rest_of_texts = zip(*prompt_outputs)
            prompt_texts, rest_of_texts = list(prompt_texts), list(rest_of_texts)
            token_lengths = [len(tokenizer_or_encoder.encode(text)) for text in rest_of_texts] if openai else \
                [len(tokenizer_or_encoder.encode(text, add_special_tokens=False)) for text in rest_of_texts]
        else:
            if args.num_words_from_end >= 1:
                if args.num_proportion_from_end != 0:
                    logger.warning("Overriding num_proportion_from_end since num_words_from_end is specified")
                    args.num_proportion_from_end = -1

                data = [remove_last_n_words(tokenizer_or_encoder, text, args.num_words_from_end, openai=openai) for text in passages]
                prompt_texts, rest_of_texts, token_lengths = map(list, zip(*data))
            else:
                assert 0 < args.num_proportion_from_end < 1, "No remove tokens set"
                text_lengths = [len(text.split()) for text in passages] # Get length of words
                remove_lengths = [max(1, min(int(l * args.num_proportion_from_end), l-1)) for l in text_lengths]
                data = [remove_last_n_words(tokenizer_or_encoder, text, remove_length, openai=openai) for text, remove_length in zip(passages, remove_lengths)]
                prompt_texts, rest_of_texts, token_lengths = map(list, zip(*data))

    return prompt_texts, rest_of_texts, token_lengths

def main(args):
    """
    Execute the text generation pipeline with parsed command line arguments.
    
    Args:
        args: Parsed command line arguments containing generation parameters,
              model settings, and data configuration.
    """

    # Seed model and get the model string
    random.seed(args.seed)
    model_str = args.model.split("/")[-1] # Splits to get actual model name
    if model_str not in task_prompts_dict_book[args.task]:
        logger.error(f"Model '{model_str}' not supported for task '{args.task}'. Available models: {list(task_prompts_dict_book[args.task].keys())}")
        return

    # Set up the task prompts
    args.task_prompt_idx = sorted(args.task_prompt_idx)
    cur_task_prompts = []
    for cur_prompt_idx in args.task_prompt_idx:
        cur_task_prompts.append(task_prompts_dict_book[args.task][model_str][cur_prompt_idx])
    chunk_size = len(cur_task_prompts)  # Number of prompt templates - used to map from prompt index back to original text index

    # Load the data
    data_path = f"data/{args.task}/split-random-overall/{args.data_split}.jsonl"
    final_subset = pd.read_json(data_path, lines=True)
    logger.info(f"Loaded {len(final_subset)} samples from {data_path}")
    if args.key_name is not None:
        final_subset["snippet"] = final_subset[args.key_name]
    if args.remove_bad_first: # Remove ill-formatted first sentence
        final_subset["snippet"] = final_subset["snippet"].apply(remove_first_sentence_if_needed)

    save_folder = os.path.join(f"outputs/ours/{args.task}/generations/{args.data_split}")
    os.makedirs(save_folder, exist_ok=True)

    if args.prompt_with_words_not_sent:
        args.start_sentence, args.num_sentences = -1, -1
    else:
        args.start_word, args.num_words_from_end, args.num_proportion_from_end = -1, -1, -1

    if args.temperature == 0: # Reduce num_sequences if using greedy decoding
        logger.info("Using greedy decoding - setting num_sequences to 1")
        args.num_sequences = 1

    # Set up the tokenizer/encoder
    if args.openai:
        import tiktoken
        tokenizer_or_encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
    else:
        generator = ModelGenerator(
            model=args.model,
            tokenizer=args.model if not args.tokenizer else args.tokenizer,
            seed=args.seed,
            hf_token=args.hf_token,
            cache_dir=CACHE_PATH,
        )
        tokenizer_or_encoder = generator.llm.get_tokenizer()

    # Process the prompts
    prompt_texts, rest_of_texts, token_lengths = process_prompts(final_subset, args, args.openai, tokenizer_or_encoder)
    assert None not in prompt_texts

    # Format the prompts
    unmerged_prompts = []
    for cur_task_prompt in cur_task_prompts:
        prompt_kwargs = {
            "prompts": prompt_texts,
            "task_prompt": cur_task_prompt["task_prompt"],
            "task_preprompt": cur_task_prompt["task_preprompt"],
            "task_postprompt": cur_task_prompt["task_postprompt"],
        }
        
        if not args.openai: # Add vLLM-specific parameters
            prompt_kwargs.update({
                "model_name": model_str,
                "prompt_key": "lightest"  # Minimal prompt formatting for vLLM models
            })
        
        unmerged_prompts.append(make_prompts(**prompt_kwargs))
    prompts = zigzag_append(unmerged_prompts) # Make indices match up

    # Generate the text
    if args.openai: # OpenAI models
        final_prompts, all_text_outputs = generate_openai(prompts, chunk_size, args, model_str, token_lengths)
    else: # vLLM models
        if args.max_length_to_sequence_length:
            cur_max_tokens = [element for element in token_lengths for _ in range(chunk_size)]
        else:
            assert args.max_tokens >= 1
            cur_max_tokens = args.max_tokens

        final_prompts, all_text_outputs, all_prompt_logprobs, all_output_logprobs = generator.generate_vllm(
            prompts=prompts,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=cur_max_tokens,
            min_tokens=args.min_tokens,
            max_length=args.max_length,
            n=args.num_sequences
        )

    # Save the results back to the original df
    final_subset["prompt"] = chunk_list(final_prompts, chunk_size)
    final_subset["generation"] = chunk_list(all_text_outputs, chunk_size)
    final_subset["model"] = [model_str] * len(final_subset)
    final_subset["snippet_no_prompt"] = rest_of_texts

    # Convert current datetime to string in 'YYYY-MM-DD HH:MM:SS' format
    date_str = datetime.now().strftime("%Y-%m-%d-%H:%M:%S").strip()
    
    # Save DataFrame to CSV with detailed info in the filename
    file_name = f"""{model_str}_maxTok{args.max_tokens}_minTokStr{args.min_tokens}_numSeq{args.num_sequences}\
_topP{args.top_p}_temp{args.temperature}_numSent{args.num_sentences}_startSent{args.start_sentence}\
_numWordFromEnd{args.num_words_from_end}-{args.num_proportion_from_end}_maxLenSeq{bool_to_first_upper(args.max_length_to_sequence_length)}\
_useSent{bool_to_first_upper(not args.prompt_with_words_not_sent)}-rmvBad{bool_to_first_upper(args.remove_bad_first)}\
_promptIdx{'-'.join(map(str, args.task_prompt_idx))}_len{len(final_subset)}_{date_str}.jsonl"""
    
    file_path = os.path.join(save_folder, file_name)
    columns = [col for col in final_subset.columns if col != 'snippet'] + ['snippet']
    final_subset = final_subset[columns]
    final_subset.to_json(file_path, index=False, lines=True, orient='records')
    logger.info(f"Saved {len(final_subset)} samples to {file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate text continuations for membership inference attack evaluation using OpenAI or vLLM models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Generation parameters
    parser.add_argument('--max_tokens', type=int, default=512, 
                        help='Maximum number of tokens to generate per sequence')
    parser.add_argument('--min_tokens', type=int, default=0, 
                        help='Minimum number of tokens to generate per sequence (vLLM only)')
    parser.add_argument('--max_length', type=int, default=2048, 
                        help='Maximum total sequence length including prompt (vLLM only)')
    parser.add_argument('--num_sequences', type=int, default=1, 
                        help='Number of sequences to generate per prompt (automatically set to 1 for greedy decoding)')
    parser.add_argument('--temperature', type=float, default=1.0, 
                        help='Sampling temperature (0.0 for greedy decoding, higher values for more randomness)')
    parser.add_argument('--top_p', type=float, default=0.95, 
                        help='Top-p (nucleus) sampling threshold (0.0-1.0, lower values for more focused sampling)')
    parser.add_argument('--max_length_to_sequence_length', action="store_true", 
                        help="Set generation length to match the remaining text length")

    # What part of the samples to prompt with
    parser.add_argument('--num_sentences', type=int, default=1, 
                        help='Number of sentences to extract as prompt from each text snippet')
    parser.add_argument('--start_sentence', type=int, default=1, 
                        help='Index of first sentence to extract (0-indexed)')

    parser.add_argument("--prompt_with_words_not_sent", action="store_true", 
                        help="Use word-based splitting instead of sentence-based splitting for prompt extraction")
    parser.add_argument('--num_words_from_end', type=int, default=-1, 
                        help='Number of words to remove from end when using word-based prompts (-1 to disable)')
    parser.add_argument('--num_proportion_from_end', type=float, default=-1, 
                        help='Proportion of words to remove from end (0.0-1.0, -1 to disable)')
    parser.add_argument('--task_prompt_idx', type=int, nargs='+', default=[1], 
                        help='Indices of task-specific prompt templates to use (can specify multiple)')

    # Model details
    parser.add_argument('--model', type=str, default="davinci-002", 
                        help='Model name or path (OpenAI model name or HuggingFace model path)')
    parser.add_argument('--tokenizer', type=str, default=None, 
                        help='Custom tokenizer path for vLLM models (optional, defaults to model tokenizer)')
    parser.add_argument('--hf_token', type=str, default=None, 
                        help='HuggingFace API token for accessing gated models')
    parser.add_argument("--openai", action="store_true", 
                        help="Use OpenAI API for generation (otherwise use vLLM)")
    parser.add_argument("--seed", type=int, default=0, 
                        help="Random seed for reproducible generation")

    # Data details
    parser.add_argument("--task", type=str, default="bookMIA", 
                        help="Dataset/task name (determines prompt templates and data loading)")
    parser.add_argument("--data_split", type=str, default="train", 
                        help="Data split to use (train/val/test)")
    parser.add_argument("--remove_bad_first", action="store_true", 
                        help="Remove malformed first sentences during preprocessing")
    parser.add_argument("--keep_n_sentences", type=int, default=-1, 
                        help="Limit text to first N sentences (-1 for no limit)")
    parser.add_argument("--key_name", type=str, default=None, 
                        help="Custom column name to use as text source (defaults to 'snippet')")

    main(parser.parse_args())