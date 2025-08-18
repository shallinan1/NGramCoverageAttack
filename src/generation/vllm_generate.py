"""
vLLM-based text generation utility for large language models.

This module provides a ModelGenerator class that uses vLLM for efficient parallel
text generation from Hugging Face models. It supports various model types including
LLaMA, OLMo, and other transformer models with optimized GPU utilization.

Key Features:
    - Parallel text generation using vLLM
    - Automatic tokenization and prompt handling
    - GPU memory optimization
    - Support for various sampling parameters
    - Multi-GPU tensor parallelism

Requirements:
    - vLLM library
    - PyTorch with CUDA support
    - Hugging Face transformers
    - Optional: Hugging Face token for private models

Usage:
    # Initialize generator
    generator = ModelGenerator(
        model="meta-llama/Llama-2-7b-hf",
        hf_token="your_hf_token",  # if needed
        gpu_memory_utilization=0.8
    )
    
    # Generate text
    prompts = ["Tell me about", "The future of AI"]
    final_prompts, outputs, prompt_logprobs, output_logprobs = generator.generate_vllm(
        prompts=prompts,
        temperature=0.7,
        max_new_tokens=256
    )

Example:
    python -m src.generation.vllm_generate

Important Configuration Notes:
    - max_model_len is hardcoded to 2048 tokens for memory efficiency with large models
    - Returns top 20 log probabilities for each token
    - Commented out prompt_logprobs, but can be uncommented if needed
    - Uses left padding for batch processing efficiency
"""

from huggingface_hub import login
import argparse
from transformers import AutoTokenizer, set_seed
import torch
import os
import json
from IPython import embed
from vllm import LLM, SamplingParams
from typing import List, Tuple, Optional, Union

class ModelGenerator:
    """
    A class for generating text using a pre-trained language model with the vLLM library.

    Args:
        model (str): The model identifier or path to the pre-trained model.
        tokenizer (str, optional): The tokenizer identifier or path. If None, uses the same identifier as the model. Defaults to None.
        seed (int, optional): Random seed for reproducibility. Defaults to 0.
        hf_token (str, optional): Hugging Face token for authentication. Required if accessing models from Hugging Face Hub. Defaults to None.
        cache_dir (str, optional): Directory to cache the model and tokenizer files. Defaults to "../cache/".

    Attributes:
        generate_function (function): The function used to generate text. Currently set to `generate_vllm`.
        llm (LLM): The language model object loaded with vLLM.
        tokenizer (AutoTokenizer): The tokenizer associated with the model for encoding and decoding text.
    """

    def __init__(
        self,
        model: str,
        tokenizer: str = None,
        seed: int = 0,
        hf_token: str = None,
        cache_dir: str = "../cache/",
        gpu_memory_utilization: int = 0.9
    ):
        print("Initializing vLLM model")
        set_seed(seed) # Set the random seed for reproducibility
        self.generate_function = self.generate_vllm
        
        if hf_token:
            login(hf_token) # Log in to Hugging Face if a token is provided

        print(f"Device count: {torch.cuda.device_count()}")

        # Load the model using the vLLM library
        self.llm = LLM(
            model=model,
            tensor_parallel_size=torch.cuda.device_count(), # Install ray if > 1 GPUs
            download_dir=cache_dir,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len = 2048, # Maximum sequence length - set to 2048 for memory efficiency with large models
        )
        
        # Create the tokenizer
        tokenizer_str = tokenizer if tokenizer else model  # Use a cached tokenizer for efficiency if provided
        add_bos_token = True if "OLMo" not in model else False # Adding the BOS token explicitly as some models have it set to False by default
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_str,
            padding_side="left",
            trust_remote_code=True,
            add_bos_token=add_bos_token
        )

    def generate_vllm(
        self,
        prompts: List[str],
        temperature: float = 1.0,
        top_p: float = 0.95,
        sample: bool = True,
        max_new_tokens: Union[int, List[int]] = 256,
        min_tokens: Optional[int] = None,
        max_length: int = 2048,
        extra_stop_tokens: Optional[List[int]] = None,
        n: int = 1,
        top_k: int = -1
    ) -> Tuple[List[str], List[str]]:
        """
        Generates text based on a list of input prompts using a language model with specific sampling parameters.

        Args:
            prompts (List[str]): A list of input text prompts for which to generate continuations.
            temperature (float, optional): Controls randomness in sampling. A higher value produces more diverse outputs. 
                                        Set to 0 for greedy decoding. Defaults to 1.0.
            top_p (float, optional): Nucleus sampling parameter. The probability threshold for top-p sampling. Defaults to 0.95.
            sample (bool, optional): If True, uses sampling; otherwise, sets temperature to 0 for deterministic decoding. Defaults to True.
            max_new_tokens (int, optional): The maximum number of new tokens to generate for each input prompt. Defaults to 256.
            min_tokens (int, optional): Force to generate at least min tokens
            max_length (int, optional): The maximum length (in tokens) of the input prompt plus the generated text. Defaults to 2048.
            extra_stop_tokens (List[int], optional): Additional token IDs that should stop the generation. Defaults to None.
            n (int, optional): The number of sequences to return for each prompt.
            top_k (int, optional): Top-k sampling parameter. The number of highest probability tokens to consider for sampling. 
                                 Set to -1 for no top-k filtering. Defaults to -1.

        Returns:
            Tuple[List[str], List[str]]: A tuple containing a list of final prompts after filtering and a list of generated outputs corresponding to those prompts.
        """
        if isinstance(max_new_tokens, int):
            max_new_tokens = [max_new_tokens] * len(prompts)
        
        # Convert extra stop tokens to integers if provided
        extra_stop_tokens = [int(a) for a in extra_stop_tokens] if extra_stop_tokens else []

        outputs, final_prompts = [], []
        print(f"Number of prompts to generate: {len(prompts)}")

        # Tokenize prompts without padding and truncate to max_length
        input_ids_list = self.tokenizer(prompts, truncation=True, max_length=max_length).input_ids
        filtered_input_ids = []
        truncated = 0

        # Filter out prompts that are too long and check for padding tokens
        for prompt, input_ids, max_new_token in zip(prompts, input_ids_list, max_new_tokens):
            assert self.tokenizer.pad_token_id not in input_ids, "Padding tokens found in input_ids."
            if len(input_ids) + max_new_token <= max_length:
                filtered_input_ids.append(input_ids)
                final_prompts.append(prompt)
            else:
                # Prompt + requested output would exceed max_length, so truncate the prompt
                truncated += 1
                print(f"Truncating from {len(input_ids)} to {max_length-max_new_tokens}")
                
                # Calculate how much space to leave for the requested output tokens
                new_max_length = max_length - max_new_token
                
                # Handle edge case where requested output tokens exceed total max_length
                if new_max_length < 0:
                    print("Warning: max_new_tokens exceeds max_length, using full context for prompt")
                    new_max_length = max_length  # Use full context, output may be truncated
                
                # Truncate prompt to fit within the calculated length and re-tokenize
                shortened = self.tokenizer([prompt], truncation=True, max_length=new_max_length).input_ids[0]
                filtered_input_ids.append(shortened)
                # Convert truncated tokens back to text for the final prompt list
                final_prompts.append(self.tokenizer.decode(shortened))
        print(f"Truncated {truncated} out of {len(final_prompts)}")
        
        # Set up the sampling parameters
        stop_token_ids = list(set(extra_stop_tokens + [self.llm.get_tokenizer().eos_token_id]))
        if not sample:
            temperature = 0  # vllm will always use sampling unless temperature = 0

        sampling_params = [SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=mnt,
            min_tokens=min_tokens,
            stop_token_ids=stop_token_ids,
            n=n,
            logprobs=20,  # Number of top log probabilities to return for each token
            # prompt_logprobs=5  # Uncomment to return number of prompt token log probabilities
        ) for mnt in max_new_tokens]

        # Generation
        output = self.llm.generate(
            sampling_params=sampling_params, 
            prompt_token_ids=filtered_input_ids
        )
        
        all_text_outputs = []
        all_prompt_logprobs = []
        all_output_logprobs = []
        
        # Decode the generated outputs
        for o in output:
            cur_output = []
            cur_output_logprobs = []
            for generated_output in o.outputs:
                output_str = self.tokenizer.decode(generated_output.token_ids, skip_special_tokens=True).strip()
                cur_output.append(output_str)
                cur_output_logprobs.append(generated_output.logprobs)

            all_prompt_logprobs.append(o.prompt_logprobs)
            all_text_outputs.append(cur_output)
            all_output_logprobs.append(cur_output_logprobs)
        
        return final_prompts, all_text_outputs, all_prompt_logprobs, all_output_logprobs 