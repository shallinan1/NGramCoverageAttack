# Generation Module

Text generation utilities for parallel processing of language model requests using OpenAI API and vLLM. This module provides efficient, rate-limited text generation with support for various model architectures and prompt formatting strategies.

## Overview

This module contains utilities for generating text using various APIs and local models, with a focus on parallel processing, rate limiting, and model-specific prompt formatting.

## Files

- **`openai_parallel_generate.py`** - Parallel text generation using OpenAI API with rate limiting
- **`rate_limits.py`** - Rate limit configurations for different OpenAI models
- **`api_request_parallel_processor.py`** - Core parallel request processing logic (modified from OpenAI Cookbook)
- **`prompt_formatting.py`** - Model-specific prompt formatting utilities
- **`vllm_generate.py`** - High-performance text generation using vLLM for local models

## ⚠️ Critical Configuration

### Rate Limits (MUST CONFIGURE)

**Before using the OpenAI generation utilities, you must configure the rate limits in `rate_limits.py` according to your OpenAI account's actual limits.**

1. Visit https://platform.openai.com/account/rate-limits to check your current rate limits
2. Update the values in `rate_limits.py` for the models you plan to use
3. The `SAFETY_MARGIN` factor (default 0.75) is applied to prevent hitting actual API limits

**Important**: The default values in `rate_limits.py` are examples only. Using incorrect rate limits will result in API errors or suboptimal performance.

## Usage Examples

### OpenAI Parallel Generation

```python
from src.generation.openai_parallel_generate import openai_parallel_generate
from src.generation.rate_limits import requests_limits_dict, requests_url_dict
import asyncio

# Prepare your requests
requests = [
    {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Tell me a story"}]},
    {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Explain quantum physics"}]}
]

# Run parallel generation
results = asyncio.run(openai_parallel_generate(
    requests=requests,
    max_requests_per_minute=requests_limits_dict["gpt-3.5-turbo-0125"]["max_requests_per_minute"],
    max_tokens_per_minute=requests_limits_dict["gpt-3.5-turbo-0125"]["max_tokens_per_minute"],
    request_url=requests_url_dict["gpt-3.5-turbo-0125"]
))
```

### vLLM Generation

```python
from src.generation.vllm_generate import ModelGenerator

# Initialize generator
generator = ModelGenerator(
    model="meta-llama/Llama-2-7b-hf",
    hf_token="your_hf_token",  # if needed for private models
    gpu_memory_utilization=0.8
)

# Generate text
prompts = ["Once upon a time", "The future of AI"]
final_prompts, outputs, prompt_logprobs, output_logprobs = generator.generate_vllm(
    prompts=prompts,
    temperature=0.7,
    max_new_tokens=256
)
```

### Prompt Formatting

```python
from src.generation.prompt_formatting import make_prompts, task_prompts_dict_book

# Format prompts for a specific model
formatted_prompts = make_prompts(
    prompts=["Once upon a time"],
    task_prompt="Continue the story: ",
    model_name="llama-2-7b-chat",
    prompt_key="lightest"
)

# Use predefined task prompts
book_prompts = task_prompts_dict_book["bookMIA"]["instruct-autoregressive"]
```

## Key Features

### Parallel Request Processing
- Handles multiple API requests concurrently with automatic rate limiting
- Intelligent retry logic with exponential backoff for failed requests
- Dual progress bars showing launched vs completed jobs
- Memory-efficient streaming for large batches

### Model Support
- **OpenAI Models**: GPT-3.5, GPT-4, GPT-4o variants
- **Open Models via vLLM**: LLaMA 2/3/3.1, Mistral, Gemma, OLMo, Tulu
- **Instruction-tuned variants**: Automatic detection and formatting

### Prompt Formatting
The module handles model-specific formatting requirements:
- LLaMA 2/3 chat templates with system instructions
- Mistral/Mixtral instruction formatting
- Gemma conversation formatting
- Custom task-specific prompts for different datasets

### Rate Limiting
- Configurable requests per minute (RPM) and tokens per minute (TPM)
- Safety margins to prevent hitting API limits
- Automatic capacity tracking and throttling

## Important Notes

### Token Limits
- vLLM max sequence length is set to 2048 tokens for memory efficiency
- OpenAI uses cl100k_base encoding for token counting
- Automatic prompt truncation when exceeding limits

### Error Handling
- Maximum 5 retry attempts for failed API requests
- 15-second cooldown after rate limit errors
- Comprehensive error logging at different verbosity levels

### Performance Considerations
- Use batch processing for similar requests to maximize throughput
- Adjust `gpu_memory_utilization` based on your GPU capacity
- Consider using tensor parallelism for multi-GPU setups with vLLM

## Testing

Test individual components:

```bash
# Test OpenAI parallel generation
python -m src.generation.openai_parallel_generate

# Test vLLM generation
python -m src.generation.vllm_generate
```

## Troubleshooting

### Common Issues

1. **Rate Limit Errors**: Update `rate_limits.py` with your actual OpenAI limits
2. **GPU Memory Issues**: Reduce `gpu_memory_utilization` or `max_model_len` in vLLM
3. **Token Encoding Errors**: Ensure using correct encoding for your model type
4. **Prompt Formatting Issues**: Check model name detection in `prompt_formatting.py`

### Logging Levels

Set logging level when calling the parallel processor:
- 40 (ERROR): Only log failures after all retries
- 30 (WARNING): Log rate limits and errors
- 20 (INFO): Log request starts and completions
- 10 (DEBUG): Detailed debugging information