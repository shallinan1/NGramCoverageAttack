"""
OpenAI API rate limits and endpoint configurations.

This module defines rate limits and API endpoints for different OpenAI models.
These values should be updated based on your actual OpenAI account limits.

Important:
    - Check your rate limits at: https://platform.openai.com/account/rate-limits
    - Update the values below to match your account's actual limits
    - The SAFETY_MARGIN is applied to prevent hitting rate limits during parallel processing

Usage:
    from src.generation.rate_limits import requests_limits_dict, requests_url_dict
    
    # Get rate limits for a specific model
    limits = requests_limits_dict["gpt-3.5-turbo-0125"]
    rpm = limits["max_requests_per_minute"]
    tpm = limits["max_tokens_per_minute"]
    
    # Get API endpoint for a model
    url = requests_url_dict["gpt-3.5-turbo-0125"]
"""

# Safety margin factor to prevent hitting actual API limits
SAFETY_MARGIN = 0.75

# Rate limits for different OpenAI models
# Contains max_requests_per_minute and max_tokens_per_minute for each model
# Values are adjusted with safety margins to prevent hitting actual limits
requests_limits_dict = {
    "gpt-3.5-turbo-0125": {
        "max_requests_per_minute": int(10000 * SAFETY_MARGIN),
        "max_tokens_per_minute": int(50000000 * SAFETY_MARGIN)
    },
    "gpt-4-0613": {
        "max_requests_per_minute": int(10000 * SAFETY_MARGIN),
        "max_tokens_per_minute": int(1000000 * SAFETY_MARGIN)
    },
    "gpt-3.5-turbo-instruct": {
        "max_requests_per_minute": int(3500 * SAFETY_MARGIN),
        "max_tokens_per_minute": int(90000 * SAFETY_MARGIN)
    },
    "gpt-4o-mini-2024-07-18": {
        "max_requests_per_minute": int(30000 * SAFETY_MARGIN),
        "max_tokens_per_minute": int(150000000 * SAFETY_MARGIN)
    },
    "gpt-4o-2024-11-20": {
        "max_requests_per_minute": int(10000 * SAFETY_MARGIN),
        "max_tokens_per_minute": int(30000000 * SAFETY_MARGIN)
    },
    "gpt-4-turbo-2024-04-09": {
        "max_requests_per_minute": int(10000 * SAFETY_MARGIN),
        "max_tokens_per_minute": int(2000000 * SAFETY_MARGIN)
    },
}

# Models that share the same rate limits as other models
requests_limits_dict["gpt-3.5-turbo-1106"] = requests_limits_dict["gpt-3.5-turbo-0125"]
requests_limits_dict["gpt-4-0314"] = requests_limits_dict["gpt-4-0613"]
requests_limits_dict["o1-mini-2024-09-12"] = requests_limits_dict["gpt-4o-mini-2024-07-18"]

# API endpoints for different model types
# Maps model names to the appropriate OpenAI API endpoint URL
# Some models use chat/completions while others use completions endpoint
requests_url_dict = {
    "gpt-3.5-turbo-0125": "https://api.openai.com/v1/chat/completions",
    "gpt-3.5-turbo-instruct": "https://api.openai.com/v1/completions"
}

# Models that use the chat completions endpoint
for model in ["gpt-4-0613", "gpt-3.5-turbo-1106", "gpt-4-0314","gpt-4o-mini-2024-07-18", "o1-mini-2024-09-12", "gpt-4o-2024-11-20", "gpt-4-turbo-2024-04-09"]:
    requests_url_dict[model] = requests_url_dict["gpt-3.5-turbo-0125"]