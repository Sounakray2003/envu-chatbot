import logging
import os
from typing import Optional
from transformers import AutoTokenizer, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

MODEL_MAPPING = {
    "bge-large": "BAAI/bge-large-en-v1.5",
}
LOCAL_EMBEDDING_MODEL_PATH_ENV = "BGE_LARGE_MODEL_PATH"

_tokenizer_cache = {}


def normalize_model_name(model_name: str) -> str:
    """Normalize model name to lowercase with hyphens."""
    return model_name.lower().replace("_", "-").strip()


def get_model_huggingface_id(model_name: str) -> Optional[str]:
    """
    Get the HuggingFace model ID for a given model name.
    
    Args:
        model_name: The model name (e.g., "bge-large", "BGE-Large")
    
    Returns:
        The HuggingFace model ID, or None if not found
    """
    normalized = normalize_model_name(model_name)
    return MODEL_MAPPING.get(normalized)


def get_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    """
    Load and cache a tokenizer for the given model using AutoTokenizer.
    
    Args:
        model_name: The model name (e.g., "bge-large", "BGE-Large")
    
    Returns:
        The loaded tokenizer
    
    Raises:
        ValueError: If the model is not supported
        Exception: If the tokenizer fails to load
    """
    normalized = normalize_model_name(model_name)
    
    # Check cache first
    if normalized in _tokenizer_cache:
        logger.debug(f"Returning cached tokenizer for {model_name}")
        return _tokenizer_cache[normalized]
    
    # Get the HuggingFace model ID
    hf_model_id = get_model_huggingface_id(normalized)
    if not hf_model_id:
        raise ValueError(
            f"Model '{model_name}' is not supported. "
            f"Supported models: {list(MODEL_MAPPING.keys())}"
        )
    
    model_source = str(os.getenv(LOCAL_EMBEDDING_MODEL_PATH_ENV, "")).strip() or hf_model_id

    try:
        logger.info(
            "Loading tokenizer for %s from %s (first keyword search may take a little longer)",
            model_name,
            model_source,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_source,
            local_files_only=True,
        )
        _tokenizer_cache[normalized] = tokenizer
        return tokenizer
    except Exception as e:
        logger.error(f"Failed to load tokenizer for {model_name}: {e}")
        raise


def get_max_tokens(model_name: str) -> int:
    """
    Get the maximum token length for a model.
    
    Args:
        model_name: The model name (e.g., "bge-large", "BGE-Large")
    
    Returns:
        The maximum token length, or a default value if not defined
    """
    try:
        tokenizer = get_tokenizer(model_name)
        max_tokens = tokenizer.model_max_length
        if max_tokens > 100000:
            return 512  # Default for models without explicit max length
        return max_tokens
    except Exception as e:
        logger.warning(f"Could not determine max tokens for {model_name}: {e}")
        return 512  # Default fallback


def print_model_stats():
    """Print max tokens for all supported models (for debugging)."""
    for model in MODEL_MAPPING:
        try:
            max_tokens = get_max_tokens(model)
            print(f"{model}: {max_tokens}")
        except Exception as e:
            print(f"{model}: Error loading tokenizer ({e})")
