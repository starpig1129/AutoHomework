# -*- coding: utf-8 -*-
"""Factory for creating LLM client instances based on configuration."""

import logging
from typing import Any, Dict, Type

from .anthropic_client import AnthropicClient
from .base import LLMInterface
from .gemini_client import GeminiClient # Import the placeholder
from .openai_client import OpenAIClient

logger = logging.getLogger(__name__)

# Mapping from configuration keys to client classes
_LLM_CLIENT_MAP: Dict[str, Type[LLMInterface]] = {
    "openai": OpenAIClient,
    "anthropic": AnthropicClient,
    "gemini": GeminiClient, # Add Gemini to the map
}


class LLMFactory:
  """Factory class to create instances of LLM clients."""

  @staticmethod
  def create_llm_client(
      provider: str, api_key: str, model_name: str, **kwargs: Any
  ) -> LLMInterface:
    """Creates an LLM client instance based on the specified provider.

    Args:
        provider: The name of the LLM provider (e.g., "openai", "anthropic", "gemini").
                  Must match a key in _LLM_CLIENT_MAP.
        api_key: The API key for the specified provider.
        model_name: The specific model name to use for the client.
        **kwargs: Additional keyword arguments to pass to the client's constructor
                  (e.g., max_tokens, temperature).

    Returns:
        An instance of the corresponding LLMInterface implementation.

    Raises:
        ValueError: If the specified provider is not supported.
        Exception: If client initialization fails.
    """
    provider_key = provider.lower()
    client_class = _LLM_CLIENT_MAP.get(provider_key)

    if client_class:
      logger.info(
          "Creating LLM client for provider: %s, model: %s",
          provider,
          model_name,
      )
      try:
        # Pass api_key, model_name, and any other relevant kwargs
        instance = client_class(
            api_key=api_key, model_name=model_name, **kwargs
        )
        return instance
      except Exception as e:
        logger.exception(
            "Failed to initialize LLM client for provider %s, model %s: %s",
            provider,
            model_name,
            e,
        )
        raise ValueError(
            f"Failed to initialize LLM client for provider '{provider}'. Error: {e}"
        ) from e
    else:
      logger.error("Unsupported LLM provider specified: %s", provider)
      raise ValueError(
          f"Unsupported LLM provider: '{provider}'. Supported providers are: {list(_LLM_CLIENT_MAP.keys())}"
      )