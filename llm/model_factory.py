"""
model_factory.py - LangChain Built-in Model Initializer

Uses LangChain's official built-in function `init_chat_model`:
  from langchain.chat_models import init_chat_model

This is LangChain's standard, universal way to initialize any LLM across providers
(Google GenAI, OpenAI, Anthropic, Ollama, etc.) with a single function call.
"""

from typing import Optional, Any
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.chat_models import init_chat_model
from config.settings import get_settings


def get_chat_model(
    model: Optional[str] = None,
    model_provider: Optional[str] = None,
    temperature: Optional[float] = None,
    **kwargs: Any,
) -> BaseChatModel:
    """
    Initializes a LangChain chat model using LangChain's official `init_chat_model()`.

    To switch providers, change LLM_PROVIDER in .env:
      - LLM_PROVIDER=google (uses model_provider='google_genai')
      - LLM_PROVIDER=openai (uses model_provider='openai')

    Args:
        model: Specific model name (e.g., 'gemini-3.5-flash-lite' or 'gpt-4o-mini').
        model_provider: Provider name ('google_genai', 'openai', 'anthropic', etc.).
        temperature: Optional temperature parameter.
        **kwargs: Additional parameters passed to the model.

    Returns:
        BaseChatModel: Standard LangChain Chat Model instance.
    """
    settings = get_settings()

    # Determine provider: map "google" -> "google_genai" for LangChain
    provider_str = (model_provider or settings.provider).lower()
    if provider_str in ["google", "gemini", "google_genai"]:
        resolved_provider = "google_genai"
        resolved_model = model or settings.gemini_model
        api_key = settings.gemini_api_key
        kwargs["google_api_key"] = api_key
    elif provider_str == "openai":
        resolved_provider = "openai"
        resolved_model = model or settings.openai_model
        api_key = settings.openai_api_key
        kwargs["api_key"] = api_key
    else:
        resolved_provider = provider_str
        resolved_model = model

    # Optional parameters
    if temperature is not None:
        kwargs["temperature"] = temperature

    # Official LangChain universal chat model initializer
    return init_chat_model(
        resolved_model,
        model_provider=resolved_provider,
        **kwargs,
    )
