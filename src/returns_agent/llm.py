from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from anthropic import AnthropicError
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable

from returns_agent.models import AgentResponse, CustomerReplyDraft
from returns_agent.tools import ORDER_TOOLS


# Base error so CLI/UI can catch LLM-related failures in one place.
class LLMError(Exception):
    pass


# Raised before calling the provider if required env vars are missing.
class LLMConfigurationError(LLMError):
    pass


# Raised when the model response cannot be parsed into our schemas.
class LLMResponseError(LLMError):
    pass


# Raised when the provider rejects the request, like quota or rate limits.
class LLMProviderError(LLMError):
    pass


# Model settings come from env so the code does not hardcode secrets or models.
@dataclass(frozen=True)
class LLMSettings:
    model: str
    temperature: float | None = None
    max_retries: int = 2
    max_tokens: int | None = None
    request_timeout: float | None = None

    # Read runtime config from .env or environment variables.
    @classmethod
    def from_env(cls) -> LLMSettings:
        load_dotenv()

        return cls(
            model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"),
            temperature=_optional_float("LLM_TEMPERATURE"),
            max_retries=int(os.getenv("LLM_MAX_RETRIES", "2")),
            max_tokens=_optional_int("LLM_MAX_TOKENS"),
            request_timeout=_optional_float("LLM_REQUEST_TIMEOUT"),
        )


# the actual Anthropic chat model used by all three LLM paths.
def build_chat_model(settings: LLMSettings | None = None) -> ChatAnthropic:
    settings = settings or LLMSettings.from_env()
    _require_anthropic_api_key()

    kwargs: dict[str, Any] = {
        "model_name": settings.model,
        "max_retries": settings.max_retries,
        "max_tokens": settings.max_tokens,
        "timeout": settings.request_timeout,
    }

    # optional temp 
    if settings.temperature is not None and _supports_temperature(settings.model):
        kwargs["temperature"] = settings.temperature

    return ChatAnthropic(**kwargs)


# Main agent model: this one is allowed to call get_order.
def bind_order_tools(model: BaseChatModel | None = None) -> Runnable[Any, AIMessage]:
    chat_model = model or build_chat_model()
    return chat_model.bind_tools(ORDER_TOOLS)


# Structured routing model: returns AgentResponse JSON for graph routing.
def build_agent_response_model(
    model: BaseChatModel | None = None,
) -> Runnable[Any, dict[str, Any]]:
    chat_model = model or build_chat_model()
    return chat_model.with_structured_output(
        schema=AgentResponse.model_json_schema(),
        method="json_schema",
    )


# Reply model: writes only the customer_reply field after policy has run.
def build_customer_reply_model(
    model: BaseChatModel | None = None,
) -> Runnable[Any, dict[str, Any]]:
    chat_model = model or build_chat_model()
    return chat_model.with_structured_output(
        schema=CustomerReplyDraft.model_json_schema(),
        method="json_schema",
    )


# Fail early with a useful message if the key is missing.
def _require_anthropic_api_key() -> None:
    load_dotenv()

    if os.getenv("ANTHROPIC_API_KEY"):
        return

    raise LLMConfigurationError("Missing Anthropic API key. Set ANTHROPIC_API_KEY.")


# Optional integer env vars should stay None when not set.
def _optional_int(name: str) -> int | None:
    value = os.getenv(name)
    return int(value) if value else None


# Optional float env vars should stay None when not set.
def _optional_float(name: str) -> float | None:
    value = os.getenv(name)
    return float(value) if value else None


# Provider compatibility helper for models that do not accept temperature.
def _supports_temperature(model: str) -> bool:
    no_temperature_prefixes = (
        "claude-sonnet-5",
        "claude-opus-4-8",
        "claude-fable-5",
        "claude-mythos",
    )

    return not model.startswith(no_temperature_prefixes)


# Convert provider errors into messages that make sense from CLI/Streamlit.
def provider_error_message(exc: AnthropicError) -> str:
    message = str(exc)

    if "rate_limit" in message.casefold() or "429" in message:
        return (
            "Anthropic quota or rate limit was hit. Wait for the retry window "
            "or use a model/account with available quota."
        )

    return f"Anthropic request failed: {message}"
