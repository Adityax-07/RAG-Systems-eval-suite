"""LLM client utilities for async API calls."""

import asyncio
import logging
import re
from typing import Optional, Any
from abc import ABC, abstractmethod
import json

from openai import AsyncOpenAI, OpenAI
from openai import RateLimitError as OpenAIRateLimitError
from anthropic import Anthropic, AsyncAnthropic
from tenacity import (
    retry,
    retry_if_exception_type,
    wait_exponential,
    stop_after_attempt,
    before_sleep_log,
)

from ..config import get_settings

logger = logging.getLogger(__name__)


def _parse_retry_after(exc: Exception) -> float:
    """Extract 'retry after N seconds' from a Groq 429 error message."""
    msg = str(exc)
    match = re.search(r"try again in ([\d\.]+)s", msg)
    if match:
        return min(float(match.group(1)) + 1.0, 60.0)
    return 5.0


async def _groq_retry_wait(retry_state):
    """Custom wait that honours the 'retry after Xs' hint in the error."""
    exc = retry_state.outcome.exception()
    secs = _parse_retry_after(exc) if exc else 5.0
    logger.warning(f"Rate limited — waiting {secs:.1f}s before retry")
    await asyncio.sleep(secs)


_groq_retry = retry(
    retry=retry_if_exception_type(OpenAIRateLimitError),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(8),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=False,
)


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    async def generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.1,
        top_p: float = 0.9,
        max_tokens: int = 2000,
    ) -> str:
        """Generate text from the LLM."""
        pass
    
    @abstractmethod
    async def generate_json(
        self,
        system_prompt: str,
        user_message: str,
        schema: dict,
        temperature: float = 0.1,
        top_p: float = 0.9,
        max_tokens: int = 2000,
    ) -> dict:
        """Generate structured JSON from the LLM."""
        pass


class OpenAIClient(LLMClient):
    """Async OpenAI client wrapper."""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        settings = get_settings()
        self.client = AsyncOpenAI(api_key=api_key or settings.openai_api_key)
        self.model = model or settings.openai_model
    
    async def generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.1,
        top_p: float = 0.9,
        max_tokens: int = 2000,
    ) -> str:
        """Generate text using OpenAI API."""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    async def generate_json(
        self,
        system_prompt: str,
        user_message: str,
        schema: dict,
        temperature: float = 0.1,
        top_p: float = 0.9,
        max_tokens: int = 2000,
    ) -> dict:
        """Generate structured JSON using OpenAI API."""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "evaluation_result",
                        "schema": schema,
                        "strict": True,
                    },
                },
            )
            content = response.choices[0].message.content or "{}"
            return json.loads(content)
        except Exception as e:
            logger.error(f"OpenAI JSON generation error: {e}")
            raise


class AnthropicClient(LLMClient):
    """Async Anthropic client wrapper."""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        settings = get_settings()
        self.client = AsyncAnthropic(api_key=api_key or settings.anthropic_api_key)
        self.model = model or settings.anthropic_model
    
    async def generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.1,
        top_p: float = 0.9,
        max_tokens: int = 2000,
    ) -> str:
        """Generate text using Anthropic API."""
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_message},
                ],
                temperature=temperature,
                top_p=top_p,
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise
    
    async def generate_json(
        self,
        system_prompt: str,
        user_message: str,
        schema: dict,
        temperature: float = 0.1,
        top_p: float = 0.9,
        max_tokens: int = 2000,
    ) -> dict:
        """Generate structured JSON using Anthropic API."""
        # Anthropic doesn't have native JSON mode, so we format the request carefully
        json_instructions = f"""
Please respond with valid JSON matching this schema:
{json.dumps(schema, indent=2)}

Only return the JSON object, no other text.
"""
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system_prompt + "\n" + json_instructions,
                messages=[
                    {"role": "user", "content": user_message},
                ],
                temperature=temperature,
                top_p=top_p,
            )
            content = response.content[0].text
            # Try to parse JSON from the response
            return json.loads(content)
        except Exception as e:
            logger.error(f"Anthropic JSON generation error: {e}")
            raise


class GroqClient(LLMClient):
    """Async Groq client wrapper.

    Groq exposes an OpenAI-compatible REST API, so we reuse AsyncOpenAI with a
    custom base_url.  Note: Groq does NOT support the ``response_format`` JSON
    schema mode, so ``generate_json`` falls back to prompt-level instructions
    and normal JSON parsing (same approach as AnthropicClient).
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        settings = get_settings()
        self.client = AsyncOpenAI(
            api_key=api_key or settings.groq_api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        self.model = model or settings.groq_model

    @_groq_retry
    async def generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.1,
        top_p: float = 0.9,
        max_tokens: int = 2000,
    ) -> str:
        """Generate text using Groq API (auto-retries on 429)."""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    @_groq_retry
    async def generate_json(
        self,
        system_prompt: str,
        user_message: str,
        schema: dict,
        temperature: float = 0.1,
        top_p: float = 0.9,
        max_tokens: int = 2000,
    ) -> dict:
        """Generate structured JSON using Groq API (prompt-based, auto-retries on 429)."""
        json_instructions = (
            f"\nRespond with valid JSON matching this schema:\n"
            f"{json.dumps(schema, indent=2)}\n"
            "Only return the JSON object, no other text."
        )
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt + json_instructions},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content or "{}"
        return json.loads(content)


def get_llm_client(provider: str = "groq", model: Optional[str] = None) -> LLMClient:
    """Get an LLM client based on provider name.

    Args:
        provider: One of "openai", "anthropic", "groq".
        model: Optional model override. If None, uses the provider's default
               from settings. Pass a specific model name to use a different
               model than the default (e.g. a cheaper judge model).
    """
    if provider.lower() == "openai":
        return OpenAIClient(model=model)
    elif provider.lower() == "anthropic":
        return AnthropicClient(model=model)
    elif provider.lower() == "groq":
        return GroqClient(model=model)
    else:
        raise ValueError(f"Unknown provider: {provider}. Choose from: openai, anthropic, groq")
