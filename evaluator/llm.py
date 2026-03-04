"""LLM backend abstraction for the evaluator module."""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLMBackend(ABC):
    """Abstract base class for LLM backends.

    Subclasses implement :meth:`query` to send a prompt and return the response.
    """

    @abstractmethod
    def query(self, prompt: str, system_prompt: str = "") -> str:
        """Send a prompt to the LLM and return the response text.

        Args:
            prompt: The user/main prompt.
            system_prompt: Optional system-level instructions.

        Returns:
            The model's response as a string.
        """


class LiteLLMBackend(LLMBackend):
    """LLM backend using litellm (supports OpenAI, Anthropic, Gemini, etc.).

    Supports API key rotation for Gemini models: if GEMINI_API_KEYS_FULL is
    set (comma-separated keys), rotates through keys on rate limit errors.

    Args:
        model: Model identifier (e.g. "gemini/gemini-3-flash-preview").
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in the response.
    """

    def __init__(
        self,
        model: str = "gemini/gemini-3-flash-preview",
        temperature: float = 0.0,
        max_tokens: int | None = 2048,
        response_format: dict | None = None,
    ) -> None:
        import itertools
        import os
        import threading

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.response_format = response_format

        # Build key rotation pool for Gemini models
        self._key_cycle = None
        self._key_lock = threading.Lock()
        if "gemini" in model.lower():
            keys_str = os.environ.get("GEMINI_API_KEYS_FULL", "")
            if keys_str:
                keys = [k.strip() for k in keys_str.split(",") if k.strip()]
                if len(keys) > 1:
                    self._key_cycle = itertools.cycle(keys)
                    import logging
                    logging.getLogger(__name__).info(
                        "Key rotation enabled: %d Gemini API keys", len(keys)
                    )

    def _next_api_key(self) -> str | None:
        """Get next API key from rotation pool (thread-safe)."""
        if self._key_cycle is None:
            return None
        with self._key_lock:
            return next(self._key_cycle)

    def query(self, prompt: str, system_prompt: str = "") -> str:
        import logging
        import time

        import litellm

        log = logging.getLogger(__name__)

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        max_retries = 10
        for attempt in range(max_retries):
            try:
                # Rotate API key on each attempt if pool available
                kwargs: dict = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                }
                if self.max_tokens is not None:
                    kwargs["max_tokens"] = self.max_tokens
                if self.response_format is not None:
                    kwargs["response_format"] = self.response_format
                api_key = self._next_api_key()
                if api_key:
                    kwargs["api_key"] = api_key

                response = litellm.completion(**kwargs)
                return response.choices[0].message.content or ""
            except (
                litellm.exceptions.ServiceUnavailableError,
                litellm.exceptions.RateLimitError,
                litellm.exceptions.APIConnectionError,
                litellm.exceptions.Timeout,
                litellm.exceptions.InternalServerError,
                litellm.exceptions.BadRequestError,
            ) as e:
                if attempt == max_retries - 1:
                    raise
                if isinstance(e, litellm.exceptions.RateLimitError):
                    # With key rotation, short backoff is enough
                    wait = min(5 * (2 ** attempt), 60) if self._key_cycle else min(30 * (2 ** attempt), 240)
                else:
                    wait = min(2 ** attempt + 1, 60)
                log.warning("LLM call failed (attempt %d/%d): %s. Retrying in %ds...", attempt + 1, max_retries, type(e).__name__, wait)
                time.sleep(wait)
