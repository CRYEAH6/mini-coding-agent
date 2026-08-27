"""DeepSeek API client used by the agent loop."""

from typing import Any, Mapping, Optional, Sequence

from openai import OpenAI

from mini_agent.config import Settings


class DeepSeekClient:
    """Send chat-completion requests through DeepSeek's compatible API."""

    def __init__(self, settings: Settings, client: Optional[Any] = None) -> None:
        self._settings = settings
        self._client = client or OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=settings.timeout_seconds,
            max_retries=settings.max_retries,
        )

    def create_message(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> Any:
        """Return the first assistant message from a model response."""
        request: dict[str, Any] = {
            "model": self._settings.model,
            "messages": list(messages),
        }
        if tools:
            request["tools"] = list(tools)

        response = self._client.chat.completions.create(**request)
        if not response.choices:
            raise RuntimeError("DeepSeek API 返回了空的 choices。")
        return response.choices[0].message
