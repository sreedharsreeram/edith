"""Claude adapter for the agent harness."""

import base64
import logging
import os

import anthropic
import httpx2

from ..agent.harness import REFUSAL_TEXT, BrainError, Exchange, OnText, Tool, ToolCall, ToolResult
from . import errors as messages

DEFAULT_MODEL = "claude-opus-5-5"

log = logging.getLogger("edith")


class Claude:
    def __init__(self, model: str | None = None, client: anthropic.AsyncAnthropic | None = None) -> None:
        self.name = model or os.environ.get("EDITH_MODEL", DEFAULT_MODEL)
        if client is None and not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            log.error("ANTHROPIC_API_KEY is not set: every question will fail. Get a key at console.anthropic.com")
        self.client = client or anthropic.AsyncAnthropic(
            timeout=30,
            max_retries=1,  # someone wearing glasses won't wait through a long retry chain
            # Keep idle connections alive long enough to survive while the wearer is talking (see warm()).
            http_client=anthropic.DefaultAsyncHttpxClient(limits=httpx2.Limits(keepalive_expiry=60)),
        )

    async def warm(self) -> None:
        """Open the TLS connection to the API now, so the real request skips the handshake."""
        try:
            await self.client.models.retrieve(self.name)
        except Exception:  # noqa: BLE001, S110 - purely an optimization; the real request surfaces real problems
            pass

    def start(
        self, *, system: str, history: list[Exchange], photo: bytes, question: str, tools: list[Tool]
    ) -> "ClaudeConversation":
        messages_: list[dict] = []
        for exchange in history:
            messages_.append({"role": "user", "content": exchange.question})
            messages_.append({"role": "assistant", "content": exchange.answer})
        messages_.append({"role": "user", "content": [_image(photo), {"type": "text", "text": question}]})
        specs = [{"name": t.name, "description": t.description, "input_schema": t.parameters} for t in tools]
        return ClaudeConversation(self, system, messages_, specs)


class ClaudeConversation:
    def __init__(self, claude: Claude, system: str, messages_: list[dict], tools: list[dict]) -> None:
        self.claude = claude
        self.system = system
        self.messages = messages_
        self.tools = tools
        self.model_name = claude.name

    async def step(self, on_text: OnText, allow_tools: bool) -> list[ToolCall]:
        extra = {"tools": self.tools, "tool_choice": {"type": "auto" if allow_tools else "none"}} if self.tools else {}
        try:
            async with self.claude.client.beta.messages.stream(
                model=self.claude.name,
                max_tokens=8000,
                output_config={"effort": "low"},  # low effort = faster replies, plenty for glasses-sized answers
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                system=self.system,
                messages=self.messages,
                **extra,
            ) as stream:
                async for text in stream.text_stream:
                    await on_text(text)
                final = await stream.get_final_message()
        except anthropic.AuthenticationError as e:
            raise BrainError(messages.INVALID_KEY) from e
        except anthropic.RateLimitError as e:
            raise BrainError(messages.BUSY) from e
        except anthropic.APITimeoutError as e:
            raise BrainError(messages.SLOW) from e
        except anthropic.APIConnectionError as e:
            raise BrainError(messages.OFFLINE) from e
        except anthropic.APIStatusError as e:
            raise BrainError(messages.SERVICE) from e

        self.model_name = final.model  # differs from the requested model if a server-side fallback answered
        # Appended unchanged: thinking blocks must go back exactly as they came.
        self.messages.append({"role": "assistant", "content": final.content})
        if final.stop_reason == "refusal":
            await on_text(REFUSAL_TEXT)
            return []
        return [ToolCall(id=b.id, name=b.name, args=dict(b.input)) for b in final.content if b.type == "tool_use"]

    def add_results(self, calls: list[ToolCall], results: list[ToolResult]) -> None:
        # Every result goes in one user message; splitting them teaches the model to stop calling tools in parallel.
        content = []
        for call, result in zip(calls, results, strict=True):
            blocks = [{"type": "text", "text": result.text or "(no output)"}, *map(_image, result.images)]
            content.append({"type": "tool_result", "tool_use_id": call.id, "content": blocks, "is_error": result.is_error})
        self.messages.append({"role": "user", "content": content})


def _image(jpeg: bytes) -> dict:
    data = base64.standard_b64encode(jpeg).decode("utf-8")
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}}
