"""Gemini adapter for the agent harness."""

import asyncio
import logging
import os
import time
import uuid

import httpx
from google import genai
from google.genai import errors, types

from ..agent.core import REFUSAL_TEXT, BrainError, Exchange, OnText, Tool, ToolCall, ToolResult
from . import errors as messages

# Newest regular Gemini models as of Oct 2026, newest first (deliberately not gemini-4-argon). When one is
# overloaded, the next one answers. Override with EDITH_GEMINI_MODELS=model-a,model-b.
DEFAULT_MODELS = ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash"]
RETRYABLE = {429, 500, 503, 504}  # busy or briefly broken: worth trying the next model
# A healthy model starts responding in ~2s; an overloaded one can hang for 30s. Past this, try the next model.
FIRST_OUTPUT_TIMEOUT_S = 6
# After a model is busy, skip it for a while so every question doesn't pay to rediscover that.
COOLDOWN_S = 60
BLOCKED = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "IMAGE_SAFETY", "IMAGE_PROHIBITED_CONTENT"}

log = logging.getLogger("edith")


class Gemini:
    def __init__(self, models: list[str] | None = None, client: genai.Client | None = None) -> None:
        env = os.environ.get("EDITH_GEMINI_MODELS")
        self.models = models or ([m.strip() for m in env.split(",") if m.strip()] if env else DEFAULT_MODELS)
        self.name = self.models[0]
        self.cooling_until: dict[str, float] = {}
        if client is None and not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
            log.error("GEMINI_API_KEY is not set: every question will fail. Get a key at aistudio.google.com")
        self.client = client or genai.Client(
            http_options=types.HttpOptions(
                timeout=30_000,  # milliseconds, per model call
                # The SDK would otherwise retry a busy model 5 times with backoff (~15s) before we could fall back.
                retry_options=types.HttpRetryOptions(attempts=1),
            )
        )

    async def warm(self) -> None:
        """Open the TLS connection to the API now, so the real request skips the handshake."""
        try:
            await self.client.aio.models.get(model=self.ordered_models()[0])
        except Exception:  # noqa: BLE001, S110 - purely an optimization; the real request surfaces real problems
            pass

    def start(
        self, *, system: str, history: list[Exchange], photo: bytes, question: str, tools: list[Tool]
    ) -> "GeminiConversation":
        contents = []
        for exchange in history:
            contents.append(types.Content(role="user", parts=[types.Part.from_text(text=exchange.question)]))
            contents.append(types.Content(role="model", parts=[types.Part.from_text(text=exchange.answer)]))
        contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_bytes(data=photo, mime_type="image/jpeg"), types.Part.from_text(text=question)],
            )
        )
        declarations = [
            types.FunctionDeclaration(name=t.name, description=t.description, parameters_json_schema=t.parameters)
            for t in tools
        ]
        return GeminiConversation(self, system, contents, declarations)

    def ordered_models(self) -> list[str]:
        """Models to try, newest first, with recently busy ones moved to the back (still a last resort)."""
        now = time.monotonic()
        ready = [m for m in self.models if self.cooling_until.get(m, 0) <= now]
        return ready + [m for m in self.models if m not in ready]

    def mark_busy(self, model: str) -> None:
        self.cooling_until[model] = time.monotonic() + COOLDOWN_S

    def mark_ok(self, model: str) -> None:
        self.cooling_until.pop(model, None)


class GeminiConversation:
    def __init__(self, gemini: Gemini, system: str, contents: list, declarations: list) -> None:
        self.gemini = gemini
        self.system = system
        self.contents = contents
        self.declarations = declarations
        self.pinned: str | None = None  # the model that answered step 1; the rest of the turn stays on it
        self.gemini_ids: dict[str, str | None] = {}  # our call id -> Gemini's (which may be absent)

    @property
    def model_name(self) -> str:
        return self.pinned or self.gemini.ordered_models()[0]

    def config(self, allow_tools: bool) -> types.GenerateContentConfig:
        tools = tool_config = None
        if self.declarations:
            tools = [types.Tool(function_declarations=self.declarations)]
            mode = types.FunctionCallingConfigMode.AUTO if allow_tools else types.FunctionCallingConfigMode.NONE
            tool_config = types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode=mode))
        return types.GenerateContentConfig(
            system_instruction=self.system,
            max_output_tokens=2048,
            tools=tools,
            tool_config=tool_config,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),  # the harness runs tools
        )

    async def step(self, on_text: OnText, allow_tools: bool) -> list[ToolCall]:
        # Only step 1 may switch models. Later steps carry this model's thought signatures, which another
        # model can't use, so mid-turn the turn stays on the model that started it.
        candidates = [self.pinned] if self.pinned else self.gemini.ordered_models()
        for i, model in enumerate(candidates):
            can_fall_back = i + 1 < len(candidates)
            parts: list[types.Part] = []
            blocked = False
            try:
                async with asyncio.timeout(FIRST_OUTPUT_TIMEOUT_S) as deadline:
                    stream = await self.gemini.client.aio.models.generate_content_stream(
                        model=model, contents=self.contents, config=self.config(allow_tools)
                    )
                    async for chunk in stream:
                        blocked = blocked or _is_blocked(chunk)
                        for part in _parts(chunk):
                            if not parts:
                                deadline.reschedule(None)  # it's responding; let it finish (the 30s cap still applies)
                            _append(parts, part)
                            if part.text and not part.thought:
                                await on_text(part.text)
            # Switch models only before anything reached the screen; never splice two answers together.
            except errors.APIError as e:
                if e.code in RETRYABLE:
                    self.gemini.mark_busy(model)
                    if not parts and can_fall_back:
                        log.warning("%s unavailable (%s), trying %s", model, e.code, candidates[i + 1])
                        continue
                raise BrainError(_message_for(e)) from e
            except TimeoutError as e:  # our first-output deadline: the model is hanging
                self.gemini.mark_busy(model)
                if can_fall_back:
                    log.warning("%s too slow to start, trying %s", model, candidates[i + 1])
                    continue
                raise BrainError(messages.SLOW) from e
            except httpx.TimeoutException as e:  # the 30s cap, mid-answer
                raise BrainError(messages.SLOW) from e
            except (httpx.TransportError, OSError) as e:  # DNS failures, refused or dropped connections
                raise BrainError(messages.OFFLINE) from e

            self.gemini.mark_ok(model)
            self.pinned = model
            break

        if parts:
            self.contents.append(types.Content(role="model", parts=parts))  # unchanged: keeps thought signatures
        elif blocked:
            await on_text(REFUSAL_TEXT)

        calls = []
        for part in parts:
            if part.function_call:
                call = ToolCall(
                    id=part.function_call.id or uuid.uuid4().hex[:8],
                    name=part.function_call.name,
                    args=dict(part.function_call.args or {}),
                )
                self.gemini_ids[call.id] = part.function_call.id
                calls.append(call)
        return calls

    def add_results(self, calls: list[ToolCall], results: list[ToolResult]) -> None:
        parts = []
        for call, result in zip(calls, results, strict=True):
            images = [types.FunctionResponsePart.from_bytes(data=img, mime_type="image/jpeg") for img in result.images]
            parts.append(
                types.Part(
                    function_response=types.FunctionResponse(
                        id=self.gemini_ids.get(call.id),
                        name=call.name,
                        response={"error": result.text} if result.is_error else {"result": result.text},
                        parts=images or None,
                    )
                )
            )
        self.contents.append(types.Content(role="user", parts=parts))


def _parts(chunk: types.GenerateContentResponse) -> list[types.Part]:
    if not chunk.candidates or not chunk.candidates[0].content:
        return []
    return chunk.candidates[0].content.parts or []


def _append(parts: list[types.Part], part: types.Part) -> None:
    """Merge streamed text pieces into one part, except where a thought signature must stay attached."""
    prev = parts[-1] if parts else None
    if prev is not None and _plain_text(prev) and _plain_text(part):
        parts[-1] = types.Part(text=prev.text + part.text)
    else:
        parts.append(part)


def _plain_text(part: types.Part) -> bool:
    return part.text is not None and not part.thought and not part.thought_signature and not part.function_call


def _is_blocked(chunk: types.GenerateContentResponse) -> bool:
    if chunk.prompt_feedback and chunk.prompt_feedback.block_reason:
        return True
    reason = chunk.candidates[0].finish_reason if chunk.candidates else None
    return reason is not None and getattr(reason, "name", str(reason)) in BLOCKED


def _message_for(e: errors.APIError) -> str:
    if e.code == 400 and "API_KEY" in str(e):  # Gemini answers a bad key with 400 API_KEY_INVALID
        return messages.INVALID_KEY
    if e.code == 429 and "quota" in str(e).lower():  # used up, not busy: waiting a moment won't help
        return messages.QUOTA
    return messages.for_status(e.code)
