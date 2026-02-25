# Copyright 2025 Horizon RL Contributors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SGLang native `/generate` API model provider for token-in/token-out training.

This provider uses SGLang's native HTTP APIs:
- `/generate` for text generation (returns output_ids directly)

It uses a HuggingFace tokenizer for:
- Applying chat templates (via tokenizer.apply_chat_template())
- Tokenizing prompts and tool results

For VLM (Vision Language Models), an optional HuggingFace processor replaces the
tokenizer for prompt encoding. The processor dynamically expands image placeholder
tokens based on actual image dimensions and produces multimodal training tensors
(pixel_values, image_grid_thw, etc.). Images are sent to SGLang as base64 data URLs
via the ``image_data`` field in the ``/generate`` payload.

This eliminates retokenization drift in RL training by maintaining token IDs
throughout the rollout instead of converting text back to tokens.
"""

from __future__ import annotations

import json
import logging
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    AsyncIterable,
    Iterator,
    Type,
    TypedDict,
    TypeVar,
    cast,
)

from pydantic import BaseModel
from strands.models import Model
from strands.models.openai import OpenAIModel
from strands.types.content import Messages, SystemContentBlock
from strands.types.exceptions import (
    ContextWindowOverflowException,
    ModelThrottledException,
)
from strands.types.streaming import StreamEvent
from strands.types.tools import ToolChoice, ToolSpec
from typing_extensions import Unpack, override

from .client import SGLangClient
from .exceptions import SGLangContextLengthError, SGLangThrottledError
from .token import TokenManager
from .tool_parsers import HermesToolParser, ToolParser, ToolParseResult

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase
    from transformers.processing_utils import ProcessorMixin

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class SGLangModel(Model):
    """SGLang native `/generate` API provider with token-in/token-out support.

    Example:
        >>> from transformers import AutoTokenizer
        >>> from strands_sglang import SGLangClient, SGLangModel
        >>> client = SGLangClient(base_url="http://localhost:30000")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
        >>> model = SGLangModel(client=client, tokenizer=tokenizer)
        >>> # After generation:
        >>> model.token_manager.token_ids    # Full token trajectory
        >>> model.token_manager.loss_mask    # Boolean mask for loss computation
        >>> model.token_manager.logprobs     # Log probabilities
    """

    class SGLangConfig(TypedDict, total=False):
        """Configuration options for SGLang generation."""

        sampling_params: dict[str, Any] | None  # Passed to /generate endpoint
        return_logprob: bool | None  # Return logprobs for all tokens (default: True)
        enable_thinking: bool | None  # Enable thinking mode for Qwen3 hybrid models

    def __init__(
        self,
        *,
        client: SGLangClient,
        tokenizer: PreTrainedTokenizerBase,
        processor: ProcessorMixin | None = None,
        tool_parser: ToolParser | None = None,
        **config: Unpack[SGLangConfig],
    ) -> None:
        """Initialize SGLang model provider.

        Args:
            client: `SGLangClient` for HTTP communication with the SGLang server.
            tokenizer: HuggingFace tokenizer for chat template and tokenization.
            processor: Optional HuggingFace processor for VLM models (e.g., Qwen3-VL).
                When provided, the processor is used instead of the tokenizer for prompt
                encoding, producing input_ids with dynamically expanded image tokens.
                The processor and SGLang server must load the same model to ensure
                identical image preprocessing (same resize, patch size, token counts).
            tool_parser: `ToolParser` for tool calls (default: `HermesToolParser`).
            **config: Additional SGLang generation configuration.
        """
        self.client = client
        self.tokenizer = tokenizer
        self.processor = processor
        self.tool_parser = tool_parser or HermesToolParser()
        self.config = dict(config)

        # State tracking (this makes SGLangModel stateful)
        self.token_manager = TokenManager()
        self._processed_message_count: int = 0
        self._current_tools: list[dict] | None = None
        self.tool_parse_errors: dict[str, int] = {}  # per-tool parse error count

        # VLM state (reset per episode)
        self._image_data: list[str] = []  # accumulated base64 data URLs for SGLang payload
        self._multimodal_train_inputs: list[dict] = []  # per-turn processor outputs for RL training

        logger.debug(f"initialized with config: {self.config}, vlm={'enabled' if processor else 'disabled'}")

    def reset(self) -> None:
        """Reset token accumulation for a new episode.

        Call this at episode start. Clears all accumulated tokens and resets
        internal state for tool tracking and VLM image accumulation.
        """
        self.token_manager.reset()
        self._processed_message_count = 0
        self._current_tools = None
        self.tool_parse_errors = {}
        self._image_data = []
        self._multimodal_train_inputs = []

    # -------------------------------------------------------------------------
    # Model interface implementation
    # -------------------------------------------------------------------------

    @override
    def update_config(self, **model_config: Unpack[SGLangConfig]) -> None:  # type: ignore[override]
        """Update the model configuration.

        Args:
            **model_config: Configuration overrides.
        """
        self.config.update(model_config)

    @override
    def get_config(self) -> SGLangConfig:
        """Get the model configuration.

        Returns:
            The model configuration dict.
        """
        return cast(SGLangModel.SGLangConfig, self.config)

    # -------------------------------------------------------------------------
    # Chat template and message formatting
    # -------------------------------------------------------------------------

    @classmethod
    def _format_message_content(cls, message: dict[str, Any]) -> None:
        """Format a single message's content for chat templates.

        For text-only messages, flattens content arrays to a plain string.
        For VLM messages with image/video blocks, preserves structured content
        as a list of ``{"type": "text", ...}`` and ``{"type": "image", ...}`` dicts
        so the HuggingFace processor can handle multimodal inputs.

        Modifies the message in-place.
        """
        if "content" in message and isinstance(message["content"], list):
            # Check if any block is multimodal (image/video)
            has_media = any(isinstance(b, dict) and b.get("type") in ("image", "video") for b in message["content"])
            if has_media:
                # Preserve structured content for VLM processor
                normalized: list[dict[str, Any]] = []
                for block in message["content"]:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") in ("image", "video"):
                        normalized.append(block)
                    elif "text" in block:
                        normalized.append({"type": "text", "text": block["text"]})
                message["content"] = normalized
            else:
                # Text-only: flatten to string (original behavior)
                text_content = ""
                for block in message["content"]:
                    if "text" in block:
                        text_content = block["text"]
                        break
                message["content"] = text_content

        # Remove strands-processed tool_calls field and let the chat template handle it.
        if "tool_calls" in message:
            del message["tool_calls"]

    @classmethod
    def format_request_messages(cls, messages: Messages, system_prompt: str | None = None) -> list[dict[str, Any]]:
        """Convert strands Messages to OpenAI format for chat templates.

        Uses strands' OpenAIModel formatter and flattens content
        for compatibility with HuggingFace apply_chat_template.
        """
        result = OpenAIModel.format_request_messages(messages=messages, system_prompt=system_prompt)

        for message in result:
            cls._format_message_content(message)

        return result

    def _format_tools(self, tool_specs: list[ToolSpec]) -> list[dict]:
        """Format strands ToolSpecs to OpenAI format for chat templates."""
        return [
            {
                "type": "function",
                "function": {
                    "name": spec["name"],
                    "description": spec["description"],
                    "parameters": spec["inputSchema"]["json"],
                },
            }
            for spec in tool_specs
        ]

    def format_prompt(
        self,
        messages: Messages,
        system_prompt: str | None = None,
        tools: list[dict] | None = None,
    ) -> str:
        """Format messages into a prompt ready for model generation.

        Applies the HuggingFace chat template with `add_generation_prompt=True`,
        which appends the assistant turn prefix for the model to continue.

        The result is manually tokenized (not model-generated) and added to
        the token trajectory with `loss_mask=False`.
        """
        chat_messages = self.format_request_messages(messages, system_prompt)
        return self.tokenizer.apply_chat_template(
            conversation=chat_messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=self.config.get("enable_thinking"),
        )

    # -------------------------------------------------------------------------
    # VLM (Vision Language Model) support
    # -------------------------------------------------------------------------

    @staticmethod
    def _extract_images_from_messages(messages: list[dict]) -> list[str]:
        """Extract base64 data URL strings from structured message content blocks.

        Returns:
            List of base64 data URL strings found in messages.
        """
        images: list[str] = []
        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if isinstance(block, dict) and block.get("type") == "image":
                    img = block.get("image")
                    if isinstance(img, str):
                        images.append(img)
        return images

    @staticmethod
    def _data_url_to_pil_image(data_url: str) -> Any:
        """Decode a base64 data URL to a PIL Image for the HF processor.

        Args:
            data_url: Base64 data URL string (e.g., ``"data:image/png;base64,iVBOR..."``).

        Returns:
            PIL Image, or None if decoding fails.
        """
        try:
            import base64
            import io

            from PIL import Image

            _, encoded = data_url.split(",", 1)
            image_bytes = base64.b64decode(encoded)
            return Image.open(io.BytesIO(image_bytes))
        except Exception:
            return None

    @classmethod
    def _format_vlm_messages(cls, messages: Messages, system_prompt: str | None = None) -> list[dict[str, Any]]:
        """Format messages for VLM, preserving image content blocks as base64 data URLs.

        Images are expected as plain base64 data URL strings throughout:

        1. Direct: ``{"type": "image", "image": "data:image/png;base64,..."}`` — passed through.
        2. Inline: ``{"image": "data:image/png;base64,..."}`` — normalized to type/image block.
        3. Tool results: ``{"toolResult": {"content": [{"image": "data:..."}]}}`` — extracted.
        """
        result: list[dict[str, Any]] = []
        if system_prompt:
            result.append({"role": "system", "content": system_prompt})

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content")

            if isinstance(content, str):
                result.append({"role": role, "content": content})
            elif isinstance(content, list):
                normalized = cls._normalize_vlm_content_blocks(content)
                formatted_msg: dict[str, Any] = {"role": role}
                has_media = any(isinstance(b, dict) and b.get("type") in ("image", "video") for b in normalized)
                formatted_msg["content"] = normalized if has_media else cls._flatten_text(normalized)
                result.append(formatted_msg)
            else:
                result.append({"role": role, "content": str(content) if content else ""})

        return result

    @classmethod
    def _normalize_vlm_content_blocks(cls, content: list) -> list[dict[str, Any]]:
        """Normalize a list of content blocks to VLM-compatible format.

        Images are expected as plain base64 data URL strings. Handles:
        - Direct typed blocks: ``{"type": "image", "image": "data:..."}``
        - Inline image blocks: ``{"image": "data:..."}``
        - Tool results with nested content: ``{"toolResult": {"content": [...]}}``
        - Plain text blocks: ``{"text": "..."}``
        """
        normalized: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                continue

            # Direct image/video block: {"type": "image", "image": "data:..."}
            if block.get("type") in ("image", "video"):
                normalized.append(block)

            # Inline image block: {"image": "data:image/...;base64,..."}
            elif "image" in block and isinstance(block["image"], str):
                normalized.append({"type": "image", "image": block["image"]})

            # Tool result with nested content: {"toolResult": {"content": [...]}}
            elif "toolResult" in block:
                tool_result = block["toolResult"]
                tr_content = tool_result.get("content", [])
                if isinstance(tr_content, list):
                    for tr_block in tr_content:
                        if not isinstance(tr_block, dict):
                            continue
                        if "text" in tr_block:
                            normalized.append({"type": "text", "text": tr_block["text"]})
                        elif "image" in tr_block and isinstance(tr_block["image"], str):
                            normalized.append({"type": "image", "image": tr_block["image"]})

            # Plain text
            elif "text" in block:
                normalized.append({"type": "text", "text": block["text"]})

        return normalized

    @staticmethod
    def _flatten_text(blocks: list[dict[str, Any]]) -> str:
        """Flatten text-only content blocks to a single string."""
        for block in blocks:
            if isinstance(block, dict) and "text" in block:
                return block["text"]
        return ""

    def _format_vlm_prompt(
        self,
        messages: Messages,
        system_prompt: str | None = None,
        tools: list[dict] | None = None,
    ) -> tuple[str, list[str]]:
        """Format messages into a prompt for VLM generation and extract images.

        Returns:
            Tuple of (formatted_prompt_text, list_of_base64_data_url_strings).
        """
        chat_messages = self._format_vlm_messages(messages, system_prompt)
        images = self._extract_images_from_messages(chat_messages)

        formatted_text = self.tokenizer.apply_chat_template(
            conversation=chat_messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=self.config.get("enable_thinking"),
        )
        return formatted_text, images

    def _run_processor(self, formatted_text: str, images: list[str]) -> list[int]:
        """Run the HuggingFace processor on formatted text and base64 data URL images.

        Decodes base64 data URLs to PIL for the processor, and accumulates the
        original data URL strings for the SGLang ``/generate`` payload.

        Returns:
            List of token IDs (with dynamically expanded image placeholder tokens).
        """
        processor_kwargs: dict[str, Any] = {"return_tensors": None}
        if images:
            # Decode base64 data URLs to PIL for the HF processor
            pil_images = [self._data_url_to_pil_image(img) for img in images]
            pil_images = [img for img in pil_images if img is not None]
            if pil_images:
                processor_kwargs["images"] = pil_images
                processor_kwargs["images_kwargs"] = {"return_tensors": "pt"}

        processor_output = self.processor(text=formatted_text, **processor_kwargs)
        input_ids = processor_output["input_ids"]
        # Unbatch if needed (processor may return [[tokens]])
        if isinstance(input_ids, list) and input_ids and isinstance(input_ids[0], list):
            input_ids = input_ids[0]

        # Store multimodal training tensors (pixel_values, image_grid_thw, etc.)
        mm_train = {k: v for k, v in processor_output.items() if k not in ("input_ids", "attention_mask")}
        if mm_train:
            self._multimodal_train_inputs.append(mm_train)

        # Accumulate base64 data URLs directly for SGLang HTTP payload (no re-encoding)
        self._image_data.extend(images)

        return list(input_ids)

    def _process_vlm_prompt(
        self,
        messages: Messages,
        system_prompt: str | None,
    ) -> list[int]:
        """Process initial VLM prompt through the processor.

        Uses the processor (instead of tokenizer.encode) to produce input_ids with
        dynamically expanded image placeholder tokens, and extracts multimodal training
        tensors (pixel_values, image_grid_thw, etc.).
        """
        formatted_text, images = self._format_vlm_prompt(messages, system_prompt, tools=self._current_tools)
        return self._run_processor(formatted_text, images)

    def _process_vlm_incremental(
        self,
        new_messages: Messages,
    ) -> list[int]:
        """Process incremental VLM prompt (new messages in a multi-turn agent loop).

        Only processes new messages and their images, appending to accumulated state.
        """
        formatted_text, images = self._format_vlm_prompt(new_messages)
        formatted_text = self.tool_parser.message_separator + formatted_text
        return self._run_processor(formatted_text, images)

    @staticmethod
    def _merge_multimodal_train_inputs(chunks: list[dict]) -> dict | None:
        """Merge per-turn multimodal training tensors by concatenating along dim 0.

        Follows SLIME's ``_merge_multimodal_train_inputs`` pattern from the
        geo3k_vlm_multi_turn example. Only torch.Tensor values are merged.
        """
        if not chunks:
            return None

        try:
            import torch
        except ImportError:
            logger.warning("torch not available; cannot merge multimodal_train_inputs")
            return None

        values_by_key: dict[str, list] = {}
        for chunk in chunks:
            if not chunk:
                continue
            for key, val in chunk.items():
                if val is not None:
                    values_by_key.setdefault(key, []).append(val)

        merged = {}
        for key, values in values_by_key.items():
            if all(isinstance(v, torch.Tensor) for v in values):
                merged[key] = torch.cat(values, dim=0)

        return merged or None

    @property
    def image_data(self) -> list[str]:
        """Accumulated base64 image data URLs for the current episode."""
        return self._image_data

    @property
    def multimodal_train_inputs(self) -> dict | None:
        """Merged multimodal processor outputs for RL training (pixel_values, etc.)."""
        return self._merge_multimodal_train_inputs(self._multimodal_train_inputs)

    # -------------------------------------------------------------------------
    # Generation
    # -------------------------------------------------------------------------

    def tokenize_prompt_messages(
        self,
        messages: Messages,
        system_prompt: str | None,
    ) -> list[int] | None:
        """Tokenize prompt messages for the next generation call.

        First call: tokenizes full prompt with system prompt and tools.
        Subsequent calls: tokenizes only new messages (tool results, user messages),
        prepending the message separator to align with chat template formatting.

        When a processor is set (VLM mode), uses the processor instead of
        tokenizer.encode to produce input_ids with expanded image tokens.
        """
        # First call: full prompt with tools
        if len(self.token_manager) == 0:
            if self.processor:
                return self._process_vlm_prompt(messages, system_prompt)
            formatted = self.format_prompt(messages, system_prompt, tools=self._current_tools)
            return self.tokenizer.encode(formatted, add_special_tokens=False)

        # Subsequent calls: only new messages
        if len(messages) > self._processed_message_count:
            new_messages = self._sort_tool_results(messages[self._processed_message_count :])
            if self.processor:
                return self._process_vlm_incremental(new_messages)
            formatted = self.tool_parser.message_separator + self.format_prompt(new_messages)
            return self.tokenizer.encode(formatted, add_special_tokens=False)

        return None

    def _sort_tool_results(self, messages: Messages) -> Messages:
        """Sort tool results by ID to match original call order (IDs are sequential: call_0000, call_0001, ...)."""
        result = []
        for msg in messages:
            if msg.get("role") != "user" or not isinstance(msg.get("content"), list):
                result.append(msg)
                continue
            content = msg["content"]
            tool_results = [b for b in content if isinstance(b, dict) and "toolResult" in b]
            if not tool_results:
                result.append(msg)
                continue
            other = [b for b in content if not (isinstance(b, dict) and "toolResult" in b)]
            tool_results.sort(key=lambda b: b.get("toolResult", {}).get("toolUseId", ""))
            result.append({**msg, "content": other + tool_results})
        return result

    def _yield_tool_use_events(
        self,
        tool_calls: list[ToolParseResult],
    ) -> Iterator[StreamEvent]:
        """Yield toolUse stream events for parsed tool calls.

        Each tool call emits three events following the Strands streaming protocol:
        - `contentBlockStart`: begins block with toolUseId and name
        - `contentBlockDelta`: contains the tool input (delta = incremental data)
        - `contentBlockStop`: ends the block
        """
        for tool_call in tool_calls:
            if tool_call.is_error:
                logger.warning(f"Tool parse error for '{tool_call.name}': {(tool_call.raw or '')[:100]}")
                # Track parse error count per tool name
                self.tool_parse_errors[tool_call.name] = self.tool_parse_errors.get(tool_call.name, 0) + 1

            yield {
                "contentBlockStart": {
                    "start": {
                        "toolUse": {
                            "toolUseId": tool_call.id,
                            "name": tool_call.name,
                        }
                    }
                }
            }
            yield {
                "contentBlockDelta": {
                    "delta": {
                        "toolUse": {
                            "input": tool_call.payload,
                        }
                    }
                }
            }
            yield {"contentBlockStop": {}}

    def _extract_logprobs(self, event: dict[str, Any], key: str) -> list[float] | None:
        """Extract logprobs from SGLang event (format: [[logprob, token_id, ...], ...])."""
        meta_info = event.get("meta_info", {})
        logprobs = meta_info.get(key) or event.get(key)
        if isinstance(logprobs, list) and logprobs:
            return [entry[0] for entry in logprobs]
        return None

    @override
    async def stream(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        *,
        tool_choice: ToolChoice | None = None,
        system_prompt_content: list[SystemContentBlock] | None = None,
        **kwargs: Any,
    ) -> AsyncIterable[StreamEvent]:
        """Chat completion with SGLangModel using the `/generate` endpoint.

        The `stream` method follows Strands' protocol but actually disabled here for training-only usage.
        This means users won't see streaming behavior such as print callbacks.
        """
        # Format tools (only on first call)
        if tool_specs and not self._current_tools:
            self._current_tools = self._format_tools(tool_specs)
            logger.debug(f"tools formatted: {len(self._current_tools)} tools")

        # Prepare request
        config = self.get_config()
        sampling_params: dict[str, Any] = dict(config.get("sampling_params") or {})
        return_logprob = config.get("return_logprob", True)
        new_input_tokens = self.tokenize_prompt_messages(messages, system_prompt)
        # Tracking token IDs in token_manager to ensure the token-in feature
        input_ids = self.token_manager.token_ids + (new_input_tokens or [])

        # Start message
        yield {"messageStart": {"role": "assistant"}}
        yield {"contentBlockStart": {"start": {}}}

        # Call SGLangClient (non-streaming POST for better parallelism)
        try:
            generate_kwargs: dict[str, Any] = {
                "sampling_params": sampling_params,
                "return_logprob": return_logprob,
                "logprob_start_len": len(self.token_manager) if return_logprob else None,
            }
            if self._image_data:
                generate_kwargs["image_data"] = self._image_data
            response = await self.client.generate(
                input_ids=input_ids,
                **generate_kwargs,
            )

            # Extract response data
            text = response.get("text", "")
            output_ids = response.get("output_ids", [])
            output_logprobs = self._extract_logprobs(response, "output_token_logprobs")
            input_logprobs = self._extract_logprobs(response, "input_token_logprobs")
            meta_info = response.get("meta_info", {})

            # Yield text as single delta (non-streaming gives complete text at once)
            if text:
                yield {"contentBlockDelta": {"delta": {"text": text}}}

        except SGLangContextLengthError as e:
            raise ContextWindowOverflowException(f"Context length exceeded: {e.body}") from e
        except SGLangThrottledError as e:
            raise ModelThrottledException(f"Service throttled (status={e.status}): {e.body}") from e

        # Update token trajectory
        if new_input_tokens:
            new_input_logprobs = input_logprobs[-len(new_input_tokens) :] if input_logprobs else None
            self.token_manager.add_prompt(token_ids=new_input_tokens, logprobs=new_input_logprobs)
        if output_ids:
            self.token_manager.add_response(token_ids=output_ids, logprobs=output_logprobs)
        self._processed_message_count = len(messages) + 1

        # End text block, start tool use blocks if there are any tool calls
        yield {"contentBlockStop": {}}

        # Parse tool calls and yield events
        parsed_tool_calls = self.tool_parser.parse(text)
        for event in self._yield_tool_use_events(parsed_tool_calls):
            yield event

        # Determine stop reason
        stop_reason: str = "tool_use" if parsed_tool_calls else "end_turn"
        if meta_info and isinstance(meta_info.get("finish_reason"), dict):
            if meta_info["finish_reason"].get("type") == "length":
                stop_reason = "max_tokens"

        yield {"messageStop": {"stopReason": stop_reason}}

        # Yield usage metadata
        if meta_info:
            prompt_tokens = int(meta_info.get("prompt_tokens") or 0)
            completion_tokens = int(meta_info.get("completion_tokens") or 0)
            yield {
                "metadata": {
                    "usage": {
                        "inputTokens": prompt_tokens,
                        "outputTokens": completion_tokens,
                        "totalTokens": prompt_tokens + completion_tokens,
                    },
                    "metrics": {"latencyMs": int(float(meta_info.get("e2e_latency") or 0) * 1000)},
                }
            }

    @override
    async def structured_output(
        self,
        output_model: Type[T],
        prompt: Messages,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[dict[str, T | Any], None]:
        """Get structured output using SGLang's constrained decoding.

        Uses SGLang's `json_schema` parameter for FSM-based constrained generation,
        guaranteeing output conforms to the Pydantic model schema.

        Note: This method does NOT update token_manager (no TITO tracking).
        Intended for inference-only use cases like LLM-as-Judge.

        Args:
            output_model: Pydantic model class defining the output schema.
            prompt: Messages to send to the model.
            system_prompt: Optional system prompt.
            **kwargs: Additional arguments (unused).

        Yields:
            Single dict with "output" key containing the parsed Pydantic model instance.

        Raises:
            ValidationError: If model output fails Pydantic validation.
            SGLangHTTPError: On non-retryable HTTP errors.
        """
        # Convert Pydantic model to JSON schema string
        json_schema = json.dumps(output_model.model_json_schema())

        # Format and tokenize prompt (no tools for structured output)
        formatted = self.format_prompt(prompt, system_prompt, tools=None)
        input_ids = self.tokenizer.encode(formatted, add_special_tokens=False)

        # Build sampling params with json_schema constraint
        config = self.get_config()
        sampling_params: dict[str, Any] = dict(config.get("sampling_params") or {})
        sampling_params["json_schema"] = json_schema

        # Call SGLang /generate endpoint
        try:
            response = await self.client.generate(
                input_ids=input_ids,
                sampling_params=sampling_params,
                return_logprob=False,  # No need for logprobs in structured output
            )
        except SGLangContextLengthError as e:
            raise ContextWindowOverflowException(f"Context length exceeded: {e.body}") from e
        except SGLangThrottledError as e:
            raise ModelThrottledException(f"Service throttled (status={e.status}): {e.body}") from e

        # Parse and validate response
        text = response.get("text", "")
        parsed = output_model.model_validate_json(text)

        yield {"output": parsed}
