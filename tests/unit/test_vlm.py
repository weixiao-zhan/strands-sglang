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

"""Unit tests for VLM (Vision Language Model) support in SGLangModel."""

import base64

from unittest.mock import MagicMock, patch

import pytest

from strands_sglang import SGLangModel
from strands_sglang.client import SGLangClient

# A minimal valid 1x1 PNG for testing
_TINY_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
    b"\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00"
    b"\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
)
SAMPLE_DATA_URL = f"data:image/png;base64,{base64.b64encode(_TINY_PNG).decode()}"
SAMPLE_DATA_URL_2 = f"data:image/jpeg;base64,{base64.b64encode(b'fake_jpeg').decode()}"


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    tokenizer.decode.return_value = "decoded text"
    tokenizer.apply_chat_template.return_value = "formatted prompt"
    return tokenizer


@pytest.fixture
def mock_processor():
    """Create a mock VLM processor."""
    processor = MagicMock()
    # Simulate processor returning input_ids with image pad tokens + pixel_values
    processor.return_value = {
        "input_ids": [[10, 20, 151652, 151655, 151655, 151655, 151653, 30, 40]],  # with vision tokens
        "attention_mask": [[1, 1, 1, 1, 1, 1, 1, 1, 1]],
        "pixel_values": MagicMock(),  # tensor-like
        "image_grid_thw": MagicMock(),  # tensor-like
    }
    return processor


@pytest.fixture
def client():
    return SGLangClient(base_url="http://localhost:30000")


@pytest.fixture
def text_model(client, mock_tokenizer):
    """Text-only SGLangModel (no processor)."""
    return SGLangModel(client=client, tokenizer=mock_tokenizer)


@pytest.fixture
def vlm_model(client, mock_tokenizer, mock_processor):
    """VLM-enabled SGLangModel with processor."""
    return SGLangModel(client=client, tokenizer=mock_tokenizer, processor=mock_processor)


class TestFormatMessageContentVLM:
    """Tests for _format_message_content with image content blocks."""

    def test_text_only_flattens_to_string(self):
        """Text-only content is flattened to a string (backwards compat)."""
        msg = {"role": "user", "content": [{"text": "Hello world"}]}
        SGLangModel._format_message_content(msg)
        assert msg["content"] == "Hello world"

    def test_image_blocks_preserved(self):
        """Content with image blocks is preserved as structured list."""
        msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this:"},
                {"type": "image", "image": SAMPLE_DATA_URL},
            ],
        }
        SGLangModel._format_message_content(msg)
        assert isinstance(msg["content"], list)
        assert len(msg["content"]) == 2
        assert msg["content"][0] == {"type": "text", "text": "Describe this:"}
        assert msg["content"][1]["type"] == "image"
        assert msg["content"][1]["image"] == SAMPLE_DATA_URL

    def test_mixed_blocks_filters_non_media(self):
        """Non-text/image blocks (like toolUse) are filtered out."""
        msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Look at this"},
                {"type": "image", "image": SAMPLE_DATA_URL},
                {"toolResult": {"toolUseId": "call_0000", "content": [{"text": "result"}]}},
            ],
        }
        SGLangModel._format_message_content(msg)
        assert isinstance(msg["content"], list)
        assert len(msg["content"]) == 2

    def test_text_block_without_type_normalized(self):
        """Text blocks without explicit type get normalized."""
        msg = {
            "role": "user",
            "content": [
                {"text": "Describe"},
                {"type": "image", "image": SAMPLE_DATA_URL},
            ],
        }
        SGLangModel._format_message_content(msg)
        assert msg["content"][0] == {"type": "text", "text": "Describe"}

    def test_video_blocks_preserved(self):
        """Video blocks are also preserved."""
        video = MagicMock()
        msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Watch this"},
                {"type": "video", "video": video},
            ],
        }
        SGLangModel._format_message_content(msg)
        assert isinstance(msg["content"], list)
        assert msg["content"][1]["type"] == "video"


class TestExtractImagesFromMessages:
    """Tests for _extract_images_from_messages."""

    def test_extract_single_image(self):
        messages = [{"role": "user", "content": [{"type": "image", "image": SAMPLE_DATA_URL}]}]
        result = SGLangModel._extract_images_from_messages(messages)
        assert result == [SAMPLE_DATA_URL]

    def test_extract_multiple_images_across_messages(self):
        messages = [
            {"role": "user", "content": [{"type": "image", "image": SAMPLE_DATA_URL}]},
            {"role": "user", "content": [{"type": "text", "text": "and"}, {"type": "image", "image": SAMPLE_DATA_URL_2}]},
        ]
        result = SGLangModel._extract_images_from_messages(messages)
        assert result == [SAMPLE_DATA_URL, SAMPLE_DATA_URL_2]

    def test_no_images(self):
        messages = [{"role": "user", "content": "just text"}]
        result = SGLangModel._extract_images_from_messages(messages)
        assert result == []

    def test_string_content_skipped(self):
        messages = [{"role": "user", "content": "plain string"}]
        result = SGLangModel._extract_images_from_messages(messages)
        assert result == []

    def test_non_string_image_skipped(self):
        """Non-string image values (e.g., leftover PIL objects) are ignored."""
        messages = [{"role": "user", "content": [{"type": "image", "image": MagicMock()}]}]
        result = SGLangModel._extract_images_from_messages(messages)
        assert result == []


class TestDataUrlToPilImage:
    """Tests for _data_url_to_pil_image."""

    def test_decodes_valid_png(self):
        result = SGLangModel._data_url_to_pil_image(SAMPLE_DATA_URL)
        assert result is not None
        assert result.size == (1, 1)

    def test_returns_none_for_invalid(self):
        result = SGLangModel._data_url_to_pil_image("not-a-data-url")
        assert result is None

    def test_returns_none_for_corrupt_base64(self):
        result = SGLangModel._data_url_to_pil_image("data:image/png;base64,!!!invalid!!!")
        assert result is None


class TestNormalizeVLMContentBlocks:
    """Tests for _normalize_vlm_content_blocks with base64 data URL convention."""

    def test_direct_typed_image_block(self):
        blocks = [{"type": "image", "image": SAMPLE_DATA_URL}]
        result = SGLangModel._normalize_vlm_content_blocks(blocks)
        assert result == [{"type": "image", "image": SAMPLE_DATA_URL}]

    def test_inline_image_block(self):
        """Inline {"image": "data:..."} is normalized to typed block."""
        blocks = [{"image": SAMPLE_DATA_URL}]
        result = SGLangModel._normalize_vlm_content_blocks(blocks)
        assert result == [{"type": "image", "image": SAMPLE_DATA_URL}]

    def test_tool_result_with_image(self):
        blocks = [
            {
                "toolResult": {
                    "toolUseId": "call_0000",
                    "status": "success",
                    "content": [
                        {"text": "Image loaded"},
                        {"image": SAMPLE_DATA_URL},
                    ],
                }
            }
        ]
        result = SGLangModel._normalize_vlm_content_blocks(blocks)
        assert len(result) == 2
        assert result[0] == {"type": "text", "text": "Image loaded"}
        assert result[1] == {"type": "image", "image": SAMPLE_DATA_URL}

    def test_plain_text_block(self):
        blocks = [{"text": "hello"}]
        result = SGLangModel._normalize_vlm_content_blocks(blocks)
        assert result == [{"type": "text", "text": "hello"}]

    def test_non_dict_blocks_skipped(self):
        blocks = ["not a dict", 42, None, {"text": "valid"}]
        result = SGLangModel._normalize_vlm_content_blocks(blocks)
        assert result == [{"type": "text", "text": "valid"}]


class TestVLMConstructorAndReset:
    """Tests for VLM state initialization and reset."""

    def test_processor_stored(self, vlm_model, mock_processor):
        assert vlm_model.processor is mock_processor

    def test_text_model_no_processor(self, text_model):
        assert text_model.processor is None

    def test_vlm_state_initialized_empty(self, vlm_model):
        assert vlm_model._image_data == []
        assert vlm_model._multimodal_train_inputs == []

    def test_reset_clears_vlm_state(self, vlm_model):
        vlm_model._image_data = ["data:image/png;base64,abc"]
        vlm_model._multimodal_train_inputs = [{"pixel_values": "tensor"}]

        vlm_model.reset()

        assert vlm_model._image_data == []
        assert vlm_model._multimodal_train_inputs == []

    def test_reset_clears_all_state(self, vlm_model):
        """Reset clears both text and VLM state."""
        vlm_model.token_manager.add_prompt([1, 2, 3])
        vlm_model._processed_message_count = 5
        vlm_model._current_tools = [{"type": "function"}]
        vlm_model._image_data = ["img"]

        vlm_model.reset()

        assert len(vlm_model.token_manager) == 0
        assert vlm_model._processed_message_count == 0
        assert vlm_model._current_tools is None
        assert vlm_model._image_data == []


class TestTokenizePromptMessagesVLM:
    """Tests for tokenize_prompt_messages with VLM processor."""

    def test_first_call_uses_processor(self, vlm_model, mock_processor, mock_tokenizer):
        """First call with processor uses processor instead of tokenizer.encode."""
        messages = [{"role": "user", "content": [{"text": "Hello"}]}]

        result = vlm_model.tokenize_prompt_messages(messages, system_prompt=None)

        # Processor was called
        mock_processor.assert_called_once()
        # Tokenizer.encode was NOT called
        mock_tokenizer.encode.assert_not_called()
        # Result is the processor's input_ids (unbatched)
        assert result == [10, 20, 151652, 151655, 151655, 151655, 151653, 30, 40]

    def test_first_call_without_processor_uses_tokenizer(self, text_model, mock_tokenizer):
        """First call without processor uses tokenizer.encode (backwards compat)."""
        messages = [{"role": "user", "content": [{"text": "Hello"}]}]

        result = text_model.tokenize_prompt_messages(messages, system_prompt=None)

        mock_tokenizer.encode.assert_called_once()
        assert result == [1, 2, 3, 4, 5]

    def test_subsequent_call_uses_processor(self, vlm_model, mock_processor, mock_tokenizer):
        """Subsequent calls with processor use _process_vlm_incremental."""
        vlm_model.token_manager.add_prompt([1, 2, 3])
        vlm_model._processed_message_count = 1

        messages = [
            {"role": "user", "content": [{"text": "Hello"}]},
            {"role": "user", "content": [{"text": "New message"}]},
        ]

        result = vlm_model.tokenize_prompt_messages(messages, system_prompt=None)

        mock_processor.assert_called_once()
        mock_tokenizer.encode.assert_not_called()
        assert result is not None


class TestImageAccumulation:
    """Tests for multi-turn image accumulation."""

    def test_first_turn_populates_image_data(self, vlm_model, mock_processor):
        """First turn with images populates _image_data with base64 data URLs."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this:"},
                    {"type": "image", "image": SAMPLE_DATA_URL},
                ],
            }
        ]

        vlm_model.tokenize_prompt_messages(messages, system_prompt=None)

        assert len(vlm_model._image_data) == 1
        assert vlm_model._image_data[0] == SAMPLE_DATA_URL

    def test_multi_turn_accumulates_images(self, vlm_model, mock_processor):
        """Multiple turns accumulate images in _image_data."""
        # First turn
        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": SAMPLE_DATA_URL}, {"type": "text", "text": "first"}],
            }
        ]
        vlm_model.tokenize_prompt_messages(messages, system_prompt=None)
        assert len(vlm_model._image_data) == 1

        # Simulate second turn
        vlm_model.token_manager.add_prompt([10, 20, 30])
        vlm_model._processed_message_count = 1
        messages.append(
            {
                "role": "user",
                "content": [{"type": "image", "image": SAMPLE_DATA_URL_2}, {"type": "text", "text": "second"}],
            }
        )
        vlm_model.tokenize_prompt_messages(messages, system_prompt=None)

        assert len(vlm_model._image_data) == 2
        assert vlm_model._image_data[0] == SAMPLE_DATA_URL
        assert vlm_model._image_data[1] == SAMPLE_DATA_URL_2

    def test_multimodal_train_inputs_buffered(self, vlm_model, mock_processor):
        """Processor outputs are buffered in _multimodal_train_inputs."""
        messages = [{"role": "user", "content": [{"text": "Hello"}]}]
        vlm_model.tokenize_prompt_messages(messages, system_prompt=None)

        # pixel_values and image_grid_thw from processor should be buffered
        assert len(vlm_model._multimodal_train_inputs) == 1
        assert "pixel_values" in vlm_model._multimodal_train_inputs[0]

    def test_image_data_property(self, vlm_model):
        """image_data property returns accumulated data."""
        vlm_model._image_data = ["data:image/png;base64,abc", "data:image/png;base64,def"]
        assert vlm_model.image_data == ["data:image/png;base64,abc", "data:image/png;base64,def"]


class TestMergeMultimodalTrainInputs:
    """Tests for _merge_multimodal_train_inputs."""

    def test_empty_chunks(self):
        assert SGLangModel._merge_multimodal_train_inputs([]) is None

    def test_single_chunk(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")

        chunk = {"pixel_values": torch.randn(2, 3)}
        result = SGLangModel._merge_multimodal_train_inputs([chunk])
        assert "pixel_values" in result
        assert result["pixel_values"].shape == (2, 3)

    def test_multiple_chunks_concatenated(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")

        chunks = [
            {"pixel_values": torch.randn(2, 3), "image_grid_thw": torch.tensor([[1, 224, 224]])},
            {"pixel_values": torch.randn(4, 3), "image_grid_thw": torch.tensor([[1, 448, 448]])},
        ]
        result = SGLangModel._merge_multimodal_train_inputs(chunks)
        assert result["pixel_values"].shape[0] == 6
        assert result["image_grid_thw"].shape[0] == 2

    def test_none_chunks_skipped(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")

        chunks = [None, {"pixel_values": torch.randn(2, 3)}, None]
        result = SGLangModel._merge_multimodal_train_inputs(chunks)
        assert result["pixel_values"].shape == (2, 3)


class TestStreamWithImageData:
    """Tests for stream() passing image_data to client."""

    @pytest.mark.asyncio
    async def test_stream_passes_image_data(self, vlm_model):
        """stream() includes image_data in generate call when images are accumulated."""
        vlm_model._image_data = ["data:image/png;base64,abc"]

        # Mock the client.generate
        vlm_model.client.generate = MagicMock()
        vlm_model.client.generate.return_value = {
            "text": "This is a test response",
            "output_ids": [100, 200, 300],
            "meta_info": {"output_token_logprobs": [[-0.1, 100], [-0.2, 200], [-0.3, 300]]},
        }
        # Make it awaitable
        import asyncio

        future = asyncio.Future()
        future.set_result(vlm_model.client.generate.return_value)
        vlm_model.client.generate = MagicMock(return_value=future)

        # Need to add initial prompt tokens
        vlm_model.token_manager.add_prompt([1, 2, 3])
        vlm_model._processed_message_count = 1

        messages = [{"role": "user", "content": [{"text": "Hello"}]}]
        events = []
        async for event in vlm_model.stream(messages):
            events.append(event)

        # Verify image_data was passed to generate
        call_kwargs = vlm_model.client.generate.call_args
        assert "image_data" in call_kwargs.kwargs
        assert call_kwargs.kwargs["image_data"] == ["data:image/png;base64,abc"]

    @pytest.mark.asyncio
    async def test_stream_no_image_data_when_empty(self, text_model):
        """stream() does NOT include image_data when no images accumulated."""
        import asyncio

        text_model.client.generate = MagicMock()
        future = asyncio.Future()
        future.set_result(
            {
                "text": "response",
                "output_ids": [100],
                "meta_info": {},
            }
        )
        text_model.client.generate = MagicMock(return_value=future)

        messages = [{"role": "user", "content": [{"text": "Hello"}]}]
        events = []
        async for event in text_model.stream(messages):
            events.append(event)

        call_kwargs = text_model.client.generate.call_args
        assert "image_data" not in call_kwargs.kwargs
