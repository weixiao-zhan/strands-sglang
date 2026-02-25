#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
import base64
import logging
import os
from pathlib import Path

from strands import Agent, tool
from transformers import AutoProcessor, AutoTokenizer

from strands_sglang import SGLangClient, SGLangModel, ToolLimiter
from strands_sglang.tool_parsers import HermesToolParser

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

IMAGE_DIR = Path(__file__).parent / "images"

SYSTEM_PROMPT = """"""

# ---------------------------------------------------------------------------
# Tool: read_image
# ---------------------------------------------------------------------------

@tool
def read_image(file_path: str) -> dict:
    """Read an image file and return it for visual inspection.

    Args:
        file_path: Filename of the image (e.g., "a.jpg", "b.png").
    """
    # Resolve: try the path as-is first, then look in IMAGE_DIR
    path = Path(file_path)
    if not path.exists():
        # Try just the filename in IMAGE_DIR (handles "images/a.jpg" -> "a.jpg")
        path = IMAGE_DIR / path.name

    if not path.exists():
        return {
            "status": "error",
            "content": [{"text": f"File not found: {path}"}],
        }

    # Read image bytes and encode as base64 data URL
    image_bytes = path.read_bytes()
    suffix = path.suffix.lower().lstrip(".")
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "gif": "gif", "webp": "webp"}.get(suffix, "png")
    data_url = f"data:image/{mime};base64,{base64.b64encode(image_bytes).decode()}"

    return {
        "status": "success",
        "content": [
            {"text": f"Image loaded: {path.name} ({len(image_bytes)} bytes)"},
            {"image": data_url},
        ],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def run_agent(base_url: str, model_path: str) -> None:
    """Run the VLM agent and inspect TITO state."""
    # --- Setup ---
    client = SGLangClient(base_url=base_url)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    model = SGLangModel(
        client=client,
        tokenizer=tokenizer,
        processor=processor,
        tool_parser=HermesToolParser(),
        sampling_params={"max_new_tokens": 1024, "temperature": 0.7},
    )

    tool_limiter = ToolLimiter(max_tool_iters=5)
    agent = Agent(
        model=model,
        tools=[read_image],
        hooks=[tool_limiter],
        system_prompt=SYSTEM_PROMPT,
    )

    # --- Build prompt with inline image ---
    # a.jpg is provided directly in the user message (image in initial prompt).
    # b.png must be loaded via the read_image tool (image from tool result).
    #
    # Strands Agent accepts list[ContentBlock] as prompt, where ContentBlock
    # is a TypedDict with optional keys: text, image, toolResult, etc.
    # Images are plain base64 data URL strings.
    image_a_bytes = (IMAGE_DIR / "a.jpg").read_bytes()
    image_a_data_url = f"data:image/jpeg;base64,{base64.b64encode(image_a_bytes).decode()}"
    prompt = [
        {"image": image_a_data_url},
        {
            "text": (
                "This is a.jpg above. "
                "Now use read_image to load b.jpg. "
                "Then describe both images and compare them."
            ),
        },
    ]

    try:
        result = await agent.invoke_async(prompt)
        logger.info("Agent completed successfully")
    except Exception as e:
        logger.warning(f"Agent stopped: {type(e).__name__}: {e}")

    # --- Inspect TITO state ---
    tm = model.token_manager
    print("\n" + "=" * 60)
    print("TITO Token Manager State")
    print("=" * 60)
    print(f"  Total tokens:    {len(tm)}")
    print(f"  Segments:        {len(tm.segments)}")
    if not tm.segments:
        print("  (no segments — agent may have failed before generation)")
        model.reset()
        agent.cleanup()
        await client.close()
        return

    for i, (is_output, length) in enumerate(tm.segment_info):
        label = "RESPONSE" if is_output else "PROMPT"
        print(f"    [{i}] {label:8s}  {length:5d} tokens")

    prompt_len = len(tm.segments[0])
    print(f"\n  Initial prompt:  {prompt_len} tokens")
    print(f"  Rollout tokens:  {len(tm) - prompt_len}")
    print(f"  Output tokens:   {sum(1 for t in tm.tokens if t.loss_mask)}")

    print(f"\n  Loss mask:       {tm.loss_mask[:10]}... (first 10)")
    has_logprobs = any(lp is not None for lp in tm.logprobs)
    print(f"  Has logprobs:    {has_logprobs}")

    # --- Inspect VLM state ---
    print("\n" + "=" * 60)
    print("VLM State")
    print("=" * 60)
    print(f"  Images accumulated:  {len(model.image_data)}")
    for i, img_url in enumerate(model.image_data):
        print(f"    [{i}] {img_url[:50]}...")

    mm_inputs = model.multimodal_train_inputs
    if mm_inputs:
        print(f"\n  Multimodal train inputs:")
        for key, val in mm_inputs.items():
            if hasattr(val, "shape"):
                print(f"    {key}: shape={val.shape}, dtype={val.dtype}")
            else:
                print(f"    {key}: {type(val).__name__}")
    else:
        print("\n  Multimodal train inputs: None")

    print(f"\n  Tool iterations: {tool_limiter.tool_iter_count}")
    print(f"  Tool calls:      {tool_limiter.tool_call_count}")

    # --- Decode full context ---
    # Decode the entire token sequence so we can see the full conversation
    # including system prompt, user message, tool calls, tool results, and
    # final response. Vision tokens decode as <|image_pad|> placeholders.
    full_text = tokenizer.decode(tm.token_ids, skip_special_tokens=False)
    print("\n" + "=" * 60)
    print("Full Context (decoded token sequence)")
    print("=" * 60)
    print(full_text)

    # Cleanup
    model.reset()
    agent.cleanup()
    await client.close()


def main():
    parser = argparse.ArgumentParser(description="VLM Agent with image-returning tools")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("SGLANG_BASE_URL", "http://localhost:30000"),
        help="SGLang server URL (default: http://localhost:30000)",
    )
    parser.add_argument(
        "--model-path",
        default=os.environ.get("MODEL_PATH", "Qwen/Qwen3-VL-2B-Instruct"),
        help="HuggingFace model path for tokenizer/processor (default: Qwen/Qwen3-VL-2B-Instruct)",
    )
    args = parser.parse_args()

    # Ensure images exist
    if not (IMAGE_DIR / "a.jpg").exists():
        print("Sample images not found. Run download_images.py first:")
        print("  python examples/vlm_agent/download_images.py")
        return

    asyncio.run(run_agent(args.base_url, args.model_path))


if __name__ == "__main__":
    main()
