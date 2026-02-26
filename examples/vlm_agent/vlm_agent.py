#!/usr/bin/env python3

import asyncio
import json
import os

from pathlib import Path
import base64

from strands import Agent, tool
from transformers import AutoProcessor, AutoTokenizer

from strands_sglang import SGLangModel, ToolLimiter
from strands_sglang.client import SGLangClient
from strands_sglang.tool_parsers import HermesToolParser


IMAGE_DIR = Path(__file__).parent / "images"

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

async def main():
    # -------------------------------------------------------------------------
    # 1. Setup
    # -------------------------------------------------------------------------

    # Create SGLangModel with token-level trajectory tracking support
    client = SGLangClient(base_url=os.environ.get("SGLANG_BASE_URL", "http://localhost:30000"))
    model_info = await client.get_model_info()
    tokenizer = AutoTokenizer.from_pretrained(model_info["model_path"])
    processor = AutoProcessor.from_pretrained(model_info["model_path"])

    model = SGLangModel(
        client=client,
        tokenizer=tokenizer,
        processor=processor,
        tool_parser=HermesToolParser(),
        sampling_params={"max_new_tokens": 8192},
    )

    # -------------------------------------------------------------------------
    # 2. VLM Agent Example
    # -------------------------------------------------------------------------

    print("\n" + "=" * 60)
    print("VLM Agent Example")
    print("=" * 60)

    # Reset for new episode
    model.reset()

    # Create agent with read_image tool
    agent = Agent(
        model=model,
        tools=[read_image],
        hooks=[ToolLimiter(max_tool_iters=5)],
        system_prompt="",
        callback_handler=None,  # Disable print callback for cleaner output
    )

    # --- Build prompt with inline image ---
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

    # Invoke agent
    await agent.invoke_async(prompt)


    def truncate_base64(obj):
        """Truncate base64 data URLs for readable output."""
        if isinstance(obj, dict):
            return {k: truncate_base64(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [truncate_base64(v) for v in obj]
        if isinstance(obj, str) and obj.startswith("data:image/") and ";base64," in obj:
            prefix = obj[: obj.index(";base64,") + len(";base64,")]
            return prefix + obj[len(prefix) : len(prefix) + 20] + "..."
        return obj

    print(f"\n[Output Trajectory]: {json.dumps(truncate_base64(agent.messages), indent=2)}")

    # -------------------------------------------------------------------------
    # 3. Access TITO Data
    # -------------------------------------------------------------------------

    print("\n" + "-" * 40)
    print("TITO Data (for RL training)")
    print("-" * 40)

    # Token trajectory
    token_ids = model.token_manager.token_ids
    print(f"Total tokens: {len(token_ids)}")

    # Output mask (True = model output, for loss computation)
    output_mask = model.token_manager.loss_mask
    n_output = sum(output_mask)
    n_prompt = len(output_mask) - n_output
    print(f"Prompt tokens: {n_prompt} (loss_mask=False)")
    print(f"Response tokens: {n_output} (loss_mask=True)")

    # Log probabilities
    logprobs = model.token_manager.logprobs
    output_logprobs = [lp for lp, mask in zip(logprobs, output_mask) if mask and lp is not None]
    if output_logprobs:
        avg_logprob = sum(output_logprobs) / len(output_logprobs)
        print(f"Average output logprob: {avg_logprob:.4f}")

    # Segment info
    segment_info = model.token_manager.segment_info
    print(f"Segments: {len(segment_info)} (Note: Segment 0 includes the system prompt and the user input)")
    for i, (is_output, length) in enumerate(segment_info):
        seg_type = "Response" if is_output else "Prompt"
        print(f"  Segment {i}: {seg_type} ({length} tokens)")

    # -------------------------------------------------------------------------
    # VLM State
    # -------------------------------------------------------------------------

    print("\n" + "-" * 40)
    print("VLM State")
    print("-" * 40)
    print(f"Images accumulated: {len(model.image_data)}")
    for i, img_url in enumerate(model.image_data):
        print(f"  [{i}] {img_url[:50]}...")

    mm_inputs = model.multimodal_train_inputs
    if mm_inputs:
        print(f"Multimodal train inputs:")
        for key, val in mm_inputs.items():
            if hasattr(val, "shape"):
                print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
            else:
                print(f"  {key}: {type(val).__name__}")
    else:
        print("Multimodal train inputs: None")

    # Decode full context (vision tokens show as <|image_pad|> placeholders)
    full_text = tokenizer.decode(model.token_manager.token_ids, skip_special_tokens=False)
    print("\n" + "-" * 40)
    print("Full Context (decoded token sequence)")
    print("-" * 40)
    print(full_text)

    # Cleanup
    model.reset()
    agent.cleanup()
    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
