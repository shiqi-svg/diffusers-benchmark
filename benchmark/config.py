"""Shared constants and configuration for the benchmark framework.

All speed-affecting parameters (steps, resolution) are unified across models.
Quality-only parameters (prompt, guidance_scale) use per-model defaults below.
"""

from pathlib import Path

# Unified inference steps (applied across all models)
STEPS = 20

# Resolutions to benchmark (height=width for T2I; height=value, width=mapped for T2V)
RESOLUTIONS = [512, 1024]

# Number of timed inference iterations per model+resolution combination
DEFAULT_ITERATIONS = 5

# Output directory for results and generated assets
OUTPUT_DIR = Path("results")

# ---------------------------------------------------------------------------
# Per-model default generation parameters that do NOT affect speed.
# These preserve each model's recommended quality settings.
# ---------------------------------------------------------------------------
MODEL_DEFAULTS: dict[str, dict] = {
    "FLUX.1-dev": {
        "guidance_scale": 3.5,
        "prompt": "A cinematic photo of a majestic mountain landscape at golden hour, highly detailed, 8K",
        "num_images_per_prompt": 1,
    },
    "Qwen-Image": {
        "guidance_scale": 4.0,
        "prompt": "A cinematic photo of a majestic mountain landscape at golden hour, highly detailed, 8K",
        "num_images_per_prompt": 1,
        "true_cfg_scale": 4.0,
    },
    "Wan2.1-T2V": {
        "guidance_scale": 5.0,
        "prompt": "A cinematic drone shot of a mountain landscape at golden hour, smooth camera pan, highly detailed, 8K",
        "num_frames": 9,
    },
    "Wan2.2-T2V": {
        "guidance_scale": 5.0,
        "prompt": "A cinematic drone shot of a mountain landscape at golden hour, smooth camera pan, highly detailed, 8K",
        "num_frames": 9,
    },
    "HunyuanVideo": {
        "guidance_scale": 6.0,
        "prompt": "A cinematic drone shot of a mountain landscape at golden hour, smooth camera pan, highly detailed, 8K",
        "num_frames": 61,
    },
    "LTX-2": {
        "guidance_scale": 3.0,
        "prompt": "A cinematic drone shot of a mountain landscape at golden hour, smooth camera pan, highly detailed, 8K",
        "num_frames": 121,
    },
}
