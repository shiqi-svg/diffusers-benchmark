"""Shared constants and configuration for the benchmark framework."""

from pathlib import Path

# Unified inference steps (applied across all models)
STEPS = 20

# Resolutions to benchmark (height=width for T2I; height=value, width=mapped for T2V)
RESOLUTIONS = [512, 1024]

# Number of timed inference iterations per model+resolution combination
DEFAULT_ITERATIONS = 5

# Output directory for results and generated assets
OUTPUT_DIR = Path("results")

# Warm-up: True means a warmup infer() run happens before timing, excluded from results
WARMUP = True

# Default guidance scale (may be overridden per model if it does not affect speed)
DEFAULT_GUIDANCE_SCALE = 3.5

# Precision mapping per model — all default to bfloat16 on H100
MODEL_PRECISION = {
    "FLUX.1-dev": "bfloat16",
    "Qwen-Image": "bfloat16",
    "Wan2.1-T2V": "bfloat16",
    "Wan2.2-T2V": "bfloat16",
    "HunyuanVideo": "bfloat16",
    "LTX-2": "bfloat16",
}

# ---------------------------------------------------------------------------
# Per-model default generation parameters that do NOT affect speed.
# These preserve each model's recommended quality settings.
# ---------------------------------------------------------------------------
MODEL_DEFAULTS = {
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
