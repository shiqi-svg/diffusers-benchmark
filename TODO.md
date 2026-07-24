# TODO: Diffusers Inference Benchmark Framework

## Phase 1 — Project Skeleton
- [x] Create `requirements.txt`
- [x] Create `benchmark/` package structure
- [x] Create `benchmark/config.py` — shared constants

## Phase 2 — Core Framework
- [x] Create `benchmark/core.py` — Timer, BenchmarkResult, BenchmarkRunner

## Phase 3 — Model Base & Registry
- [x] Create `benchmark/models/base.py` — BaseModel abstract class
- [x] Create `benchmark/models/__init__.py` — ModelRegistry + `@register_model` decorator

## Phase 4 — Model Adapters
- [x] Create `benchmark/models/flux.py` — FLUX.1-dev
- [x] Create `benchmark/models/qwen_image.py` — Qwen-Image
- [x] Create `benchmark/models/wan.py` — Wan2.1-T2V + Wan2.2-T2V
- [x] Create `benchmark/models/hunyuan.py` — HunyuanVideo
- [x] Create `benchmark/models/ltx2.py` — LTX-2

## Phase 5 — Reporter & CLI
- [x] Create `benchmark/reporter.py` — JSON export + `rich` terminal table
- [x] Create `benchmark/cli.py` — argparse entrypoint

## Phase 6 — Verification
- [x] Import check: verify all modules import without errors
- [x] Quick-test: `python -m benchmark.cli --models FLUX.1-dev --resolutions 512 --iterations 1` (gated model, error handled correctly)
- [x] End-to-end test with tiny open SD model — load/infer timing, JSON, table all verified
- [x] Verify JSON output structure is complete
- [x] Verify terminal table renders correctly
- [ ] Test all models at 512 resolution, 1 iteration (requires HF auth for gated models)
- [ ] Full run: all models, both resolutions, 5 iterations (requires HF auth for gated models)

## Notes
- Most target models (FLUX.1-dev, Qwen-Image, Wan*, HunyuanVideo, LTX-2) are gated and require `HF_TOKEN` authentication.
- Before running the full benchmark: `hf auth login` or set `HF_TOKEN` env var.
- Run: `python -m benchmark.cli --models FLUX.1-dev,Qwen-Image --resolutions 512,1024 --iterations 5`
- Run all: `python -m benchmark.cli --iterations 5`
