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
- [x] End-to-end test with tiny open SD model
- [x] Full benchmark run (steps=20, iterations=3, both 512 and 1024)
- [x] Fix Wan resolution (divisible by 16) and re-run
- [x] Results merged and saved to `results/benchmark_final.json`

## Phase 7 — Deploy
- [x] Git commit + push to https://github.com/shiqi-svg/diffusers-benchmark

## Usage
```bash
hf auth login
python -m benchmark.cli --iterations 5
python -m benchmark.cli --models FLUX.1-dev,Qwen-Image --resolutions 512,1024 --iterations 5
```
