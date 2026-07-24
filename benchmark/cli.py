"""CLI entrypoint for the benchmark framework.

Usage::

    python -m benchmark.cli \\
        --models FLUX.1-dev,Qwen-Image \\
        --resolutions 512,1024 \\
        --steps 20 \\
        --iterations 5 \\
        --output results/benchmark.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from benchmark.config import DEFAULT_ITERATIONS, OUTPUT_DIR, RESOLUTIONS, STEPS
from benchmark.core import BenchmarkRunner
from benchmark.models import get_all_models, get_model
from benchmark.reporter import print_table, save_json

logger = logging.getLogger("benchmark")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark diffusers image/video model inference times",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model names to benchmark (default: all registered)",
    )
    parser.add_argument(
        "--resolutions",
        type=str,
        default="512,1024",
        help="Comma-separated resolution values (default: 512,1024)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=STEPS,
        help=f"Number of inference steps (default: {STEPS})",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
        help=f"Number of timed iterations per run (default: {DEFAULT_ITERATIONS})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR / "benchmark.json",
        help=f"JSON output path (default: {OUTPUT_DIR / 'benchmark.json'})",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Resolve models
    if args.models:
        names = [n.strip() for n in args.models.split(",")]
        all_models = get_all_models()
        model_classes = []
        for name in names:
            cls = all_models.get(name)
            if cls is None:
                logger.error("Unknown model: %s (available: %s)", name, list(all_models.keys()))
                sys.exit(1)
            model_classes.append(cls)
    else:
        model_classes = list(get_all_models().values())

    logger.info("Models to benchmark: %s", [m.model_name for m in model_classes])

    # Resolve resolutions
    resolutions = [int(x.strip()) for x in args.resolutions.split(",")]

    # Run
    runner = BenchmarkRunner(
        steps=args.steps,
        resolutions=resolutions,
        iterations=args.iterations,
    )
    results = runner.run(model_classes)

    # Report
    print_table(results)
    save_json(results, args.output)


if __name__ == "__main__":
    main()
