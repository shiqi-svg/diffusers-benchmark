"""Reporter: JSON export + rich terminal table."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.table import Table

from benchmark.core import BenchmarkResult

console = Console()


def save_json(results: list[BenchmarkResult], path: Path | str) -> None:
    """Write results to a JSON file with metadata."""
    path = Path(path)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_entries": len(results),
        "results": [
            {
                "model_name": r.model_name,
                "model_type": r.model_type,
                "resolution": r.resolution,
                "precision": r.precision,
                "steps": r.steps,
                "guidance_scale": r.guidance_scale,
                "weight_load_time_s": r.weight_load_time_s,
                "pure_inference_time_s": r.pure_inference_time_s,
                "total_inference_time_s": r.total_inference_time_s,
                "gpu_memory_peak_mb": r.gpu_memory_peak_mb,
                "iterations": r.iterations,
                "iteration_times": r.iteration_times,
                "error": r.error,
            }
            for r in results
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    console.print(f"\n[green]✓[/] Results saved to [bold]{path}[/]")


def _truncate(s: str, max_len: int = 80) -> str:
    """Shorten a string for table display."""
    s = s.split("\n")[0]  # keep only first line
    if len(s) > max_len:
        s = s[: max_len - 3] + "..."
    return s


def print_table(results: list[BenchmarkResult]) -> None:
    """Render a rich terminal table summarising the benchmark."""
    table = Table(title="Diffusers Inference Benchmark", title_style="bold white")

    columns = [
        ("Model", "cyan"),
        ("Type", "dim"),
        ("Resolution", "yellow"),
        ("Prec", "dim"),
        ("Steps", "dim"),
        ("Load (s)", "green"),
        ("Infer (s)", "green"),
        ("Total (s)", "bold green"),
        ("VRAM (MB)", "magenta"),
        ("Error", "red"),
    ]
    for name, style in columns:
        table.add_column(name, style=style, no_wrap=True)

    for r in sorted(results, key=lambda x: (x.model_name, x.resolution)):
        error_style = " red" if r.error else ""
        table.add_row(
            r.model_name,
            r.model_type,
            r.resolution,
            r.precision,
            str(r.steps),
            f"{r.weight_load_time_s:.3f}",
            f"{r.pure_inference_time_s:.3f}",
            f"{r.total_inference_time_s:.3f}",
            str(r.gpu_memory_peak_mb),
            _truncate(r.error or ""),
            style=error_style,
        )

    console.print()
    console.print(table)
    console.print()

    # Quick summary line
    ok = sum(1 for r in results if not r.error)
    fail = sum(1 for r in results if r.error)
    console.print(f"[bold]{ok}[/] succeeded, [bold red]{fail}[/] failed")
