from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


AUTO_COMPARATORS = {
    "psnr": "max",
    "ssim": "max",
    "lpips": "min",
}


@dataclass
class StudyPaths:
    root: Path
    manifest_path: Path
    leaderboard_path: Path
    stage_root: Path
    result_root: Path
    hydra_root: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch staged PDHG hyperparameter sweeps and summarize the results."
    )
    parser.add_argument(
        "--config",
        default="tuning/pdhg_tuning.template.yaml",
        help="Path to the tuning YAML file.",
    )
    parser.add_argument(
        "--stage",
        action="append",
        default=[],
        help="Optional stage name filter. Pass multiple times to run several stages.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Optional cap on the number of launched runs across all stages.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands and write manifests without launching the experiments.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at root of {path}")
    return data


def sanitize_for_path(text: str) -> str:
    allowed = []
    for char in text:
        if char.isalnum() or char in {"-", "_", "."}:
            allowed.append(char)
        else:
            allowed.append("_")
    return "".join(allowed).strip("._") or "item"


def serialize_override_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.15g}"
    if isinstance(value, list):
        inner = ",".join(serialize_override_value(item) for item in value)
        return f"[{inner}]"
    return str(value)


def expand_parameter_spec(name: str, spec: dict[str, Any]) -> list[Any]:
    if not isinstance(spec, dict):
        raise ValueError(f"Parameter spec for {name} must be a mapping")

    if "values" in spec:
        values = spec["values"]
        if not isinstance(values, list):
            raise ValueError(f"'values' for {name} must be a list")
        if not values:
            raise ValueError(f"'values' for {name} is empty")
        return values

    if "linspace" in spec:
        cfg = spec["linspace"]
        if not isinstance(cfg, dict):
            raise ValueError(f"'linspace' for {name} must be a mapping")
        start = float(cfg["start"])
        end = float(cfg["end"])
        num = int(cfg["num"])
        if num <= 0:
            raise ValueError(f"'linspace.num' for {name} must be positive")
        if num == 1:
            return [start]
        step = (end - start) / (num - 1)
        return [start + idx * step for idx in range(num)]

    if "logspace" in spec:
        cfg = spec["logspace"]
        if not isinstance(cfg, dict):
            raise ValueError(f"'logspace' for {name} must be a mapping")
        start = float(cfg["start"])
        end = float(cfg["end"])
        num = int(cfg["num"])
        base = float(cfg.get("base", 10.0))
        if num <= 0:
            raise ValueError(f"'logspace.num' for {name} must be positive")
        if start <= 0 or end <= 0:
            raise ValueError(f"'logspace' values for {name} must be > 0")
        if num == 1:
            return [start]
        log_start = math.log(start, base)
        log_end = math.log(end, base)
        step = (log_end - log_start) / (num - 1)
        return [base ** (log_start + idx * step) for idx in range(num)]

    raise ValueError(
        f"Parameter {name} must define one of: values, linspace, logspace"
    )


def expand_stage(stage_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    parameters = stage_cfg.get("parameters", {})
    if not parameters:
        raise ValueError(f"Stage '{stage_cfg.get('name', '<unnamed>')}' has no parameters")

    keys = list(parameters.keys())
    value_lists = [expand_parameter_spec(key, parameters[key]) for key in keys]
    combinations = []
    for values in itertools.product(*value_lists):
        combinations.append(dict(zip(keys, values)))
    return combinations


def resolve_comparator(scoring_cfg: dict[str, Any]) -> str:
    primary_metric = scoring_cfg["primary_metric"]
    comparator = scoring_cfg.get("comparator", "auto")
    if comparator == "auto":
        if primary_metric not in AUTO_COMPARATORS:
            raise ValueError(
                f"Comparator is 'auto' but metric '{primary_metric}' is unknown"
            )
        return AUTO_COMPARATORS[primary_metric]
    if comparator not in {"max", "min", "mean"}:
        raise ValueError("Comparator must be one of: auto, max, min, mean")
    return comparator


def score_metrics(metrics: dict[str, Any], scoring_cfg: dict[str, Any]) -> dict[str, float]:
    primary_metric = scoring_cfg["primary_metric"]
    comparator = resolve_comparator(scoring_cfg)
    aggregate = scoring_cfg.get("aggregate", "mean")
    if aggregate != "mean":
        raise ValueError("Only aggregate='mean' is currently supported")

    if primary_metric not in metrics:
        raise KeyError(f"Primary metric '{primary_metric}' not found in metrics.json")

    selected_values = metrics[primary_metric][comparator]
    score = sum(selected_values) / len(selected_values)

    summary = {"score": score}
    for metric_name, comparator_name in AUTO_COMPARATORS.items():
        if metric_name in metrics:
            values = metrics[metric_name][comparator_name]
            summary[f"{metric_name}_{comparator_name}_mean"] = sum(values) / len(values)
    return summary


def create_study_paths(cfg: dict[str, Any], config_path: Path) -> StudyPaths:
    study_cfg = cfg["study"]
    study_name = sanitize_for_path(study_cfg["name"])
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    root = Path(study_cfg.get("output_root", "tuning_runs")) / study_name / timestamp
    stage_root = root / "stages"
    result_root = Path(study_cfg.get("result_root", "results/tuning")) / study_name / timestamp
    hydra_root = Path(study_cfg.get("hydra_root", "outputs/tuning")) / study_name / timestamp

    root.mkdir(parents=True, exist_ok=True)
    stage_root.mkdir(parents=True, exist_ok=True)

    manifest_path = root / "study_manifest.json"
    leaderboard_path = root / "leaderboard.csv"

    manifest = {
        "study_name": study_cfg["name"],
        "created_at": timestamp,
        "tuning_config": str(config_path.as_posix()),
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    return StudyPaths(
        root=root,
        manifest_path=manifest_path,
        leaderboard_path=leaderboard_path,
        stage_root=stage_root,
        result_root=result_root,
        hydra_root=hydra_root,
    )


def locate_metrics_file(save_dir: Path) -> Path:
    matches = sorted(save_dir.rglob("metrics.json"))
    if not matches:
        raise FileNotFoundError(f"No metrics.json found under {save_dir}")
    if len(matches) > 1:
        raise FileNotFoundError(
            f"Expected one metrics.json under {save_dir}, found {len(matches)}"
        )
    return matches[0]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_command(
    cfg: dict[str, Any],
    combination: dict[str, Any],
    run_save_dir: Path,
    hydra_run_dir: Path,
    stage_cfg: dict[str, Any],
) -> list[str]:
    runner_cfg = cfg["runner"]
    base_overrides = list(cfg.get("base_overrides", []))
    stage_overrides = list(stage_cfg.get("overrides", []))

    overrides = base_overrides + stage_overrides
    overrides.extend(
        f"{key}={serialize_override_value(value)}" for key, value in combination.items()
    )
    overrides.append(f"save_dir={run_save_dir.as_posix()}")
    overrides.append(f"hydra.run.dir={hydra_run_dir.as_posix()}")

    command = [
        runner_cfg.get("python_executable", sys.executable),
        runner_cfg.get("entrypoint", "recover_inverse2.py"),
        "--config-name",
        runner_cfg.get("config_name", "default_ffhq.yaml"),
        *overrides,
    ]
    return command


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    cfg = load_yaml(config_path)

    workdir = Path(cfg["runner"].get("workdir", ".")).resolve()
    study_paths = create_study_paths(cfg, config_path)
    scoring_cfg = cfg["scoring"]

    stage_filter = set(args.stage)
    all_rows: list[dict[str, Any]] = []
    launched_runs = 0

    for stage_cfg in cfg.get("stages", []):
        stage_name = stage_cfg["name"]
        if stage_filter and stage_name not in stage_filter:
            continue

        combinations = expand_stage(stage_cfg)
        stage_dir = study_paths.stage_root / sanitize_for_path(stage_name)
        stage_dir.mkdir(parents=True, exist_ok=True)

        stage_manifest = {
            "stage_name": stage_name,
            "description": stage_cfg.get("description", ""),
            "num_combinations": len(combinations),
            "parameters": stage_cfg.get("parameters", {}),
            "overrides": stage_cfg.get("overrides", []),
        }
        with (stage_dir / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(stage_manifest, handle, indent=2)

        stage_rows: list[dict[str, Any]] = []
        for combo_index, combination in enumerate(combinations, start=1):
            if args.max_runs is not None and launched_runs >= args.max_runs:
                break

            run_slug = f"run_{combo_index:03d}"
            stage_rel = Path(sanitize_for_path(stage_name))
            run_save_dir = study_paths.result_root / stage_rel / run_slug
            hydra_run_dir = study_paths.hydra_root / stage_rel / run_slug
            run_log_path = stage_dir / f"{run_slug}.launcher.log"

            command = build_command(
                cfg=cfg,
                combination=combination,
                run_save_dir=run_save_dir,
                hydra_run_dir=hydra_run_dir,
                stage_cfg=stage_cfg,
            )

            row: dict[str, Any] = {
                "stage": stage_name,
                "run_id": run_slug,
                "status": "dry_run" if args.dry_run else "pending",
                "score": "",
                "metrics_path": "",
                "launcher_log": str(run_log_path.as_posix()),
                "save_dir": str(run_save_dir.as_posix()),
                "hydra_run_dir": str(hydra_run_dir.as_posix()),
                "command": " ".join(command),
            }
            row.update({key: serialize_override_value(value) for key, value in combination.items()})

            print(f"[{stage_name} | {run_slug}]")
            print("  " + " ".join(command))

            launched_runs += 1
            if args.dry_run:
                stage_rows.append(row)
                all_rows.append(row)
                continue

            run_save_dir.mkdir(parents=True, exist_ok=True)
            hydra_run_dir.mkdir(parents=True, exist_ok=True)

            started = time.time()
            with run_log_path.open("w", encoding="utf-8") as log_handle:
                process = subprocess.run(
                    command,
                    cwd=workdir,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            duration_sec = time.time() - started

            row["duration_sec"] = f"{duration_sec:.2f}"
            if process.returncode != 0:
                row["status"] = f"failed({process.returncode})"
                stage_rows.append(row)
                all_rows.append(row)
                continue

            try:
                metrics_path = locate_metrics_file(run_save_dir.resolve())
                with metrics_path.open("r", encoding="utf-8") as handle:
                    metrics = json.load(handle)
                score_summary = score_metrics(metrics, scoring_cfg)
                row["status"] = "ok"
                row["score"] = f"{score_summary['score']:.6f}"
                row["metrics_path"] = str(metrics_path.as_posix())
                for key, value in score_summary.items():
                    if key == "score":
                        continue
                    row[key] = f"{value:.6f}"
            except Exception as exc:  # noqa: BLE001
                row["status"] = f"metrics_error({exc})"

            stage_rows.append(row)
            all_rows.append(row)

        stage_rows.sort(
            key=lambda item: (
                1 if item["status"] != "ok" else 0,
                -(float(item["score"]) if item["score"] not in {"", None} else float("-inf")),
            )
        )
        write_csv(stage_dir / "leaderboard.csv", stage_rows)
        with (stage_dir / "leaderboard.json").open("w", encoding="utf-8") as handle:
            json.dump(stage_rows, handle, indent=2)

        if args.max_runs is not None and launched_runs >= args.max_runs:
            break

    all_rows.sort(
        key=lambda item: (
            1 if item["status"] != "ok" else 0,
            -(float(item["score"]) if item["score"] not in {"", None} else float("-inf")),
        )
    )
    write_csv(study_paths.leaderboard_path, all_rows)
    with (study_paths.root / "leaderboard.json").open("w", encoding="utf-8") as handle:
        json.dump(all_rows, handle, indent=2)

    print(f"\nStudy artifacts written to: {study_paths.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
