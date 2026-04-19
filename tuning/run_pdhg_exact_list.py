from __future__ import annotations

import argparse
import csv
import json
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
    progress_path: Path
    leaderboard_csv: Path
    leaderboard_json: Path
    candidates_path: Path
    run_root: Path
    current_candidate_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an exact subprocess-based PDHG sweep over an explicit candidate list."
    )
    parser.add_argument(
        "--config",
        default="tuning/pdhg_exact_list.template.yaml",
        help="Path to the exact candidate-list tuning YAML file.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Optional cap on the number of candidates to evaluate.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write manifests and print commands without launching runs.",
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


def resolve_comparator(scoring_cfg: dict[str, Any]) -> str:
    primary_metric = scoring_cfg["primary_metric"]
    comparator = scoring_cfg.get("comparator", "auto")
    if comparator == "auto":
        if primary_metric not in AUTO_COMPARATORS:
            raise ValueError(f"Unknown auto comparator for metric '{primary_metric}'")
        return AUTO_COMPARATORS[primary_metric]
    if comparator not in {"max", "min"}:
        raise ValueError("Comparator must be one of: auto, max, min")
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
    root.mkdir(parents=True, exist_ok=True)
    run_root = root / "runs"
    run_root.mkdir(parents=True, exist_ok=True)

    manifest_path = root / "study_manifest.json"
    progress_path = root / "progress.json"
    leaderboard_csv = root / "leaderboard.csv"
    leaderboard_json = root / "leaderboard.json"
    candidates_path = root / "candidates.json"
    current_candidate_path = root / "current_candidate.json"

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
        progress_path=progress_path,
        leaderboard_csv=leaderboard_csv,
        leaderboard_json=leaderboard_json,
        candidates_path=candidates_path,
        run_root=run_root,
        current_candidate_path=current_candidate_path,
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


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def sort_rows(rows: list[dict[str, Any]], scoring_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    comparator = resolve_comparator(scoring_cfg)
    reverse = comparator == "max"
    return sorted(rows, key=lambda row: row["score"], reverse=reverse)


def get_successful_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("status") == "ok"]


def normalize_candidate(candidate: dict[str, Any], index: int) -> dict[str, Any]:
    if not isinstance(candidate, dict):
        raise ValueError(f"Candidate {index} must be a mapping")
    params = candidate.get("params", {})
    if not isinstance(params, dict) or not params:
        raise ValueError(f"Candidate {index} must define a non-empty 'params' mapping")
    name = str(candidate.get("name", f"candidate_{index:03d}"))
    return {"name": name, "params": params}


def load_candidates(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    raw_candidates = cfg.get("candidates", [])
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("Expected a non-empty 'candidates' list")
    return [normalize_candidate(candidate, index) for index, candidate in enumerate(raw_candidates)]


def build_command(
    cfg: dict[str, Any],
    candidate: dict[str, Any],
    run_save_dir: Path,
    hydra_run_dir: Path,
) -> list[str]:
    runner_cfg = cfg["runner"]
    base_overrides = list(cfg.get("base_overrides", []))
    param_overrides = [
        f"{key}={serialize_override_value(value)}" for key, value in candidate["params"].items()
    ]

    overrides = list(base_overrides)
    overrides.extend(param_overrides)
    overrides.append(f"save_dir={run_save_dir.as_posix()}")
    overrides.append(f"hydra.run.dir={hydra_run_dir.as_posix()}")

    return [
        runner_cfg.get("python_executable", sys.executable),
        runner_cfg.get("entrypoint", "recover_inverse2.py"),
        "--config-name",
        runner_cfg.get("config_name", "default_ffhq.yaml"),
        *overrides,
    ]


def update_progress(
    path: Path,
    *,
    status: str,
    total_candidates: int,
    completed_candidates: int,
    started_at: float,
    current_candidate_index: int | None = None,
    current_candidate: dict[str, Any] | None = None,
    rows: list[dict[str, Any]] | None = None,
    current_log_path: str | None = None,
) -> None:
    elapsed_seconds = max(0.0, time.time() - started_at)
    avg_candidate_seconds = elapsed_seconds / completed_candidates if completed_candidates > 0 else None
    remaining_candidates = max(0, total_candidates - completed_candidates)
    eta_seconds = None if avg_candidate_seconds is None else avg_candidate_seconds * remaining_candidates

    payload: dict[str, Any] = {
        "status": status,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": elapsed_seconds,
        "completed_candidates": completed_candidates,
        "total_candidates": total_candidates,
        "current_candidate_index": current_candidate_index,
        "current_candidate": current_candidate,
        "current_log_path": current_log_path,
        "eta_seconds": eta_seconds,
    }

    successful_rows = get_successful_rows(rows or [])
    if successful_rows:
        best_row = successful_rows[0]
        payload["best_so_far"] = {
            "candidate_index": best_row["candidate_index"],
            "candidate_name": best_row.get("candidate_name"),
            "score": best_row["score"],
            "sigma_max": best_row.get("sampler.annealing_scheduler_config.sigma_max"),
            "sigma_min": best_row.get("sampler.annealing_scheduler_config.sigma_min"),
            "tau": best_row.get("inverse_task.admm_config.pdhg.tau"),
            "sigma_dual": best_row.get("inverse_task.admm_config.pdhg.sigma_dual"),
        }

    write_json(path, payload)


def print_summary(rows: list[dict[str, Any]], scoring_cfg: dict[str, Any], top_k: int) -> None:
    if not rows:
        print("No candidate rows to summarize.")
        return

    primary_metric = scoring_cfg["primary_metric"]
    comparator = resolve_comparator(scoring_cfg)
    direction = "highest" if comparator == "max" else "lowest"
    print(f"\nTop {min(top_k, len(rows))} candidates by {direction} mean {primary_metric}:")
    for rank, row in enumerate(rows[:top_k], start=1):
        print(
            f"  {rank:>2}. idx={row['candidate_index']:>3} "
            f"name={row.get('candidate_name')} "
            f"score={row['score']:.4f} "
            f"sigma_max={row.get('sampler.annealing_scheduler_config.sigma_max')} "
            f"sigma_min={row.get('sampler.annealing_scheduler_config.sigma_min')} "
            f"tau={row.get('inverse_task.admm_config.pdhg.tau')} "
            f"sigma_dual={row.get('inverse_task.admm_config.pdhg.sigma_dual')}"
        )


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    cfg = load_yaml(config_path)

    candidates = load_candidates(cfg)
    if args.max_candidates is not None:
        candidates = candidates[:args.max_candidates]
    if not candidates:
        raise ValueError("Candidate list is empty after applying filters.")

    study_paths = create_study_paths(cfg, config_path)
    with study_paths.candidates_path.open("w", encoding="utf-8") as handle:
        json.dump(candidates, handle, indent=2)

    print(f"Loaded {len(candidates)} candidates from {config_path}")

    if args.dry_run:
        started_at = time.time()
        update_progress(
            study_paths.progress_path,
            status="dry_run",
            total_candidates=len(candidates),
            completed_candidates=0,
            started_at=started_at,
        )
        preview = candidates[: min(5, len(candidates))]
        print("Preview candidates:")
        for candidate in preview:
            print(f"  {candidate}")
        return 0

    scoring_cfg = cfg["scoring"]
    report_cfg = cfg.get("report", {})
    top_k = int(report_cfg.get("top_k", 10))
    runner_cfg = cfg["runner"]
    workdir = Path(runner_cfg.get("workdir", ".")).resolve()

    rows: list[dict[str, Any]] = []
    started_at = time.time()
    update_progress(
        study_paths.progress_path,
        status="running",
        total_candidates=len(candidates),
        completed_candidates=0,
        started_at=started_at,
    )

    for candidate_index, candidate in enumerate(candidates):
        run_slug = f"candidate_{candidate_index:03d}"
        run_name = sanitize_for_path(candidate["name"])
        run_dir = study_paths.run_root / f"{run_slug}_{run_name}"
        run_dir.mkdir(parents=True, exist_ok=True)
        hydra_run_dir = run_dir / "hydra"
        run_save_dir = run_dir / "results"
        run_log_path = run_dir / "launcher.log"

        command = build_command(
            cfg=cfg,
            candidate=candidate,
            run_save_dir=run_save_dir,
            hydra_run_dir=hydra_run_dir,
        )

        row: dict[str, Any] = {
            "candidate_index": candidate_index,
            "candidate_name": candidate["name"],
            "status": "pending",
            "score": "",
            "metrics_path": "",
            "launcher_log": str(run_log_path.as_posix()),
            "save_dir": str(run_save_dir.as_posix()),
            "hydra_run_dir": str(hydra_run_dir.as_posix()),
            "command": " ".join(command),
        }
        row.update(candidate["params"])

        write_json(
            study_paths.current_candidate_path,
            {
                "candidate_index": candidate_index,
                "candidate_name": candidate["name"],
                "params": candidate["params"],
                "run_dir": str(run_dir.as_posix()),
                "launcher_log": str(run_log_path.as_posix()),
            },
        )
        update_progress(
            study_paths.progress_path,
            status="running",
            total_candidates=len(candidates),
            completed_candidates=len(rows),
            started_at=started_at,
            current_candidate_index=candidate_index,
            current_candidate={
                "name": candidate["name"],
                "params": candidate["params"],
            },
            rows=sort_rows(get_successful_rows(rows), scoring_cfg) if rows else [],
            current_log_path=str(run_log_path.as_posix()),
        )

        print(f"[candidate {candidate_index + 1}/{len(candidates)}] {candidate['name']}")
        print("  " + " ".join(command))

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
            rows.append(row)
        else:
            try:
                metrics_path = locate_metrics_file(run_save_dir.resolve())
                with metrics_path.open("r", encoding="utf-8") as handle:
                    metrics = json.load(handle)
                score_summary = score_metrics(metrics, scoring_cfg)
                row["status"] = "ok"
                row["score"] = float(score_summary["score"])
                row["metrics_path"] = str(metrics_path.as_posix())
                for key, value in score_summary.items():
                    if key == "score":
                        continue
                    row[key] = float(value)
            except Exception as exc:  # noqa: BLE001
                row["status"] = f"metrics_error({exc})"
            rows.append(row)

        sorted_successful = sort_rows(get_successful_rows(rows), scoring_cfg)
        write_csv(study_paths.leaderboard_csv, sorted_successful if sorted_successful else rows)
        write_json(study_paths.leaderboard_json, sorted_successful if sorted_successful else rows)
        update_progress(
            study_paths.progress_path,
            status="running",
            total_candidates=len(candidates),
            completed_candidates=len(rows),
            started_at=started_at,
            current_candidate_index=candidate_index,
            current_candidate={
                "name": candidate["name"],
                "params": candidate["params"],
            },
            rows=sorted_successful,
            current_log_path=str(run_log_path.as_posix()),
        )

    sorted_successful = sort_rows(get_successful_rows(rows), scoring_cfg)
    final_rows = sorted_successful if sorted_successful else rows
    write_csv(study_paths.leaderboard_csv, final_rows)
    write_json(study_paths.leaderboard_json, final_rows)
    update_progress(
        study_paths.progress_path,
        status="completed",
        total_candidates=len(candidates),
        completed_candidates=len(rows),
        started_at=started_at,
        current_candidate_index=None,
        current_candidate=None,
        rows=sorted_successful,
        current_log_path=None,
    )

    print(f"\nSaved leaderboard to {study_paths.leaderboard_csv}")
    if sorted_successful:
        print_summary(sorted_successful, scoring_cfg, top_k=top_k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
