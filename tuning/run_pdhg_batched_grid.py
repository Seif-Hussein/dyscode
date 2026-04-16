from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


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
    chunks_root: Path
    config_snapshot_path: Path
    candidates_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a chunked candidate-batched PDHG hyperparameter sweep."
    )
    parser.add_argument(
        "--config",
        default="tuning/pdhg_batched_grid.template.yaml",
        help="Path to the batched tuning YAML file.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Optional cap on the number of grid candidates to evaluate.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Optional override for batched.candidate_chunk_size.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Expand the grid and show the chunk plan without loading the model.",
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


def expand_parameter_spec(name: str, spec: dict[str, Any]) -> list[Any]:
    if not isinstance(spec, dict):
        raise ValueError(f"Parameter spec for {name} must be a mapping")

    if "values" in spec:
        values = spec["values"]
        if not isinstance(values, list) or not values:
            raise ValueError(f"'values' for {name} must be a non-empty list")
        return values

    raise ValueError(f"Parameter {name} must define 'values'")


def expand_grid(grid_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(grid_cfg, dict) or not grid_cfg:
        raise ValueError("Grid configuration must be a non-empty mapping")

    keys = list(grid_cfg.keys())
    value_lists = [expand_parameter_spec(key, grid_cfg[key]) for key in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*value_lists)]


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


def lookup_dotted(mapping: dict[str, Any], dotted_key: str) -> Any:
    current: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(f"Key '{dotted_key}' not found in resolved config")
        current = current[part]
    return current


def create_study_paths(cfg: dict[str, Any], config_path: Path) -> StudyPaths:
    study_cfg = cfg["study"]
    study_name = sanitize_for_path(study_cfg["name"])
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    root = Path(study_cfg.get("output_root", "tuning_runs")) / study_name / timestamp
    root.mkdir(parents=True, exist_ok=True)
    chunks_root = root / "chunks"
    chunks_root.mkdir(parents=True, exist_ok=True)

    manifest_path = root / "study_manifest.json"
    progress_path = root / "progress.json"
    leaderboard_csv = root / "leaderboard.csv"
    leaderboard_json = root / "leaderboard.json"
    config_snapshot_path = root / "effective_base_config.yaml"
    candidates_path = root / "candidates.json"

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
        chunks_root=chunks_root,
        config_snapshot_path=config_snapshot_path,
        candidates_path=candidates_path,
    )


def compose_base_config(repo_root: Path, runner_cfg: dict[str, Any], overrides: list[str]):
    import hydra

    config_name = runner_cfg.get("config_name", "default_ffhq.yaml")
    if config_name.endswith(".yaml"):
        config_name = Path(config_name).stem

    config_dir = str((repo_root / "configs").resolve())
    with hydra.initialize_config_dir(version_base="1.3", config_dir=config_dir):
        return hydra.compose(config_name=config_name, overrides=overrides)


def chunked(items: list[dict[str, Any]], chunk_size: int):
    for start in range(0, len(items), chunk_size):
        yield start // chunk_size, items[start:start + chunk_size]


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


def build_sigma_schedule_tensor(
    annealing_cfg: dict[str, Any],
    candidates: list[dict[str, Any]],
    max_iter: int,
    device,
    dtype,
):
    import torch
    from utils.diffusion import Scheduler

    schedules = []
    for candidate in candidates:
        sigma_cfg = dict(annealing_cfg)
        if "sampler.annealing_scheduler_config.sigma_max" in candidate:
            sigma_cfg["sigma_max"] = float(candidate["sampler.annealing_scheduler_config.sigma_max"])
        if "sampler.annealing_scheduler_config.sigma_min" in candidate:
            sigma_cfg["sigma_min"] = float(candidate["sampler.annealing_scheduler_config.sigma_min"])

        scheduler = Scheduler(**sigma_cfg)
        sigma_steps = scheduler.sigma_steps[:max_iter]
        if len(sigma_steps) < max_iter:
            raise ValueError(
                f"Annealing schedule has {len(sigma_steps)} steps, but max_iter={max_iter}"
            )
        schedules.append(torch.as_tensor(sigma_steps, device=device, dtype=dtype))
    return torch.stack(schedules, dim=0)


def tensorize_candidate_values(
    candidates: list[dict[str, Any]],
    resolved_cfg: dict[str, Any],
    key: str,
    device,
    dtype,
):
    import torch

    default_value = float(lookup_dotted(resolved_cfg, key))
    values = [float(candidate.get(key, default_value)) for candidate in candidates]
    return torch.as_tensor(values, device=device, dtype=dtype)


def evaluate_candidate_metrics(
    eval_fns: dict[str, Any],
    gt,
    measurement,
    samples,
) -> dict[str, dict[str, list[float] | list[list[float]]]]:
    import torch

    num_candidates, batch_size = samples.shape[:2]
    gt_flat = gt.unsqueeze(0).expand(num_candidates, *gt.shape).reshape(num_candidates * batch_size, *gt.shape[1:])
    sample_flat = samples.reshape(num_candidates * batch_size, *samples.shape[2:])

    measurement_flat: torch.Tensor | list[torch.Tensor]
    if torch.is_tensor(measurement):
        measurement_flat = measurement.unsqueeze(0).expand(
            num_candidates, *measurement.shape
        ).reshape(num_candidates * batch_size, *measurement.shape[1:])
    else:
        measurement_flat = measurement

    results: dict[str, dict[str, list[float] | list[list[float]]]] = {}
    for name, fn in eval_fns.items():
        values = fn(gt_flat, measurement_flat, sample_flat, reduction="none")
        values = values.reshape(num_candidates, batch_size, -1)
        if values.shape[-1] != 1:
            values = values.mean(dim=-1)
        else:
            values = values.squeeze(-1)

        results[name] = {
            "per_image": values.cpu().tolist(),
            "mean": values.mean(dim=1).cpu().tolist(),
            "std": (
                values.std(dim=1, unbiased=False)
                if batch_size > 1
                else torch.zeros_like(values.mean(dim=1))
            ).cpu().tolist(),
        }
    return results


def summarize_candidate_row(
    candidate_index: int,
    chunk_index: int,
    candidate: dict[str, Any],
    metric_results: dict[str, dict[str, list[float] | list[list[float]]]],
    scoring_cfg: dict[str, Any],
    local_idx: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "candidate_index": candidate_index,
        "chunk_index": chunk_index,
    }
    row.update(candidate)

    for metric_name, values in metric_results.items():
        row[f"{metric_name}_mean"] = float(values["mean"][local_idx])
        row[f"{metric_name}_std"] = float(values["std"][local_idx])

    primary_metric = scoring_cfg["primary_metric"]
    row["score"] = float(metric_results[primary_metric]["mean"][local_idx])
    return row


def sort_rows(rows: list[dict[str, Any]], scoring_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    comparator = resolve_comparator(scoring_cfg)
    reverse = comparator == "max"
    return sorted(rows, key=lambda row: row["score"], reverse=reverse)


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
            f"score={row['score']:.4f} "
            f"sigma_max={row.get('sampler.annealing_scheduler_config.sigma_max')} "
            f"sigma_min={row.get('sampler.annealing_scheduler_config.sigma_min')} "
            f"tau={row.get('inverse_task.admm_config.pdhg.tau')} "
            f"sigma_dual={row.get('inverse_task.admm_config.pdhg.sigma_dual')}"
        )


def update_progress(
    path: Path,
    *,
    status: str,
    total_candidates: int,
    completed_candidates: int,
    total_chunks: int,
    completed_chunks: int,
    started_at: float,
    current_chunk_index: int | None = None,
    rows: list[dict[str, Any]] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    elapsed_seconds = max(0.0, time.time() - started_at)
    avg_chunk_seconds = elapsed_seconds / completed_chunks if completed_chunks > 0 else None
    remaining_chunks = max(0, total_chunks - completed_chunks)
    eta_seconds = avg_chunk_seconds * remaining_chunks if avg_chunk_seconds is not None else None

    payload: dict[str, Any] = {
        "status": status,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": elapsed_seconds,
        "completed_candidates": completed_candidates,
        "total_candidates": total_candidates,
        "completed_chunks": completed_chunks,
        "total_chunks": total_chunks,
        "current_chunk_index": current_chunk_index,
        "eta_seconds": eta_seconds,
    }

    if rows:
        best_row = rows[0]
        payload["best_so_far"] = {
            "candidate_index": best_row["candidate_index"],
            "score": best_row["score"],
            "sigma_max": best_row.get("sampler.annealing_scheduler_config.sigma_max"),
            "sigma_min": best_row.get("sampler.annealing_scheduler_config.sigma_min"),
            "tau": best_row.get("inverse_task.admm_config.pdhg.tau"),
            "sigma_dual": best_row.get("inverse_task.admm_config.pdhg.sigma_dual"),
        }

    if extra:
        payload.update(extra)

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    cfg = load_yaml(config_path)

    candidates = expand_grid(cfg["grid"])
    if args.max_candidates is not None:
        candidates = candidates[:args.max_candidates]
    if not candidates:
        raise ValueError("The grid expansion produced no candidates")

    chunk_size = int(args.chunk_size or cfg["batched"]["candidate_chunk_size"])
    if chunk_size <= 0:
        raise ValueError("candidate_chunk_size must be positive")
    progress_update_every = int(cfg.get("batched", {}).get("progress_update_every", 25))

    study_paths = create_study_paths(cfg, config_path)
    with study_paths.candidates_path.open("w", encoding="utf-8") as handle:
        json.dump(candidates, handle, indent=2)

    total_chunks = ((len(candidates) - 1) // chunk_size) + 1
    print(f"Expanded {len(candidates)} candidates from {config_path}")
    print(f"Chunk size: {chunk_size} -> {total_chunks} chunks")

    if args.dry_run:
        started_at = time.time()
        update_progress(
            study_paths.progress_path,
            status="dry_run",
            total_candidates=len(candidates),
            completed_candidates=0,
            total_chunks=total_chunks,
            completed_chunks=0,
            current_chunk_index=None,
            started_at=started_at,
            rows=None,
        )
        preview = candidates[: min(5, len(candidates))]
        print("Preview candidates:")
        for candidate in preview:
            print(f"  {candidate}")
        return

    repo_root = REPO_ROOT
    runner_cfg = cfg["runner"]
    base_overrides = list(cfg.get("base_overrides", []))
    experiment_cfg = compose_base_config(repo_root, runner_cfg, base_overrides)
    from omegaconf import OmegaConf
    import torch
    from torch.utils.data import DataLoader

    resolved_cfg = OmegaConf.to_container(experiment_cfg, resolve=True)
    if not isinstance(resolved_cfg, dict):
        raise ValueError("Resolved Hydra config is not a mapping")

    with study_paths.config_snapshot_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(resolved_cfg, handle, sort_keys=False)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the batched PDHG tuning runner.")

    from datasets import get_dataset
    from measurements import get_operator
    from model import get_model
    from sampler import PDHG, get_sampler
    from utils import set_seed
    from utils.eval import get_eval_fn

    set_seed(int(resolved_cfg["seed"]))
    gpu_index = int(resolved_cfg["gpu"])
    torch.cuda.set_device(gpu_index)
    device = torch.device(f"cuda:{gpu_index}")

    dataset = get_dataset(**experiment_cfg.data)
    num_images = min(int(resolved_cfg["total_images"]), len(dataset))
    if num_images <= 0:
        raise RuntimeError("Dataset is empty after applying the configured slice.")

    dataloader = DataLoader(dataset, batch_size=num_images, shuffle=False)
    images = next(iter(dataloader)).to(device)

    operator = get_operator(**experiment_cfg.inverse_task.operator)
    measurements = operator.measure(images)

    model = get_model(**experiment_cfg.model)
    sampler = get_sampler(**experiment_cfg.sampler, **experiment_cfg.inverse_task)
    if not isinstance(sampler, PDHG):
        raise TypeError(f"Expected PDHG sampler, got {type(sampler).__name__}")

    eval_fns = {
        name: get_eval_fn(name)
        for name in resolved_cfg.get("eval_fn_list", ["psnr", "ssim", "lpips"])
    }

    scoring_cfg = cfg["scoring"]
    annealing_cfg = dict(resolved_cfg["sampler"]["annealing_scheduler_config"])
    max_iter = int(resolved_cfg["inverse_task"]["admm_config"]["max_iter"])
    dtype = images.dtype

    base_start_triplet = sampler.get_start(images)
    leaderboard_rows: list[dict[str, Any]] = []
    started_at = time.time()
    update_progress(
        study_paths.progress_path,
        status="running",
        total_candidates=len(candidates),
        completed_candidates=0,
        total_chunks=total_chunks,
        completed_chunks=0,
        current_chunk_index=None,
        started_at=started_at,
        rows=None,
    )

    try:
        for chunk_index, candidate_chunk in chunked(candidates, chunk_size):
            print(
                f"Running chunk {chunk_index + 1}/"
                f"{total_chunks} "
                f"with {len(candidate_chunk)} candidates..."
            )

            sigma_schedule = build_sigma_schedule_tensor(
                annealing_cfg=annealing_cfg,
                candidates=candidate_chunk,
                max_iter=max_iter,
                device=device,
                dtype=dtype,
            )
            tau_values = tensorize_candidate_values(
                candidates=candidate_chunk,
                resolved_cfg=resolved_cfg,
                key="inverse_task.admm_config.pdhg.tau",
                device=device,
                dtype=dtype,
            )
            sigma_dual_values = tensorize_candidate_values(
                candidates=candidate_chunk,
                resolved_cfg=resolved_cfg,
                key="inverse_task.admm_config.pdhg.sigma_dual",
                device=device,
                dtype=dtype,
            )

            def on_chunk_progress(*, step: int, num_steps: int, sigma_mean: float) -> None:
                sorted_so_far = sort_rows(leaderboard_rows, scoring_cfg) if leaderboard_rows else []
                completed_candidates = min(chunk_index * chunk_size, len(candidates))
                completed_chunks = chunk_index
                chunk_progress = step / max(1, num_steps)
                update_progress(
                    study_paths.progress_path,
                    status="running",
                    total_candidates=len(candidates),
                    completed_candidates=completed_candidates,
                    total_chunks=total_chunks,
                    completed_chunks=completed_chunks,
                    current_chunk_index=chunk_index,
                    started_at=started_at,
                    rows=sorted_so_far,
                    extra={
                        "current_chunk_iteration": step,
                        "current_chunk_total_iterations": num_steps,
                        "current_chunk_candidate_count": len(candidate_chunk),
                        "current_chunk_progress": chunk_progress,
                        "current_sigma_mean": sigma_mean,
                    },
                )

            samples = sampler.sample_hparam_candidates(
                model=model,
                ref_img=images,
                operator=operator,
                measurement=measurements,
                sigma_schedule_by_candidate=sigma_schedule,
                tau_by_candidate=tau_values,
                sigma_dual_by_candidate=sigma_dual_values,
                start_triplet=base_start_triplet,
                progress_callback=on_chunk_progress,
                progress_every=progress_update_every,
            )

            metric_results = evaluate_candidate_metrics(
                eval_fns=eval_fns,
                gt=images,
                measurement=measurements,
                samples=samples,
            )

            chunk_rows = []
            for local_idx, candidate in enumerate(candidate_chunk):
                candidate_index = chunk_index * chunk_size + local_idx
                row = summarize_candidate_row(
                    candidate_index=candidate_index,
                    chunk_index=chunk_index,
                    candidate=candidate,
                    metric_results=metric_results,
                    scoring_cfg=scoring_cfg,
                    local_idx=local_idx,
                )
                chunk_rows.append(row)
                leaderboard_rows.append(row)

            chunk_payload = {
                "chunk_index": chunk_index,
                "candidates": candidate_chunk,
                "rows": chunk_rows,
                "metrics": metric_results,
            }
            chunk_path = study_paths.chunks_root / f"chunk_{chunk_index:03d}.json"
            with chunk_path.open("w", encoding="utf-8") as handle:
                json.dump(chunk_payload, handle, indent=2)

            sorted_so_far = sort_rows(leaderboard_rows, scoring_cfg)
            completed_chunks = chunk_index + 1
            completed_candidates = min((chunk_index + 1) * chunk_size, len(candidates))
            elapsed = time.time() - started_at
            avg_chunk = elapsed / completed_chunks
            remaining_chunks = total_chunks - completed_chunks
            eta_seconds = avg_chunk * remaining_chunks
            print(
                f"Completed chunk {completed_chunks}/{total_chunks} "
                f"({completed_candidates}/{len(candidates)} candidates). "
                f"Elapsed: {elapsed / 60:.1f} min, ETA: {eta_seconds / 60:.1f} min."
            )
            if sorted_so_far:
                best = sorted_so_far[0]
                print(
                    "Current best: "
                    f"idx={best['candidate_index']} score={best['score']:.4f} "
                    f"sigma_max={best.get('sampler.annealing_scheduler_config.sigma_max')} "
                    f"sigma_min={best.get('sampler.annealing_scheduler_config.sigma_min')} "
                    f"tau={best.get('inverse_task.admm_config.pdhg.tau')} "
                    f"sigma_dual={best.get('inverse_task.admm_config.pdhg.sigma_dual')}"
                )
            update_progress(
                study_paths.progress_path,
                status="running",
                total_candidates=len(candidates),
                completed_candidates=completed_candidates,
                total_chunks=total_chunks,
                completed_chunks=completed_chunks,
                current_chunk_index=chunk_index,
                started_at=started_at,
                rows=sorted_so_far,
                extra={
                    "current_chunk_iteration": max_iter,
                    "current_chunk_total_iterations": max_iter,
                    "current_chunk_candidate_count": len(candidate_chunk),
                    "current_chunk_progress": 1.0,
                    "current_sigma_mean": None,
                },
            )

        sorted_rows = sort_rows(leaderboard_rows, scoring_cfg)
        write_csv(study_paths.leaderboard_csv, sorted_rows)
        with study_paths.leaderboard_json.open("w", encoding="utf-8") as handle:
            json.dump(sorted_rows, handle, indent=2)
        update_progress(
            study_paths.progress_path,
            status="completed",
            total_candidates=len(candidates),
            completed_candidates=len(candidates),
            total_chunks=total_chunks,
            completed_chunks=total_chunks,
            current_chunk_index=(total_chunks - 1) if total_chunks > 0 else None,
            started_at=started_at,
            rows=sorted_rows,
            extra={
                "current_chunk_iteration": max_iter,
                "current_chunk_total_iterations": max_iter,
                "current_chunk_candidate_count": 0,
                "current_chunk_progress": 1.0,
                "current_sigma_mean": None,
            },
        )
    except Exception:
        sorted_so_far = sort_rows(leaderboard_rows, scoring_cfg) if leaderboard_rows else []
        update_progress(
            study_paths.progress_path,
            status="failed",
            total_candidates=len(candidates),
            completed_candidates=len(leaderboard_rows),
            total_chunks=total_chunks,
            completed_chunks=len(leaderboard_rows) // chunk_size,
            current_chunk_index=None,
            started_at=started_at,
            rows=sorted_so_far,
            extra={
                "current_chunk_iteration": None,
                "current_chunk_total_iterations": max_iter,
                "current_chunk_candidate_count": 0,
                "current_chunk_progress": None,
                "current_sigma_mean": None,
            },
        )
        raise

    top_k = int(cfg.get("report", {}).get("top_k", 10))
    print_summary(sorted_rows, scoring_cfg, top_k=top_k)
    print(f"\nSaved leaderboard to {study_paths.leaderboard_csv}")


if __name__ == "__main__":
    main()
