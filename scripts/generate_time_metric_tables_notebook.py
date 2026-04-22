from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "notebooks" / "time_metric_tables_colab.ipynb"


def md_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(text).lstrip("\n").splitlines(keepends=True),
    }


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(text).lstrip("\n").splitlines(keepends=True),
    }


def build_notebook() -> dict:
    cells = [
        md_cell(
            """
            # Time-Metric Tables In Colab

            This notebook runs a small FFHQ sweep for:
            - `PDHG`
            - `AC-DC-ADMM` (`sampler=edm_admm` with `lgvd.num_steps=10`)

            It then builds tables that pair actual measured time per image with:
            - `PSNR`
            - `SSIM`
            - `LPIPS`

            The notebook uses per-iteration sampler-side time from `metric_history.json`.

            Final output metrics follow the sampler's returned sample:
            - `PDHG`: `x_k`
            - `AC-DC-ADMM`: `z_k`

            The defaults are intentionally conservative so the notebook is usable in Colab.
            For paper-style reporting, increase `TOTAL_IMAGES` to `100`.
            """
        ),
        md_cell(
            """
            ## Runtime

            In Colab, go to `Runtime > Change runtime type` and choose:
            - `Python 3`
            - `GPU`

            Prefer `A100` when available.

            If you want DAPS-style *single-image* latency, keep `BATCH_SIZE=1`.
            If you use `BATCH_SIZE>1`, the tables become throughput-normalized seconds per image.
            """
        ),
        code_cell(
            """
            #@title Project Settings

            SETUP_MODE = "git"  #@param ["git", "drive_zip"]
            REPO_URL = "https://github.com/Seif-Hussein/dyscode.git"  #@param {type:"string"}
            REPO_BRANCH = "codex-pdhg-colab-light-100"  #@param {type:"string"}
            DRIVE_ZIP_PATH = "/content/drive/MyDrive/mycode2.zip"  #@param {type:"string"}

            REPO_DIR = "/content/mycode2"  #@param {type:"string"}
            PYTHON_BIN = "/usr/bin/python3"  #@param {type:"string"}
            DRIVE_EXPORT_DIR = "/content/drive/MyDrive/time_metric_tables_exports"  #@param {type:"string"}
            DRIVE_FFHQ_DATA_DIR = "/content/drive/MyDrive/mycode/test-ffhq"  #@param {type:"string"}

            STUDY_NAME = "Time_Metric_Tables"  #@param {type:"string"}
            SESSION_TAG = ""  #@param {type:"string"}
            CONFIG_NAME = "default_ffhq.yaml"  #@param ["default_ffhq.yaml"]
            INVERSE_TASK = "phase_retrieval"  #@param ["phase_retrieval", "phase_retrieval_explicit", "inpainting", "inpainting_explicit", "inpainting_rand", "inpainting_rand_explicit", "motion_blur", "motion_blur_explicit", "gaussian_blur", "gaussian_blur_explicit", "down_sampling", "down_sampling_explicit", "hdr", "hdr_explicit"]
            RUN_SERIES = "both"  #@param ["both", "pdhg", "admm"]

            SEED = 99  #@param {type:"integer"}
            TOTAL_IMAGES = 20  #@param {type:"integer"}
            BATCH_SIZE = 1  #@param {type:"integer"}
            DATA_START_IDX = 0  #@param {type:"integer"}

            PDHG_W_LIST = "50,100,200,400"  #@param {type:"string"}
            ADMM_W_LIST = "5,10,20,40"  #@param {type:"string"}

            PDHG_SIGMA_MAX = 27.0  #@param {type:"number"}
            PDHG_SIGMA_MIN = 0.075  #@param {type:"number"}
            PDHG_TAU = 0.01  #@param {type:"number"}
            PDHG_SIGMA_DUAL = 1600.0  #@param {type:"number"}

            ADMM_SIGMA_MAX = 10.0  #@param {type:"number"}
            ADMM_SIGMA_MIN = 0.1  #@param {type:"number"}
            ADMM_RHO = 100.0  #@param {type:"number"}
            ADMM_ML_LR = 0.1  #@param {type:"number"}
            ADMM_LGVD_NUM_STEPS = 10  #@param {type:"integer"}

            MEASUREMENT_SIGMA = 0.05  #@param {type:"number"}
            DENOISE_FINAL_STEP = "tweedie"  #@param ["tweedie", "ode"]
            DENOISER_AC_NOISE = True  #@param {type:"boolean"}

            EVAL_METRICS = "psnr;ssim;lpips"  #@param {type:"string"}
            SHOW_CONFIG = False  #@param {type:"boolean"}
            SAVE_SAMPLES = False  #@param {type:"boolean"}
            SAVE_TRAJ = False  #@param {type:"boolean"}
            SAVE_TRAJ_RAW_DATA = False  #@param {type:"boolean"}
            SKIP_COMPLETED = True  #@param {type:"boolean"}

            TIME_TABLE_ROWS = 8  #@param {type:"integer"}
            TIME_TABLE_TARGETS = ""  #@param {type:"string"}

            # Optional extra Hydra overrides, separated by semicolons.
            EXTRA_HYDRA_OVERRIDES = ""  #@param {type:"string"}
            """
        ),
        code_cell(
            """
            #@title Mount Google Drive
            from google.colab import drive
            drive.mount('/content/drive')
            """
        ),
        code_cell(
            """
            #@title Fetch The Repo
            import os
            import shutil
            import subprocess
            import zipfile
            from pathlib import Path

            repo_dir = Path(REPO_DIR)
            repo_dir.parent.mkdir(parents=True, exist_ok=True)
            os.chdir(repo_dir.parent)

            if repo_dir.exists():
                shutil.rmtree(repo_dir)

            if SETUP_MODE == "git":
                subprocess.run(
                    [
                        "git",
                        "clone",
                        "--branch",
                        REPO_BRANCH,
                        "--single-branch",
                        REPO_URL,
                        repo_dir.as_posix(),
                    ],
                    check=True,
                )
            elif SETUP_MODE == "drive_zip":
                zip_path = Path(DRIVE_ZIP_PATH)
                if not zip_path.exists():
                    raise FileNotFoundError(f"Zip file not found: {zip_path}")
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(repo_dir.parent)
                extracted_root = repo_dir.parent / zip_path.stem
                if extracted_root.exists() and extracted_root != repo_dir:
                    if repo_dir.exists():
                        shutil.rmtree(repo_dir)
                    extracted_root.rename(repo_dir)
            else:
                raise ValueError(f"Unsupported SETUP_MODE: {SETUP_MODE}")

            os.chdir(repo_dir)
            print(f"Repo ready: {repo_dir}")
            """
        ),
        code_cell(
            """
            #@title Install Colab Dependencies
            import os
            import subprocess

            os.chdir(REPO_DIR)
            subprocess.run([PYTHON_BIN, "-m", "pip", "install", "-q", "-r", "requirements-colab.txt"], check=True)
            print("Installed requirements-colab.txt")
            """
        ),
        code_cell(
            """
            #@title Download The FFHQ Checkpoint If Needed
            import os
            import subprocess
            from pathlib import Path

            os.chdir(REPO_DIR)
            ckpt_path = Path("pretrained-models/ffhq_10m.pt")
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)

            if ckpt_path.exists():
                print(f"Checkpoint already present: {ckpt_path}")
            else:
                subprocess.run(
                    [
                        "gdown",
                        "--id",
                        "1BGwhRWUoguF-D8wlZ65tf227gp3cDUDh",
                        "-O",
                        ckpt_path.as_posix(),
                    ],
                    check=True,
                )
                print(f"Downloaded checkpoint to: {ckpt_path}")
            """
        ),
        code_cell(
            """
            #@title Build The Sweep Study
            import json
            import os
            import shlex
            from datetime import datetime
            from pathlib import Path

            import pandas as pd
            from IPython.display import display

            os.chdir(REPO_DIR)
            repo_dir = Path(REPO_DIR)
            drive_data_dir = Path(DRIVE_FFHQ_DATA_DIR)
            if not drive_data_dir.exists():
                raise FileNotFoundError(f"FFHQ dataset path not found: {drive_data_dir}")

            def parse_text_list(text: str):
                pieces = []
                for chunk in str(text).replace("\\n", ",").replace(";", ",").split(","):
                    chunk = chunk.strip()
                    if chunk:
                        pieces.append(chunk)
                return pieces

            def parse_int_list(text: str):
                return [int(item) for item in parse_text_list(text)]

            def parse_float_list(text: str):
                return [float(item) for item in parse_text_list(text)]

            def sanitize(text: str) -> str:
                out = []
                for ch in text:
                    if ch.isalnum() or ch in "-_":
                        out.append(ch)
                    else:
                        out.append("_")
                return "".join(out).strip("_") or "item"

            metric_list = parse_text_list(EVAL_METRICS)
            if not metric_list:
                raise ValueError("EVAL_METRICS must contain at least one metric name.")

            run_series = str(RUN_SERIES).strip().lower()
            if run_series not in {"both", "pdhg", "admm"}:
                raise ValueError("RUN_SERIES must be one of: both, pdhg, admm")

            pdhg_w_values = parse_int_list(PDHG_W_LIST) if run_series in {"both", "pdhg"} else []
            admm_w_values = parse_int_list(ADMM_W_LIST) if run_series in {"both", "admm"} else []
            if run_series in {"both", "pdhg"} and not pdhg_w_values:
                raise ValueError("PDHG_W_LIST must contain at least one integer when RUN_SERIES includes PDHG.")
            if run_series in {"both", "admm"} and not admm_w_values:
                raise ValueError("ADMM_W_LIST must contain at least one integer when RUN_SERIES includes ADMM.")

            extra_overrides = parse_text_list(EXTRA_HYDRA_OVERRIDES)
            study_tag = SESSION_TAG.strip() or datetime.now().strftime("%Y%m%d-%H%M%S")
            study_slug = sanitize(f"{STUDY_NAME}_{INVERSE_TASK}_{study_tag}")
            study_root = repo_dir / "results" / "time_metric_tables" / study_slug
            run_root = study_root / "runs"
            study_root.mkdir(parents=True, exist_ok=True)
            run_root.mkdir(parents=True, exist_ok=True)

            summary_json_path = study_root / "final_summary.json"
            summary_csv_path = study_root / "final_summary.csv"
            anytime_json_path = study_root / "anytime_tables.json"
            anytime_csv_path = study_root / "anytime_tables.csv"
            context_path = study_root / "study_context.json"

            data_end_idx = DATA_START_IDX + TOTAL_IMAGES
            eval_fn_override = f"eval_fn_list=[{','.join(metric_list)}]"

            common_overrides = [
                f"seed={SEED}",
                "gpu=0",
                "wandb=false",
                f"show_config={'true' if SHOW_CONFIG else 'false'}",
                f"save_samples={'true' if SAVE_SAMPLES else 'false'}",
                f"save_traj={'true' if SAVE_TRAJ else 'false'}",
                f"save_traj_raw_data={'true' if SAVE_TRAJ_RAW_DATA else 'false'}",
                f"total_images={TOTAL_IMAGES}",
                f"batch_size={BATCH_SIZE}",
                "num_runs=1",
                f"inverse_task.operator.sigma={MEASUREMENT_SIGMA}",
                f"++inverse_task.admm_config.denoise.ac_noise={'true' if DENOISER_AC_NOISE else 'false'}",
                f"inverse_task.admm_config.denoise.final_step={DENOISE_FINAL_STEP}",
                eval_fn_override,
                f"data.image_root_path={drive_data_dir.as_posix()}",
                f"data.start_idx={DATA_START_IDX}",
                f"data.end_idx={data_end_idx}",
            ]

            def make_run_entry(series: str, sampler_config: str, W: int, method_overrides: list[str]):
                label = f"{series}-W{W}"
                run_slug = sanitize(label.lower())
                run_dir = run_root / run_slug
                run_save_dir = run_dir / "results"
                hydra_dir = run_dir / "hydra"
                log_path = run_dir / "run.log"
                run_name = sanitize(label.replace("-", "_"))

                command = [
                    PYTHON_BIN,
                    "recover_inverse2.py",
                    "--config-name",
                    CONFIG_NAME,
                    f"sampler={sampler_config}",
                    f"inverse_task={INVERSE_TASK}",
                    f"name={run_name}",
                    *common_overrides,
                    f"sampler.annealing_scheduler_config.num_steps={W}",
                    f"inverse_task.admm_config.max_iter={W}",
                    *method_overrides,
                    *extra_overrides,
                    f"save_dir={run_save_dir.as_posix()}",
                    f"hydra.run.dir={hydra_dir.as_posix()}",
                ]

                return {
                    "series": series,
                    "label": label,
                    "sampler_config": sampler_config,
                    "W": int(W),
                    "run_dir": run_dir.as_posix(),
                    "run_save_dir": run_save_dir.as_posix(),
                    "hydra_dir": hydra_dir.as_posix(),
                    "log_path": log_path.as_posix(),
                    "command": command,
                }

            plan = []
            if run_series in {"both", "pdhg"}:
                for W in pdhg_w_values:
                    plan.append(
                        make_run_entry(
                            series="PDHG",
                            sampler_config="edm_pdhg",
                            W=W,
                            method_overrides=[
                                f"sampler.annealing_scheduler_config.sigma_max={PDHG_SIGMA_MAX}",
                                f"sampler.annealing_scheduler_config.sigma_min={PDHG_SIGMA_MIN}",
                                "inverse_task.admm_config.denoise.lgvd.num_steps=0",
                                f"++inverse_task.admm_config.pdhg.tau={PDHG_TAU}",
                                f"++inverse_task.admm_config.pdhg.sigma_dual={PDHG_SIGMA_DUAL}",
                            ],
                        )
                    )

            if run_series in {"both", "admm"}:
                for W in admm_w_values:
                    plan.append(
                        make_run_entry(
                            series="AC-DC-ADMM",
                            sampler_config="edm_admm",
                            W=W,
                            method_overrides=[
                                f"sampler.annealing_scheduler_config.sigma_max={ADMM_SIGMA_MAX}",
                                f"sampler.annealing_scheduler_config.sigma_min={ADMM_SIGMA_MIN}",
                                f"inverse_task.admm_config.rho={ADMM_RHO}",
                                f"inverse_task.admm_config.ml.lr={ADMM_ML_LR}",
                                f"inverse_task.admm_config.denoise.lgvd.num_steps={ADMM_LGVD_NUM_STEPS}",
                            ],
                        )
                    )

            if not plan:
                raise ValueError("No runs were scheduled. Check RUN_SERIES and the W lists.")

            study_context = {
                "study_name": STUDY_NAME,
                "study_tag": study_tag,
                "run_series": run_series,
                "study_root": study_root.as_posix(),
                "run_root": run_root.as_posix(),
                "context_path": context_path.as_posix(),
                "summary_json_path": summary_json_path.as_posix(),
                "summary_csv_path": summary_csv_path.as_posix(),
                "anytime_json_path": anytime_json_path.as_posix(),
                "anytime_csv_path": anytime_csv_path.as_posix(),
                "metric_list": metric_list,
                "time_table_rows": int(TIME_TABLE_ROWS),
                "time_table_targets": TIME_TABLE_TARGETS,
                "skip_completed": bool(SKIP_COMPLETED),
                "plan": plan,
            }
            context_path.write_text(json.dumps(study_context, indent=2), encoding="utf-8")

            preview_df = pd.DataFrame(
                [
                    {
                        "series": entry["series"],
                        "label": entry["label"],
                        "W": entry["W"],
                        "run_save_dir": entry["run_save_dir"],
                    }
                    for entry in plan
                ]
            )
            display(preview_df)

            print(f"Study root: {study_root}")
            print(f"Context saved to: {context_path}")
            print()
            print("Example command:")
            print(" ".join(shlex.quote(part) for part in plan[0]["command"]))
            """
        ),
        code_cell(
            """
            #@title Run Or Resume The Sweep
            import json
            import os
            import subprocess
            from pathlib import Path

            import pandas as pd
            from IPython.display import display

            os.chdir(REPO_DIR)

            if "study_context" not in globals():
                raise RuntimeError("Run the build cell first so study_context is available.")

            context_path = Path(study_context["context_path"])
            if not context_path.exists():
                raise FileNotFoundError(f"Study context not found: {context_path}")
            context = json.loads(context_path.read_text(encoding="utf-8"))

            def load_json(path: Path):
                with path.open("r", encoding="utf-8") as handle:
                    return json.load(handle)

            def find_single(root: Path, filename: str) -> Path:
                matches = sorted(root.rglob(filename))
                if not matches:
                    raise FileNotFoundError(f"No {filename} found under {root}")
                if len(matches) > 1:
                    raise FileNotFoundError(f"Expected one {filename} under {root}, found {len(matches)}")
                return matches[0]

            def metric_mean(metrics: dict, metric_name: str) -> float | None:
                if metric_name not in metrics:
                    return None
                comparator = "min" if metric_name == "lpips" else "max"
                values = metrics[metric_name].get(comparator, [])
                if not values:
                    return None
                return float(sum(values) / len(values))

            def history_view(payload: dict):
                if isinstance(payload, dict) and isinstance(payload.get("runs"), list):
                    return payload["runs"][0] if payload["runs"] else {}
                return payload

            def final_time_from_metric_history(metric_history_payload: dict):
                view = history_view(metric_history_payload)
                if not isinstance(view, dict):
                    return None
                times = view.get("elapsed_seconds_per_image", [])
                if isinstance(times, list) and times:
                    return float(times[-1])
                return None

            def parse_float_list(text: str):
                pieces = []
                for chunk in str(text).replace("\\n", ",").replace(";", ",").split(","):
                    chunk = chunk.strip()
                    if chunk:
                        pieces.append(float(chunk))
                return pieces

            def choose_indices(times: list[float], num_rows: int, target_text: str):
                if not times:
                    return []

                explicit_targets = []
                if str(target_text).strip() and str(target_text).strip().lower() != "auto":
                    explicit_targets = parse_float_list(target_text)

                if explicit_targets:
                    chosen = []
                    seen = set()
                    for target in explicit_targets:
                        idx = min(range(len(times)), key=lambda i: abs(times[i] - target))
                        if idx not in seen:
                            seen.add(idx)
                            chosen.append(idx)
                    return chosen

                if len(times) <= num_rows:
                    return list(range(len(times)))

                return sorted(
                    {
                        round(i * (len(times) - 1) / max(1, num_rows - 1))
                        for i in range(num_rows)
                    }
                )

            def build_anytime_rows(metric_history_payload: dict, *, entry: dict, num_rows: int, target_text: str):
                view = history_view(metric_history_payload)
                if not isinstance(view, dict):
                    return []

                prefix = "x_k" if entry["series"] == "PDHG" else "z_k"
                times = view.get("elapsed_seconds_per_image", [])
                steps = view.get("step", [])
                psnr_series = view.get(f"{prefix}_psnr", [])
                ssim_series = view.get(f"{prefix}_ssim", [])
                lpips_series = view.get(f"{prefix}_lpips", [])

                lengths = [len(series) for series in [times, psnr_series, ssim_series, lpips_series] if isinstance(series, list)]
                if not lengths:
                    return []
                usable_len = min(lengths)
                if usable_len <= 0:
                    return []

                times = list(times[:usable_len])
                steps = list(steps[:usable_len]) if isinstance(steps, list) and steps else list(range(1, usable_len + 1))
                psnr_series = list(psnr_series[:usable_len])
                ssim_series = list(ssim_series[:usable_len])
                lpips_series = list(lpips_series[:usable_len])

                indices = choose_indices(times, num_rows=num_rows, target_text=target_text)
                rows = []
                for rank, idx in enumerate(indices, start=1):
                    rows.append(
                        {
                            "series": entry["series"],
                            "label": entry["label"],
                            "W": int(entry["W"]),
                            "checkpoint_rank": rank,
                            "step": int(steps[idx]),
                            "time_per_image_seconds": float(times[idx]),
                            "psnr": float(psnr_series[idx]),
                            "ssim": float(ssim_series[idx]),
                            "lpips": float(lpips_series[idx]),
                            "run_dir": entry["run_save_dir"],
                        }
                    )
                return rows

            summary_rows = []
            anytime_rows = []

            for run_number, entry in enumerate(context["plan"], start=1):
                run_dir = Path(entry["run_dir"])
                run_save_dir = Path(entry["run_save_dir"])
                log_path = Path(entry["log_path"])
                run_dir.mkdir(parents=True, exist_ok=True)
                run_save_dir.mkdir(parents=True, exist_ok=True)
                log_path.parent.mkdir(parents=True, exist_ok=True)

                print(f"[{run_number}/{len(context['plan'])}] {entry['label']}")
                print("  " + " ".join(entry["command"]))

                skipped_existing = False
                returncode = 0

                if bool(context.get("skip_completed", False)):
                    try:
                        existing_metrics_path = find_single(run_save_dir, "metrics.json")
                        existing_history_path = find_single(run_save_dir, "metric_history.json")
                        skipped_existing = True
                        print("  Reusing completed artifacts.")
                    except FileNotFoundError:
                        skipped_existing = False

                if not skipped_existing:
                    with log_path.open("w", encoding="utf-8") as handle:
                        proc = subprocess.run(
                            entry["command"],
                            cwd=REPO_DIR,
                            stdout=handle,
                            stderr=subprocess.STDOUT,
                            text=True,
                            check=False,
                        )
                    returncode = int(proc.returncode)

                    if returncode != 0:
                        summary_rows.append(
                            {
                                "series": entry["series"],
                                "label": entry["label"],
                                "sampler": entry["sampler_config"],
                                "W": int(entry["W"]),
                                "status": f"failed({returncode})",
                                "mode": "ran_now",
                                "time_per_image_seconds": None,
                                "psnr_mean": None,
                                "ssim_mean": None,
                                "lpips_mean": None,
                                "run_dir": entry["run_save_dir"],
                                "log_path": entry["log_path"],
                            }
                        )
                        print(f"  Failed with return code {returncode}. See log: {log_path}")
                        continue

                    existing_metrics_path = find_single(run_save_dir, "metrics.json")
                    existing_history_path = find_single(run_save_dir, "metric_history.json")

                metrics = load_json(existing_metrics_path)
                metric_history_payload = load_json(existing_history_path)
                time_per_image = final_time_from_metric_history(metric_history_payload)

                row = {
                    "series": entry["series"],
                    "label": entry["label"],
                    "sampler": entry["sampler_config"],
                    "W": int(entry["W"]),
                    "status": "ok",
                    "mode": "skipped_existing" if skipped_existing else "ran_now",
                    "time_per_image_seconds": float(time_per_image) if time_per_image is not None else None,
                    "psnr_mean": metric_mean(metrics, "psnr"),
                    "ssim_mean": metric_mean(metrics, "ssim"),
                    "lpips_mean": metric_mean(metrics, "lpips"),
                    "run_dir": entry["run_save_dir"],
                    "log_path": entry["log_path"],
                    "metrics_path": existing_metrics_path.as_posix(),
                    "metric_history_path": existing_history_path.as_posix(),
                }
                summary_rows.append(row)
                anytime_rows.extend(
                    build_anytime_rows(
                        metric_history_payload,
                        entry=entry,
                        num_rows=int(context["time_table_rows"]),
                        target_text=context.get("time_table_targets", ""),
                    )
                )

                print(
                    "  "
                    f"time/image={row['time_per_image_seconds']:.3f}s "
                    f"psnr={row['psnr_mean']:.4f} "
                    f"ssim={row['ssim_mean']:.4f} "
                    f"lpips={row['lpips_mean']:.4f}"
                )

            summary_df = pd.DataFrame(summary_rows)
            if not summary_df.empty:
                summary_df = summary_df.sort_values(["time_per_image_seconds", "series", "W"], na_position="last")
                summary_df.to_json(context["summary_json_path"], orient="records", indent=2)
                summary_df.to_csv(context["summary_csv_path"], index=False)

            anytime_df = pd.DataFrame(anytime_rows)
            if not anytime_df.empty:
                anytime_df = anytime_df.sort_values(["series", "W", "time_per_image_seconds"])
                anytime_df.to_json(context["anytime_json_path"], orient="records", indent=2)
                anytime_df.to_csv(context["anytime_csv_path"], index=False)

            print()
            print(f"Saved final summary to: {context['summary_json_path']}")
            print(f"Saved anytime tables to: {context['anytime_json_path']}")

            if not summary_df.empty:
                display(
                    summary_df[
                        [
                            "series",
                            "label",
                            "W",
                            "time_per_image_seconds",
                            "psnr_mean",
                            "ssim_mean",
                            "lpips_mean",
                            "status",
                            "mode",
                        ]
                    ]
                )
            """
        ),
        code_cell(
            """
            #@title Show Final Time-Vs-Metric Tables
            import json
            from pathlib import Path

            import pandas as pd
            from IPython.display import display

            if "study_context" not in globals():
                raise RuntimeError("Run the build cell first so study_context is available.")

            context_path = Path(study_context["context_path"])
            context = json.loads(context_path.read_text(encoding="utf-8"))
            summary_path = Path(context["summary_json_path"])
            if not summary_path.exists():
                raise FileNotFoundError(f"Final summary not found: {summary_path}")

            summary_rows = json.loads(summary_path.read_text(encoding="utf-8"))
            summary_df = pd.DataFrame(summary_rows)
            if summary_df.empty:
                print("Final summary is empty.")
            else:
                ok_df = summary_df[summary_df["status"] == "ok"].copy()
                if ok_df.empty:
                    print("No successful runs are available yet.")
                    display(summary_df)
                else:
                    ok_df["time_per_image_seconds"] = ok_df["time_per_image_seconds"].astype(float)
                    for key in ["psnr_mean", "ssim_mean", "lpips_mean"]:
                        ok_df[key] = ok_df[key].astype(float)
                    ok_df = ok_df.sort_values(["time_per_image_seconds", "series", "W"]).reset_index(drop=True)

                    display(
                        ok_df[
                            [
                                "series",
                                "label",
                                "W",
                                "time_per_image_seconds",
                                "psnr_mean",
                                "ssim_mean",
                                "lpips_mean",
                            ]
                        ].round(
                            {
                                "time_per_image_seconds": 4,
                                "psnr_mean": 4,
                                "ssim_mean": 4,
                                "lpips_mean": 4,
                            }
                        )
                    )

                    print()
                    print("PDHG:")
                    display(
                        ok_df[ok_df["series"] == "PDHG"][
                            [
                                "label",
                                "W",
                                "time_per_image_seconds",
                                "psnr_mean",
                                "ssim_mean",
                                "lpips_mean",
                            ]
                        ].round(
                            {
                                "time_per_image_seconds": 4,
                                "psnr_mean": 4,
                                "ssim_mean": 4,
                                "lpips_mean": 4,
                            }
                        )
                    )

                    print()
                    print("AC-DC-ADMM:")
                    display(
                        ok_df[ok_df["series"] == "AC-DC-ADMM"][
                            [
                                "label",
                                "W",
                                "time_per_image_seconds",
                                "psnr_mean",
                                "ssim_mean",
                                "lpips_mean",
                            ]
                        ].round(
                            {
                                "time_per_image_seconds": 4,
                                "psnr_mean": 4,
                                "ssim_mean": 4,
                                "lpips_mean": 4,
                            }
                        )
                    )

                print()
                print(f"JSON: {summary_path}")
                print(f"CSV:  {context['summary_csv_path']}")
            """
        ),
        code_cell(
            """
            #@title Show Anytime Metric-Vs-Time Tables
            import json
            from pathlib import Path

            import pandas as pd
            from IPython.display import display

            if "study_context" not in globals():
                raise RuntimeError("Run the build cell first so study_context is available.")

            context_path = Path(study_context["context_path"])
            context = json.loads(context_path.read_text(encoding="utf-8"))
            anytime_path = Path(context["anytime_json_path"])
            if not anytime_path.exists():
                raise FileNotFoundError(f"Anytime table file not found: {anytime_path}")

            anytime_rows = json.loads(anytime_path.read_text(encoding="utf-8"))
            anytime_df = pd.DataFrame(anytime_rows)
            if anytime_df.empty:
                print("Anytime table is empty.")
            else:
                anytime_df["time_per_image_seconds"] = anytime_df["time_per_image_seconds"].astype(float)
                anytime_df["psnr"] = anytime_df["psnr"].astype(float)
                anytime_df["ssim"] = anytime_df["ssim"].astype(float)
                anytime_df["lpips"] = anytime_df["lpips"].astype(float)
                anytime_df = anytime_df.sort_values(["series", "W", "time_per_image_seconds"]).reset_index(drop=True)

                for (series, label), df_run in anytime_df.groupby(["series", "label"], sort=False):
                    print()
                    print(f"{label}")
                    display(
                        df_run[
                            [
                                "step",
                                "time_per_image_seconds",
                                "psnr",
                                "ssim",
                                "lpips",
                            ]
                        ].round(
                            {
                                "time_per_image_seconds": 4,
                                "psnr": 4,
                                "ssim": 4,
                                "lpips": 4,
                            }
                        )
                    )

                print()
                print(f"JSON: {anytime_path}")
                print(f"CSV:  {context['anytime_csv_path']}")
            """
        ),
        code_cell(
            """
            #@title Copy The Study Back To Google Drive
            import json
            import shutil
            from pathlib import Path

            if "study_context" not in globals():
                raise RuntimeError("Run the build cell first so study_context is available.")

            context_path = Path(study_context["context_path"])
            context = json.loads(context_path.read_text(encoding="utf-8"))
            study_root = Path(context["study_root"])
            if not study_root.exists():
                raise FileNotFoundError(f"Study root not found: {study_root}")

            export_root = Path(DRIVE_EXPORT_DIR)
            export_root.mkdir(parents=True, exist_ok=True)
            target = export_root / study_root.name
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(study_root, target)
            print(f"Copied study to: {target}")
            """
        ),
        md_cell(
            """
            ## Notes

            Typical workflow:
            1. run the setup cells
            2. build the sweep study
            3. run or resume the sweep
            4. inspect the final time-vs-metric table
            5. inspect the per-run anytime tables
            6. copy the study folder back to Drive

            The notebook writes:
            - `final_summary.json`
            - `final_summary.csv`
            - `anytime_tables.json`
            - `anytime_tables.csv`
            """
        ),
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    notebook = build_notebook()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
