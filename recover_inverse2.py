import hydra
import json
import time
from datasets import get_dataset
from measurements import get_operator
from setproctitle import setproctitle
from utils import set_seed
import torch
from omegaconf import OmegaConf
from model import get_model
from sampler import get_sampler
from utils.inverse_sampler import sample_in_batch
from utils.eval import Evaluator, get_eval_fn
from utils.logging import log_results
import wandb
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# Few codes adapted from: https://github.com/zhangbingliang2019/DAPS/blob/main/posterior_sample.py
# Original author: bingliang
def _write_json_atomic(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)


@hydra.main(version_base="1.3", config_path="configs", config_name="default_ffhq.yaml")
def main(args):
    print("============================================================")
    print("Running the inverse task")
    print("============================================================")
    if args.show_config:
        print(OmegaConf.to_yaml(args))
        print("\n")

    '''Set the seed, gpu and process name'''
    set_seed(args.seed)
    torch.cuda.set_device(f'cuda:{args.gpu}')

    '''Init wandb if required'''
    if args.wandb:
        wandb.init(
            project=args.project_name,
            name=args.name,
            config=OmegaConf.to_container(args, resolve=True)
        )

    '''Get the dataset'''
    dataset = get_dataset(**args.data)

    '''Get the forward measurement operator'''
    operator = get_operator(**args.inverse_task.operator)

    '''Get image from the dataset'''
    total_number = min(args.total_images, len(dataset))
    dataloader = DataLoader(dataset, batch_size=args.total_images, shuffle=False)
    images = next(iter(dataloader)).to(f'cuda:{args.gpu}')
    y = operator.measure(images)

    '''Load the model'''
    model = get_model(**args.model)

    '''Load daps/admm sampler'''
    sampler = get_sampler(
        **args.sampler,
          **args.inverse_task)

    # get evaluator
    eval_fn_list = []
    for eval_fn_name in args.eval_fn_list:
        eval_fn_list.append(get_eval_fn(eval_fn_name))
    evaluator = Evaluator(eval_fn_list)

    # Recording full trajectories is extremely memory-heavy for long runs and
    # should only be enabled when the user explicitly asked to save them.
    record_trajectory = bool(args.save_traj or args.save_traj_raw_data)
    results_root = Path(args.save_dir) / f"{args.name}_{args.data.name}_{args.inverse_task.operator.name}"
    progress_json_override = getattr(args, "progress_json_path", None)
    progress_json_path = (
        Path(progress_json_override)
        if progress_json_override
        else results_root / "progress.json"
    )
    progress_json_every = max(1, int(getattr(args, "progress_json_every", 1)))
    total_batches = max(1, (total_number + int(args.batch_size) - 1) // int(args.batch_size))

    def write_progress(status: str, **payload):
        progress_payload = {
            "status": status,
            "name": str(args.name),
            "sampler": str(getattr(args.sampler, "name", "unknown")),
            "inverse_task": str(getattr(args.inverse_task.operator, "name", "unknown")),
            "data_name": str(getattr(args.data, "name", "unknown")),
            "num_runs": int(args.num_runs),
            "total_images": int(total_number),
            "batch_size": int(args.batch_size),
            "num_batches": int(total_batches),
            "updated_at_unix": float(time.time()),
            "progress_json_path": progress_json_path.as_posix(),
        }
        progress_payload.update(payload)
        _write_json_atomic(progress_json_path, progress_payload)

    write_progress(
        "starting",
        run_index=0,
        runs_completed=0,
        overall_fraction_complete=0.0,
    )

    try:
        # main sampling process
        full_samples = []
        full_trajs = []
        full_metric_histories = []
        per_run_sampling_seconds = []
        for r in range(args.num_runs):
            print(f'Run: {r}')
            write_progress(
                "running",
                run_index=int(r + 1),
                runs_completed=int(r),
                run_fraction_complete=0.0,
                overall_fraction_complete=float(r) / float(max(1, args.num_runs)),
            )
            sampling_started = time.time()

            def progress_callback(**payload):
                batch_index = int(payload.get("batch_index", 1))
                num_batches_local = int(payload.get("num_batches", total_batches))
                step = int(payload.get("step", 0))
                max_iter = int(payload.get("max_iter", 0))
                run_fraction_complete = (
                    float(step) / float(max_iter)
                    if max_iter > 0
                    else None
                )

                overall_fraction_complete = None
                if max_iter > 0 and num_batches_local > 0 and args.num_runs > 0:
                    completed_batch_units = max(0, batch_index - 1)
                    run_units = completed_batch_units + float(step) / float(max_iter)
                    overall_fraction_complete = (
                        (float(r) * float(num_batches_local) + run_units)
                        / float(args.num_runs * num_batches_local)
                    )

                write_progress(
                    "running",
                    run_index=int(r + 1),
                    runs_completed=int(r),
                    run_fraction_complete=run_fraction_complete,
                    overall_fraction_complete=overall_fraction_complete,
                    **payload,
                )

            samples, trajs, metric_history = sample_in_batch(
                sampler,
                model,
                images,
                operator,
                y,
                evaluator,
                verbose=True,
                record=record_trajectory,
                batch_size=args.batch_size,
                gt=images,
                wandb=args.wandb,
                progress_callback=progress_callback,
                progress_every=progress_json_every,
            )
            per_run_sampling_seconds.append(float(time.time() - sampling_started))
            full_samples.append(samples)
            if record_trajectory:
                full_trajs.append(trajs)
            if metric_history:
                full_metric_histories.append(metric_history)

            write_progress(
                "running",
                run_index=int(r + 1),
                runs_completed=int(r + 1),
                run_fraction_complete=1.0,
                overall_fraction_complete=float(r + 1) / float(max(1, args.num_runs)),
                elapsed_seconds_total=float(time.time() - sampling_started),
                elapsed_seconds_per_image=(
                    float(time.time() - sampling_started) / float(max(1, total_number))
                ),
            )
        full_samples = torch.stack(full_samples, dim=0)
        """trace = sampler.get_trace()  # dict[str, list[float]]
        if trace is not None and "sigma" in trace and "dual_inject_norm" in trace:
            sig = np.asarray(trace["sigma"], dtype=float)
            inj = np.asarray(trace["dual_inject_norm"], dtype=float)
            ratio = np.asarray(trace.get("dual_inject_over_sigma", []), dtype=float)
            it = np.arange(len(sig))

            plt.figure()
            plt.semilogy(it, sig, label="sigma")
            plt.semilogy(it, inj, label="dual_inject_norm")
            if ratio.size == sig.size:
                plt.semilogy(it, ratio, label="dual_inject_over_sigma")
            plt.legend()
            plt.xlabel("recorded iteration")
            plt.grid(True, which="both")
            plt.tight_layout()
            plt.savefig(f"trace_run{r}_sigma_vs_inject.png", dpi=200)
            plt.close()

            plt.figure()
            plt.loglog(sig, inj, ".-")
            plt.gca().invert_xaxis()  # optional: so sigma decreasing goes left->right
            plt.xlabel("sigma")
            plt.ylabel("dual_inject_norm")
            plt.grid(True, which="both")
            plt.tight_layout()
            plt.savefig(f"trace_run{r}_inject_vs_sigma.png", dpi=200)
            plt.close()"""

        # log metrics
        results = evaluator.report(images, y, full_samples)
        markdown_text = evaluator.display(results)
        print(markdown_text)

        # log results
        metric_history_artifact = None
        if full_metric_histories:
            metric_history_artifact = (
                full_metric_histories[0]
                if len(full_metric_histories) == 1
                else {"runs": full_metric_histories}
            )

        metric_history_time_per_image = None
        if metric_history_artifact is not None:
            history_views = (
                metric_history_artifact.get("runs", [])
                if isinstance(metric_history_artifact, dict) and isinstance(metric_history_artifact.get("runs"), list)
                else [metric_history_artifact]
            )
            elapsed_per_image_values = []
            for history_view in history_views:
                if not isinstance(history_view, dict):
                    continue
                elapsed_per_image = history_view.get("elapsed_seconds_per_image")
                if isinstance(elapsed_per_image, list) and elapsed_per_image:
                    elapsed_per_image_values.append(float(elapsed_per_image[-1]))
            if elapsed_per_image_values:
                metric_history_time_per_image = (
                    sum(elapsed_per_image_values) / len(elapsed_per_image_values)
                )

        total_sampling_seconds = float(sum(per_run_sampling_seconds))
        write_progress(
            "saving_results",
            run_index=int(args.num_runs),
            runs_completed=int(args.num_runs),
            run_fraction_complete=1.0,
            overall_fraction_complete=1.0,
            sampling_seconds_total_observed=total_sampling_seconds,
            sampling_seconds_per_image=metric_history_time_per_image,
        )
        log_results(args, full_trajs, results, images, y,
                    full_samples, markdown_text, total_number,
                    metric_history=metric_history_artifact)
        write_progress(
            "completed",
            run_index=int(args.num_runs),
            runs_completed=int(args.num_runs),
            run_fraction_complete=1.0,
            overall_fraction_complete=1.0,
            sampling_seconds_total_observed=total_sampling_seconds,
            sampling_seconds_per_image=metric_history_time_per_image,
            results_root=results_root.as_posix(),
            metric_history_path=(results_root / "metric_history.json").as_posix(),
            metrics_path=(results_root / "metrics.json").as_posix(),
        )
        if args.wandb:
            evaluator.log_wandb(results, args.batch_size)
            wandb.finish()
        print(f"Finish the inverse tasks {args.name}")
    except Exception as exc:
        write_progress(
            "failed",
            error=str(exc),
        )
        raise


if __name__ == "__main__":
    main()
