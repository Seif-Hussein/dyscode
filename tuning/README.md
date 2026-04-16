# PDHG Tuning

This folder now contains two tuning paths:

- `run_pdhg_batched_grid.py`
  Preferred for strong GPUs. It evaluates many PDHG candidates in batched chunks
  inside one process, so the diffusion model, images, and measurements are loaded once.

- `run_pdhg_exact_grid.py`
  Faithful fallback. It launches one real `recover_inverse2.py` run per candidate,
  so the ranking path matches the normal experiment path exactly.

- `run_pdhg_tuning.py`
  Older staged fallback. It launches one Hydra run per candidate and is mainly
  useful when you want the original subprocess-style sweep behavior.

## Recommended Files

- `run_pdhg_batched_grid.py`
  Chunked candidate-batched grid runner.

- `pdhg_batched_grid.template.yaml`
  Batched search template aligned with the current `test2.sh` baseline.

- `run_pdhg_exact_grid.py`
  Exact subprocess-based grid runner.

- `pdhg_exact_grid.template.yaml`
  Exact search template aligned with the current `test2.sh` baseline.

## Batched Workflow

1. Edit `pdhg_batched_grid.template.yaml`.
2. Set the parameter lists under `grid`.
3. Adjust `batched.candidate_chunk_size` for the available GPU.
4. Dry-run the expansion:

```bash
python tuning/run_pdhg_batched_grid.py --config tuning/pdhg_batched_grid.template.yaml --dry-run
```

5. Launch the sweep:

```bash
python tuning/run_pdhg_batched_grid.py --config tuning/pdhg_batched_grid.template.yaml
```

6. Inspect the saved study:
   - `tuning_runs/.../leaderboard.csv`
   - `tuning_runs/.../leaderboard.json`
   - `tuning_runs/.../chunks/chunk_XXX.json`
   - `tuning_runs/.../current_chunk.json`
   - `tuning_runs/.../best_preview.png`
   - `tuning_runs/.../best_samples.pt`
   - `tuning_runs/.../best_candidate.json`
   - `tuning_runs/.../progress.json`

## Exact Workflow

Use this when you want the candidate ranking to match the normal
`recover_inverse2.py` path rather than the candidate-batched proxy.

1. Edit `pdhg_exact_grid.template.yaml`.
2. Dry-run the expansion:

```bash
python tuning/run_pdhg_exact_grid.py --config tuning/pdhg_exact_grid.template.yaml --dry-run
```

3. Launch the exact sweep:

```bash
python tuning/run_pdhg_exact_grid.py --config tuning/pdhg_exact_grid.template.yaml
```

4. Inspect the saved study:
   - `tuning_runs/.../leaderboard.csv`
   - `tuning_runs/.../leaderboard.json`
   - `tuning_runs/.../progress.json`
   - `tuning_runs/.../current_candidate.json`
   - `tuning_runs/.../runs/candidate_XXX/launcher.log`

## Current Default Search Space

- `sigma_max`: `20, 27, 30`
- `sigma_min`: `0.05, 0.075, 0.1`
- `tau`: `0.005, 0.0075, 0.01`
- `sigma_dual`: `800, 1200, 1600`

That is a full `3 x 3 x 3 x 3 = 81` candidate grid.

## Baseline Alignment

The batched template follows the current `test2.sh` algorithmic settings for
phase retrieval:

- `sampler=edm_pdhg`
- `inverse_task=phase_retrieval`
- `inverse_task.operator.sigma=0.05`
- `sampler.annealing_scheduler_config.num_steps=500`
- `inverse_task.admm_config.max_iter=500`
- `inverse_task.admm_config.denoise.final_step=tweedie`
- `inverse_task.admm_config.dys.gamma=0.0075`
- `inverse_task.admm_config.dys.lambda_schedule.activate=true`
- `inverse_task.admm_config.dys.lambda_schedule.start=1`
- `inverse_task.admm_config.dys.lambda_schedule.end=1`
- `inverse_task.admm_config.dys.lambda_schedule.warmup=0`
- `inverse_task.admm_config.denoise.lgvd.num_steps=0`

Artifact-heavy save options remain disabled so the sweep stays light.

The exact template uses the same baseline and search space, but it evaluates
each candidate via a separate `recover_inverse2.py` subprocess.

## Chunk Size Guidance

- Start conservatively if you do not yet know the GPU headroom.
- Increase `candidate_chunk_size` until GPU memory becomes a concern.
- Keep `total_images` fixed for fairness, and use chunk size to exploit the GPU.

On large GPUs, the main tuning control becomes chunk size rather than staging.

## Progress Tracking

During a live batched sweep, the runner updates `progress.json` after each
completed chunk and during the PDHG loop. It includes:

- current status: `running`, `completed`, or `failed`
- completed vs total chunks
- completed vs total candidates
- elapsed time and ETA
- the current best candidate and score
- current chunk iteration / chunk progress
- current chunk candidate list in `current_chunk.json`

The exact runner also writes:

- current candidate metadata in `current_candidate.json`
- partial `leaderboard.csv` / `leaderboard.json` after every finished candidate

The batched runner also keeps artifacts for the best-so-far candidate:

- `best_preview.png`
- `best_samples.pt`
- `best_candidate.json`

The Colab notebook includes cells to:

- launch the sweep in the background
- inspect the latest `progress.json`
- tail the latest log file

## Colab Installs

For Colab, do not run the full `requirements.txt`. It downgrades the runtime's
preinstalled PyTorch and notebook stack and can destabilize the session.

Use `requirements-colab.txt` instead. It only installs the small set of
packages needed for the batched phase-retrieval tuning workflow:

- `hydra-core`
- `omegaconf`
- `gdown`
- `piq`
- `prettytable`
- `setproctitle`
- `wandb`
- `imageio`
