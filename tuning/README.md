# PDHG Tuning Scaffold

This folder contains a small staged sweep driver for `recover_inverse2.py`.

## Files

- `run_pdhg_tuning.py`
  Launches Hydra runs, assigns each run its own `save_dir` and `hydra.run.dir`,
  collects `metrics.json`, and writes stage/global leaderboards.

- `pdhg_tuning.template.yaml`
  Stage-based tuning config. Replace the placeholder singleton values with the
  ranges you want to search.

## Typical Workflow

1. Edit `pdhg_tuning.template.yaml`.
2. Fill in or adjust the numeric ranges for the active stage.
3. Dry-run the generated commands:

```bash
python tuning/run_pdhg_tuning.py --config tuning/pdhg_tuning.template.yaml --dry-run
```

4. Launch one stage:

```bash
python tuning/run_pdhg_tuning.py --config tuning/pdhg_tuning.template.yaml --stage stage_a_steps
```

5. Inspect the outputs:
   - `tuning_runs/.../leaderboard.csv`
   - `tuning_runs/.../stages/<stage>/leaderboard.csv`
   - `results/tuning/.../<stage>/run_XXX/.../metrics.json`

## Notes

- The tuning baseline is aligned with the algorithmic settings in `test2.sh`
  for PDHG phase retrieval. The main difference is that the scaffold keeps
  artifact-heavy options such as `save_samples` disabled so sweeps stay lighter.
- The default stage ranges currently use:
  - `sigma_min`: `0.05, 0.075, 0.1`
  - `sigma_max`: `20, 27, 30`
  - `tau`: `0.005, 0.0075, 0.01`
  - `sigma_dual`: `800, 1200, 1600`
- The current scoring default is average PSNR using the same direction as the
  repo's evaluator conventions (`psnr`/`ssim`: maximize, `lpips`: minimize).
- Each launched run gets isolated output folders so repeated sweeps do not
  overwrite one another.
- The stage template is intentionally conservative. Once you share the actual
  numeric ranges, we can tighten the stages and add any task-specific overrides.
