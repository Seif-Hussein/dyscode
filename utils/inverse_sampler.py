# Code adapted from: https://github.com/zhangbingliang2019/DAPS/blob/main/posterior_sample.py
# Original author: bingliang

from .logging import Trajectory
import torch
from numbers import Real


def merge_metric_histories(histories, weights):
    """
        Merge per-batch metric histories into one weighted mean history.

        Histories are assumed to contain scalar-per-iteration lists. When
        different batches terminate at different iterations, later points only
        average over the batches that still contributed values.
    """
    valid = [
        (history, float(weight))
        for history, weight in zip(histories, weights)
        if history
    ]
    if not valid:
        return None

    ordered_keys = []
    seen = set()
    for history, _ in valid:
        for key, value in history.items():
            if isinstance(value, list) and key not in seen:
                seen.add(key)
                ordered_keys.append(key)

    merged = {}
    max_series_len = 0
    for history, _ in valid:
        hist_len = max((len(value) for value in history.values() if isinstance(value, list)), default=0)
        max_series_len = max(max_series_len, hist_len)

    if max_series_len > 0:
        num_samples = [0.0 for _ in range(max_series_len)]
        for history, weight in valid:
            hist_len = max((len(value) for value in history.values() if isinstance(value, list)), default=0)
            for idx in range(hist_len):
                num_samples[idx] += weight
        merged["num_samples"] = [int(round(value)) for value in num_samples]

    integer_keys = {"step"}
    for key in ordered_keys:
        weighted_sums = []
        weight_sums = []
        saw_numeric = False
        for history, weight in valid:
            series = history.get(key)
            if not isinstance(series, list):
                continue
            if len(series) > len(weighted_sums):
                extra = len(series) - len(weighted_sums)
                weighted_sums.extend([0.0] * extra)
                weight_sums.extend([0.0] * extra)
            for idx, value in enumerate(series):
                if value is None or not isinstance(value, Real):
                    continue
                weighted_sums[idx] += float(value) * weight
                weight_sums[idx] += weight
                saw_numeric = True
        if not saw_numeric:
            continue
        merged_series = [
            weighted_sum / weight_sum
            for weighted_sum, weight_sum in zip(weighted_sums, weight_sums)
            if weight_sum > 0
        ]
        if key in integer_keys:
            merged_series = [int(round(value)) for value in merged_series]
        merged[key] = merged_series

    return merged


def sample_in_batch(sampler,
                    model,
                    x_start,
                    operator,
                    y,
                    evaluator,
                    verbose,
                    record,
                    batch_size,
                    gt,
                    **kwargs):
    """
        posterior sampling in batch
    """
    samples = []
    trajs = []
    metric_histories = []
    batch_weights = []
    for s in range(0, len(x_start), batch_size):
        # update evaluator to correct batch index
        cur_x_start = x_start[s:s + batch_size]
        if type(y) in [tuple, list]:
            cur_y = tuple([y_part[s:s + batch_size] for y_part in y])
        else:
            cur_y = y[s:s+batch_size]
        
        cur_gt = gt[s: s + batch_size]
        cur_samples = sampler.sample(model, cur_x_start, operator, cur_y,
                                     evaluator, verbose=verbose, record=record,
                                     gt=cur_gt, **kwargs)

        samples.append(cur_samples)
        cur_metric_history = getattr(sampler, "metric_history", None)
        if cur_metric_history:
            metric_histories.append(cur_metric_history)
            batch_weights.append(len(cur_x_start))
        if record:
            trajs.append(sampler.trajectory.compile())
    if record:
        trajs = Trajectory.merge(trajs)
    merged_metric_history = merge_metric_histories(metric_histories, batch_weights)
    return torch.cat(samples, dim=0), trajs, merged_metric_history
