from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from .params import PARAM_NAMES

# describe()-style CSV with a "std" row per parameter, produced over the
# training set. Used to turn each parameter's raw MSE into a variance-
# normalized loss so no single parameter's scale dominates `total`.
_STATS_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "standardized_audio_files"
    / "training_set"
    / "visualization"
    / "parameter_value_stats_train.csv"
)

_param_variances: dict[Path, dict[str, float]] = {}


def _get_param_variances(stats_path: Path = _STATS_PATH) -> dict[str, float]:
    """Load and cache per-parameter variance (std**2) from the stats CSV."""
    stats_path = Path(stats_path).resolve()
    if stats_path not in _param_variances:
        df = pd.read_csv(stats_path, index_col=0)
        variances = {
            name: float(df.loc["std", name]) ** 2 for name in PARAM_NAMES
        }
        invalid = {
            name: value
            for name, value in variances.items()
            if not torch.isfinite(torch.tensor(value)) or value <= 0
        }
        if invalid:
            raise ValueError(
                f"Invalid parameter variances in {stats_path}: {invalid}"
            )
        _param_variances[stats_path] = variances
    return _param_variances[stats_path]


def compute_loss(
    model: torch.nn.Module,
    preds: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    stats_path: Path = _STATS_PATH,
) -> dict[str, torch.Tensor]:
    """Per‑parameter MSE, combined into a variance-normalized total.

    NaN positions within a parameter (genuine reference failures) are masked
    out. If a parameter has no valid reference values at all in this batch,
    its loss is NaN (not 0.0) so it neither falsely looks perfect nor
    contaminates `total` — total sums only over parameters that had data.
    Predictions are pooled to match each target's frame count when they differ.

    `losses[name]` stays a raw, un-normalized MSE for interpretable
    per-parameter logging/plotting (same units/scale as before). `total`
    instead sums each parameter's MSE divided by that parameter's training-
    set variance (i.e. its fraction of variance unexplained), so parameters
    on very different natural scales (e.g. loudness_zwtv ~O(10) vs.
    sii_ansi ~O(0.01)) contribute comparably to the gradient and to whatever
    is driving the LR scheduler, instead of the largest-scale parameters
    drowning out the rest.
    """
    device = next(model.parameters()).device
    variances = _get_param_variances(stats_path)
    losses = {}
    valid_losses = []

    for name in PARAM_NAMES:
        prediction, target = preds[name], targets[name]

        if prediction.shape[-1] != target.shape[-1]:
            prediction = F.adaptive_avg_pool1d(
                prediction.unsqueeze(1), target.shape[-1]
            ).squeeze(1)

        mask = ~torch.isnan(target)
        if not mask.any():
            # No valid reference values for this parameter in this batch
            # (e.g. tnr_ecma_perseg with no detected tones, or sii_ansi).
            # NaN signals "not evaluated" — 0.0 would falsely look perfect.
            losses[name] = torch.tensor(float("nan"), device=device)
            continue

        prediction_masked = prediction[mask]
        target_masked = target[mask]

        loss = ((prediction_masked - target_masked) ** 2).mean()
        losses[name] = loss
        valid_losses.append(loss / variances[name])

    if valid_losses:
        losses["total"] = torch.stack(valid_losses).sum()
    else:
        losses["total"] = torch.tensor(float("nan"), device=device)

    return losses
