from __future__ import annotations

import numpy as np
import pandas as pd
import torch


def unit_vector(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    norm = np.linalg.norm(x)
    return x / max(norm, eps)


def moment_dipole(W: np.ndarray, weights: np.ndarray, *, center_weights: bool = True) -> np.ndarray:
    """Charge-weighted embedding direction used by the original dipole moment."""
    W = np.asarray(W, dtype=float)
    weights = np.asarray(weights, dtype=float)
    charges = weights - weights.mean() if center_weights else weights
    return unit_vector((charges[:, None] * W).sum(axis=0))


def prepare_log_frequency_fit(W: np.ndarray, p: np.ndarray, *, exclude_zeros: bool = True):
    """Return W, p, centered W, and centered log-probabilities on usable tokens."""
    W = np.asarray(W, dtype=float)
    p = np.asarray(p, dtype=float)
    mask = p > 0 if exclude_zeros else np.ones_like(p, dtype=bool)
    if not np.any(mask):
        raise ValueError("No positive frequency entries.")

    W_fit = W[mask]
    p_fit = p[mask]
    p_fit = p_fit / p_fit.sum()
    W_centered = W_fit - W_fit.mean(axis=0, keepdims=True)
    ell = np.log(p_fit)
    ell = ell - ell.mean()
    return W_fit, p_fit, W_centered, ell, mask


def optimal_direction_for_target(W_centered: np.ndarray, target: np.ndarray, ridge_lambda=None):
    """Best unit direction d for beta * W_centered @ d ~= target."""
    W_centered = np.asarray(W_centered, dtype=float)
    target = np.asarray(target, dtype=float)
    gram = W_centered.T @ W_centered
    rhs = W_centered.T @ target
    d_model = W_centered.shape[1]
    if ridge_lambda is None:
        ridge_lambda = 1e-4 * float(np.trace(gram)) / max(d_model, 1)
    coef = np.linalg.solve(gram + ridge_lambda * np.eye(d_model), rhs)
    return unit_vector(coef), float(ridge_lambda)


def fit_direction_r2(W_centered: np.ndarray, target: np.ndarray, direction: np.ndarray):
    """Fit beta for a fixed direction and return R2, beta, prediction, residual."""
    direction = unit_vector(direction)
    wd = W_centered @ direction
    denom = float(wd @ wd)
    beta = float((wd @ target) / denom) if denom > 0 else np.nan
    prediction = beta * wd
    residual = target - prediction
    target_norm_sq = float(target @ target)
    r2 = 1.0 - float(residual @ residual) / target_norm_sq if target_norm_sq > 0 else np.nan
    return {
        "r2": r2,
        "beta": beta,
        "prediction": prediction,
        "residual": residual,
    }


def deflated_residual_fit(W_centered: np.ndarray, target: np.ndarray, direction: np.ndarray, ridge_lambda=None):
    """Remove direction from embedding space, then fit the remaining target residual."""
    first = fit_direction_r2(W_centered, target, direction)
    d = unit_vector(direction)
    W_deflated = W_centered @ (np.eye(W_centered.shape[1]) - np.outer(d, d))

    residual = first["residual"]
    d_residual, residual_ridge = optimal_direction_for_target(
        W_deflated,
        residual,
        ridge_lambda=ridge_lambda,
    )
    second = fit_direction_r2(W_deflated, residual, d_residual)
    total_residual = residual - second["prediction"]
    target_norm_sq = float(target @ target)
    total_r2 = (
        1.0 - float(total_residual @ total_residual) / target_norm_sq
        if target_norm_sq > 0
        else np.nan
    )

    return {
        "r2": first["r2"],
        "beta": first["beta"],
        "residual_r2": second["r2"],
        "residual_beta": second["beta"],
        "total_r2_after_residual": total_r2,
        "residual_direction": d_residual,
        "residual_ridge_lambda": residual_ridge,
    }


def frequency_direction_diagnostics(W: np.ndarray, p: np.ndarray, *, exclude_zeros: bool = True):
    """Compare d*, log-moment dipole, and probability-moment dipole on model W."""
    W_fit, p_fit, W_centered, ell, mask = prepare_log_frequency_fit(
        W,
        p,
        exclude_zeros=exclude_zeros,
    )

    d_star, ridge_lambda = optimal_direction_for_target(W_centered, ell)
    d_log = moment_dipole(W_fit, np.log(p_fit))
    d_prob = moment_dipole(W_fit, p_fit)

    rows = []
    for name, direction in {
        "d*": d_star,
        "d_log_tilde": d_log,
        "d_tilde": d_prob,
    }.items():
        stats = deflated_residual_fit(W_centered, ell, direction)
        rows.append(
            {
                "direction": name,
                "R2": stats["r2"],
                "beta": stats["beta"],
                "residual_R2": stats["residual_r2"],
                "residual_beta": stats["residual_beta"],
                "total_R2_after_residual": stats["total_r2_after_residual"],
            }
        )

    table = pd.DataFrame(rows)
    return {
        "table": table,
        "W_fit": W_fit,
        "p_fit": p_fit,
        "W_centered": W_centered,
        "ell": ell,
        "mask": mask,
        "d_star": d_star,
        "d_log_tilde": d_log,
        "d_tilde": d_prob,
        "ridge_lambda": ridge_lambda,
        "n_tokens": int(W_fit.shape[0]),
    }


def sentence_input_ids(sentence: str, tokenizer, device) -> torch.Tensor:
    """Tokenize a prompt, treating the empty string as a single BOS token."""
    if sentence == "":
        bos_id = tokenizer.bos_token_id
        if bos_id is None:
            raise ValueError("Tokenizer has no bos_token_id for empty prompt.")
        ids = [bos_id]
    else:
        ids = tokenizer.encode(sentence, add_special_tokens=True)
    return torch.tensor([ids], device=device, dtype=torch.long)


def last_hidden_vector(
    sentence: str,
    tokenizer,
    device,
    forward_backbone,
) -> np.ndarray:
    """Return the final-position hidden state used as the lm_head input."""
    input_ids = sentence_input_ids(sentence, tokenizer, device)
    with torch.no_grad():
        outputs = forward_backbone(input_ids)
    return outputs.last_hidden_state[0, -1].float().cpu().numpy()


def next_token_entropy(sentence: str, tokenizer, model, device) -> float:
    """Entropy in nats of the next-token softmax distribution."""
    input_ids = sentence_input_ids(sentence, tokenizer, device)
    with torch.no_grad():
        logits = model(input_ids).logits[0, -1].float()
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
    return float(-(probs * log_probs).sum().cpu())


def sequence_position_metrics(
    token_ids,
    model,
    device,
    forward_backbone,
    direction: np.ndarray,
    *,
    expected_len: int | None = None,
    return_hidden: bool = False,
):
    """Return per-position metrics, optionally including the raw hidden states."""
    input_ids = torch.tensor([token_ids], device=device, dtype=torch.long)
    with torch.no_grad():
        outputs = forward_backbone(input_ids)
        logits = model(input_ids).logits[0].float()

    y_by_pos = outputs.last_hidden_state[0].float().cpu().numpy()
    if expected_len is not None and y_by_pos.shape[0] != expected_len:
        raise RuntimeError(f"Expected {expected_len} hidden positions, got {y_by_pos.shape[0]}.")

    direction = unit_vector(direction)
    y_norms = np.linalg.norm(y_by_pos, axis=1)
    cos_vals = (y_by_pos @ direction) / np.maximum(y_norms, 1e-12)
    proj_vals = y_by_pos @ direction

    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    entropy_vals = (-(probs * log_probs).sum(dim=-1)).cpu().numpy()

    metrics = (cos_vals, proj_vals, y_norms, entropy_vals)
    if return_hidden:
        return (*metrics, y_by_pos)
    return metrics
