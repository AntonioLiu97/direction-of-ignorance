from __future__ import annotations

import json
import math
import random
import re
import string
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# In the self-contained direction_of_ignorance layout, utils.py sits next to
# this module inside scripts/.
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
import utils as repo_utils


DEFAULT_FEWSHOT_EXAMPLES = [
    {"question": "What is the largest planet in the Solar System?", "answer": "Jupiter"},
    {"question": "Who wrote the play Romeo and Juliet?", "answer": "William Shakespeare"},
    {"question": "What gas do plants absorb from the atmosphere?", "answer": "carbon dioxide"},
    {"question": "What is the chemical symbol for gold?", "answer": "Au"},
    {"question": "Which ocean is on the east coast of the United States?", "answer": "Atlantic Ocean"},
    {"question": "How many days are in a leap year?", "answer": "366"},
]

REFUSAL_PATTERNS = {
    "",
    "i dont know",
    "i do not know",
    "dont know",
    "do not know",
    "not sure",
    "unknown",
    "cannot answer",
    "cant answer",
    "i am not sure",
}

SPAN_SCALARS = [
    "entropy",
    "alpha_dhat",
    "alpha_dstar",
    "pmi_realized",
    "kl_to_prior",
    "cross_entropy_to_prior",
    "c_hat",
    "dark_mass_fraction",
    "norm_y",
    "alpha_dhat_x_norm_y",
]

STAGE1_MEASUREMENT_VERSION = "stage1_v5_repo_frequency_dipole_dstar"


@dataclass
class Directions:
    d_hat: torch.Tensor
    d_star: torch.Tensor
    unigram_logprior: torch.Tensor
    unigram_prior: torch.Tensor
    dark_projector: torch.Tensor
    unembedding_centered: torch.Tensor
    prior_log_centered: torch.Tensor
    prior_log_centered_norm_sq: torch.Tensor
    hidden_size: int
    vocab_size: int
    d_star_r2: float | None = None
    d_star_beta: float | None = None
    cos_dstar_dhat: float | None = None


def _hidden_size(model) -> int:
    for attr in ("hidden_size", "n_embd", "d_model"):
        value = getattr(model.config, attr, None)
        if value is not None:
            return int(value)
    return int(model.get_output_embeddings().weight.shape[1])


def _vocab_size(model) -> int:
    return int(model.get_output_embeddings().weight.shape[0])


def _resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if path.exists():
        return path

    base_dirs = [
        Path(__file__).resolve().parent,
        Path(__file__).resolve().parents[1],
        Path(__file__).resolve().parents[2],
    ]
    attempted = []
    for base_dir in base_dirs:
        candidate = base_dir / path
        attempted.append(candidate)
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Required artifact not found: {path_like}. Tried cwd, "
        + ", ".join(str(candidate) for candidate in attempted)
        + "."
    )


def _torch_load(path_like: str | Path) -> Any:
    return torch.load(_resolve_path(path_like), map_location="cpu")


def _first_present(mapping: dict[str, Any], keys: tuple[str, ...], *, path: str | Path):
    for key in keys:
        if key in mapping:
            return mapping[key]
    raise KeyError(f"{path} does not contain any of required keys: {keys}")


def _as_1d_tensor(value: Any, name: str, expected: int) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32).flatten()
    if tensor.numel() != expected:
        raise ValueError(f"{name} must have shape ({expected},), got {tuple(tensor.shape)}.")
    norm = torch.linalg.vector_norm(tensor)
    if not torch.isfinite(norm) or norm <= 0:
        raise ValueError(f"{name} has invalid norm {norm}.")
    return tensor / norm.clamp_min(1e-12)


def _as_prior_logprobs(value: Any, vocab_size: int, *, already_log: bool) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32).flatten()
    if tensor.numel() != vocab_size:
        raise ValueError(f"Unigram prior must have shape ({vocab_size},), got {tuple(tensor.shape)}.")
    if already_log:
        log_prior = tensor
    else:
        if torch.any(tensor < 0):
            raise ValueError("Unigram prior probabilities must be non-negative.")
        log_prior = torch.log(tensor.clamp_min(1e-45))
    log_prior = log_prior - torch.logsumexp(log_prior, dim=0)
    if not torch.all(torch.isfinite(log_prior)):
        raise ValueError("Unigram log-prior contains non-finite values after normalization.")
    return log_prior


def _as_dark_projector(value: Any, hidden_size: int) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.ndim != 2 or tensor.shape[0] != hidden_size:
        raise ValueError(
            "Dark projector/basis must have shape (hidden_size, k) or "
            f"(hidden_size, hidden_size); got {tuple(tensor.shape)} for hidden_size={hidden_size}."
        )
    return tensor


def load_trivia_qa_questions(n: int, seed: int) -> list[dict]:
    """
    Load TriviaQA validation questions and return seeded random sample dicts.

    The returned aliases are the union of answer.normalized_aliases and
    answer.normalized_value, with empty strings removed.
    """
    try:
        dataset = load_dataset("trivia_qa", "rc.nocontext", split="validation")
    except Exception:
        dataset = load_dataset("trivia_qa", "unfiltered.nocontext", split="validation")

    if n > len(dataset):
        raise ValueError(f"Requested {n} prompts but TriviaQA validation only has {len(dataset)} rows.")

    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    indices = indices[:n]
    rows = []
    for idx in indices:
        item = dataset[idx]
        answer = item.get("answer", {}) or {}
        aliases = set(answer.get("normalized_aliases") or [])
        normalized_value = answer.get("normalized_value")
        if normalized_value:
            aliases.add(normalized_value)
        aliases = sorted(alias for alias in aliases if alias)
        rows.append(
            {
                "question_id": str(item.get("question_id", idx)),
                "question": str(item["question"]),
                "aliases": aliases,
            }
        )
    return rows


def build_fewshot_prompt(question: str, fewshot_examples: list[dict]) -> tuple[str, int]:
    """Return a few-shot QA prompt and the character length of the few-shot prefix."""
    prefix_parts = []
    for example in fewshot_examples:
        prefix_parts.append(f"Q: {example['question']}\nA: {example['answer']}\n\n")
    prefix = "".join(prefix_parts)
    return f"{prefix}Q: {question}\nA:", len(prefix)


def load_model_and_tokenizer(model_id: str, dtype: str, device: str):
    """Load a causal LM and tokenizer in eval/no-grad mode."""
    dtype_map = {
        "float32": torch.float32,
        "float": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    torch_dtype = dtype_map.get(dtype.lower())
    if torch_dtype is None:
        raise ValueError(f"Unsupported dtype {dtype!r}; expected one of {sorted(dtype_map)}.")

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, tokenizer


def load_directions(direction_path, unigram_prior_path, dark_projector_path, model) -> Directions:
    """Load and validate all geometric/prior artifacts against the model shapes."""
    hidden_size = _hidden_size(model)
    vocab_size = _vocab_size(model)

    direction_obj = _torch_load(direction_path)
    if isinstance(direction_obj, dict):
        d_hat = _first_present(direction_obj, ("d_hat", "dhat", "d_tilde", "direction"), path=direction_path)
        d_star = _first_present(direction_obj, ("d_star", "d*", "d_opt", "dstar"), path=direction_path)
    elif isinstance(direction_obj, (tuple, list)) and len(direction_obj) >= 2:
        d_hat, d_star = direction_obj[:2]
    else:
        raise ValueError(f"{direction_path} must be a dict or tuple/list containing d_hat and d_star.")

    prior_obj = _torch_load(unigram_prior_path)
    if isinstance(prior_obj, dict):
        if any(key in prior_obj for key in ("unigram_logprior", "log_prior", "log_probs", "logprob")):
            prior_value = _first_present(
                prior_obj,
                ("unigram_logprior", "log_prior", "log_probs", "logprob"),
                path=unigram_prior_path,
            )
            log_prior = _as_prior_logprobs(prior_value, vocab_size, already_log=True)
        else:
            prior_value = _first_present(
                prior_obj,
                ("unigram_prior", "prior", "probs", "probabilities", "p"),
                path=unigram_prior_path,
            )
            log_prior = _as_prior_logprobs(prior_value, vocab_size, already_log=False)
    else:
        log_prior = _as_prior_logprobs(prior_obj, vocab_size, already_log=True)

    dark_obj = _torch_load(dark_projector_path)
    if isinstance(dark_obj, dict):
        dark_value = _first_present(
            dark_obj,
            ("dark_projector", "projector", "P_dark", "basis", "dark_basis"),
            path=dark_projector_path,
        )
    else:
        dark_value = dark_obj

    d_hat = _as_1d_tensor(d_hat, "d_hat", hidden_size)
    d_star = _as_1d_tensor(d_star, "d_star", hidden_size)
    dark_projector = _as_dark_projector(dark_value, hidden_size)
    unigram_prior = torch.exp(log_prior)

    w_u = model.get_output_embeddings().weight.detach().float().cpu()
    if w_u.shape != (vocab_size, hidden_size):
        raise ValueError(f"Unembedding has unexpected shape {tuple(w_u.shape)}.")
    unembedding_centered = w_u - w_u.mean(dim=0, keepdim=True)
    prior_log_centered = log_prior - log_prior.mean()
    prior_log_centered_norm_sq = (prior_log_centered @ prior_log_centered).clamp_min(1e-12)

    return Directions(
        d_hat=d_hat,
        d_star=d_star,
        unigram_logprior=log_prior,
        unigram_prior=unigram_prior,
        dark_projector=dark_projector,
        unembedding_centered=unembedding_centered,
        prior_log_centered=prior_log_centered,
        prior_log_centered_norm_sq=prior_log_centered_norm_sq,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        d_star_r2=None,
        d_star_beta=None,
        cos_dstar_dhat=None,
    )


def _unit_vector_torch(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / torch.linalg.vector_norm(x).clamp_min(eps)


def zipf_unigram_logprior(vocab_size: int, alpha: float = 1.0) -> torch.Tensor:
    """
    Analytic Zipf prior over token ids.

    This is a tutorial-friendly proxy when empirical unigram counts are not yet
    available. Token id order is treated as an approximate frequency ranking.
    """
    ranks = torch.arange(1, vocab_size + 1, dtype=torch.float32)
    log_prior = -float(alpha) * torch.log(ranks)
    return log_prior - torch.logsumexp(log_prior, dim=0)


def _safe_tag(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "__", text).strip("_") or "tag"


def tokenizer_cache_tag(tokenizer, *, vocab_size: int) -> str:
    # Content-based tag (no model name), so models sharing a tokenizer —
    # e.g. Llama-3.1 and Llama-3.2 — share one empirical frequency cache.
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None:
        fingerprint_source = backend.to_str()
    else:
        vocab = tokenizer.get_vocab()
        fingerprint_source = "\n".join(
            f"{idx}\t{token}" for token, idx in sorted(vocab.items(), key=lambda kv: kv[1])
        )
    digest = hashlib.sha1(fingerprint_source.encode("utf-8")).hexdigest()[:12]
    return f"tokenizer__vocab{int(vocab_size)}__{digest}"


def empirical_token_counts(
    tokenizer,
    *,
    vocab_size: int,
    coverage: float = 0.60,
    min_count: int = 2,
    dataset_name: str = "monology/pile-uncopyrighted",
    dataset_config: str | None = None,
    split: str = "train",
    text_column: str = "text",
    cache_dir: str | Path = "data/token_stats",
    max_examples: int | None = None,
    report_every: int = 1_000,
) -> np.ndarray:
    """
    Stream text and count tokens until coverage fraction has count >= min_count.

    The cache stores raw counts so future runs can resume and stricter coverage
    requests can continue from the best previous scan.
    """
    if not 0 < coverage <= 1:
        raise ValueError("coverage must be in the interval (0, 1].")
    if min_count < 1:
        raise ValueError("min_count must be >= 1.")

    target_covered = int(math.ceil(coverage * vocab_size))
    source_tag = _safe_tag(dataset_name if dataset_config is None else f"{dataset_name}_{dataset_config}")
    tok_tag = tokenizer_cache_tag(tokenizer, vocab_size=vocab_size)
    cache_path = Path(cache_dir) / f"{tok_tag}__{source_tag}__counts.npz"

    counts = np.zeros(vocab_size, dtype=np.int64)
    n_tokens = 0
    n_consumed_examples = 0
    n_counted_examples = 0

    if cache_path.exists():
        data = np.load(cache_path)
        cached_counts = np.asarray(data["counts"], dtype=np.int64)
        if cached_counts.shape != (vocab_size,):
            raise ValueError(
                f"Cached counts have shape {cached_counts.shape}, expected {(vocab_size,)}."
            )
        counts = cached_counts.copy()
        n_tokens = int(data["n_tokens"])
        n_consumed_examples = int(data["n_consumed_examples"])
        n_counted_examples = int(data["n_counted_examples"])
        n_covered = int((counts >= min_count).sum())
        print(
            f"Loaded empirical-count cache {cache_path}: "
            f"tokens={n_tokens:,}, count>={min_count} coverage={n_covered / vocab_size:.1%}"
        )
        if n_covered >= target_covered:
            return counts

    def save_cache() -> None:
        if n_tokens == 0:
            return
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            counts=counts,
            n_tokens=n_tokens,
            n_consumed_examples=n_consumed_examples,
            n_counted_examples=n_counted_examples,
            coverage_count_ge_min=float((counts >= min_count).sum() / vocab_size),
            min_count=int(min_count),
            dataset_name=dataset_name,
            dataset_config="" if dataset_config is None else dataset_config,
            split=split,
            text_column=text_column,
        )

    load_kwargs = {"split": split, "streaming": True}
    if dataset_config is None:
        dataset = load_dataset(dataset_name, **load_kwargs)
    else:
        dataset = load_dataset(dataset_name, dataset_config, **load_kwargs)

    for row_idx, example in enumerate(dataset):
        if row_idx < n_consumed_examples:
            continue
        n_consumed_examples = row_idx + 1

        text = example.get(text_column)
        if not text:
            continue

        ids = tokenizer.encode(str(text), add_special_tokens=False)
        if not ids:
            continue
        ids_arr = np.asarray(ids, dtype=np.int64)
        ids_arr = ids_arr[(0 <= ids_arr) & (ids_arr < vocab_size)]
        if ids_arr.size == 0:
            continue

        counts += np.bincount(ids_arr, minlength=vocab_size)
        n_tokens += int(ids_arr.size)
        n_counted_examples += 1

        n_covered = int((counts >= min_count).sum())
        if report_every and n_counted_examples % report_every == 0:
            print(
                f"counted_examples={n_counted_examples:,} tokens={n_tokens:,} "
                f"count>={min_count} coverage={n_covered / vocab_size:.1%}"
            )
            save_cache()

        if n_covered >= target_covered:
            break
        if max_examples is not None and n_counted_examples >= max_examples:
            break

    save_cache()
    n_covered = int((counts >= min_count).sum())
    if n_covered < target_covered:
        raise RuntimeError(
            f"Stopped at count>={min_count} coverage {n_covered / vocab_size:.1%} after "
            f"{n_counted_examples:,} counted examples; target was {coverage:.1%}. "
            f"Saved partial counts to {cache_path}."
        )
    return counts


def empirical_logprior_from_counts(
    counts: np.ndarray,
    *,
    coverage: float,
    min_count: int,
    probability_floor: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build a full-vocab log prior plus a fit mask from empirical counts.

    The fit mask keeps the most frequent tokens among those with count >=
    min_count, exactly up to the requested coverage fraction. Tokens outside
    the fit mask receive a tiny floor probability for KL/PMI stability, but are
    not used when fitting the prior direction.
    """
    counts = np.asarray(counts, dtype=np.int64)
    vocab_size = counts.size
    target_covered = int(math.ceil(coverage * vocab_size))
    eligible_ids = np.flatnonzero(counts >= min_count)
    if eligible_ids.size < target_covered:
        raise ValueError(
            f"Only {eligible_ids.size / vocab_size:.1%} tokens have count >= {min_count}; "
            f"need {coverage:.1%}."
        )

    if eligible_ids.size > target_covered:
        order = np.argsort(counts[eligible_ids], kind="stable")
        keep_ids = eligible_ids[order[-target_covered:]]
    else:
        keep_ids = eligible_ids

    fit_mask_np = np.zeros(vocab_size, dtype=bool)
    fit_mask_np[keep_ids] = True
    probs = np.full(vocab_size, float(probability_floor), dtype=np.float64)
    probs[fit_mask_np] = counts[fit_mask_np].astype(np.float64)
    probs /= probs.sum()
    return torch.log(torch.as_tensor(probs, dtype=torch.float32)), torch.as_tensor(fit_mask_np)


def empirical_prob_from_counts_for_fit(
    counts: np.ndarray,
    *,
    coverage: float,
    min_count: int,
) -> np.ndarray:
    """Return a probability vector with zeros outside the covered empirical support."""
    counts = np.asarray(counts, dtype=np.int64)
    vocab_size = counts.size
    target_covered = int(math.ceil(coverage * vocab_size))
    eligible_ids = np.flatnonzero(counts >= min_count)
    if eligible_ids.size < target_covered:
        raise ValueError(
            f"Only {eligible_ids.size / vocab_size:.1%} tokens have count >= {min_count}; "
            f"need {coverage:.1%}."
        )
    if eligible_ids.size > target_covered:
        order = np.argsort(counts[eligible_ids], kind="stable")
        keep_ids = eligible_ids[order[-target_covered:]]
    else:
        keep_ids = eligible_ids
    probs = np.zeros(vocab_size, dtype=np.float64)
    probs[keep_ids] = counts[keep_ids].astype(np.float64)
    probs /= probs.sum()
    return probs


def compute_directions_from_model(
    model,
    tokenizer=None,
    *,
    prior_kind: str = "empirical",
    zipf_alpha: float = 1.0,
    empirical_coverage: float = 0.60,
    empirical_min_count: int = 2,
    empirical_dataset_name: str = "monology/pile-uncopyrighted",
    empirical_dataset_config: str | None = None,
    empirical_split: str = "train",
    empirical_text_column: str = "text",
    empirical_cache_dir: str | Path = "data/token_stats",
    empirical_max_examples: int | None = None,
    dark_fraction: float = 0.05,
    ridge_lambda: float | None = None,
    directions_cache_path: str | Path | None = None,
    force_recompute: bool = False,
    verbose: bool = True,
) -> Directions:
    """
    Compute the Stage 1 directions directly from the loaded model.

    Quantities:
      - unigram prior: empirical token frequencies by default.
      - d_hat: the probability-moment dipole direction used to orient d_star,
        matching utils.frequency_dipole(..., method="optimal_prior").
      - d_star: ridge least-squares direction that best predicts centered
        log empirical probability from mean-centered unembedding rows, with
        sign flipped to agree with d_hat.
      - dark projector: bottom dark_fraction eigenspace of W_U^T W_U.
    """
    hidden_size = _hidden_size(model)
    vocab_size = _vocab_size(model)
    cache_path = Path(directions_cache_path) if directions_cache_path is not None else None

    if cache_path is not None and cache_path.exists() and not force_recompute:
        if verbose:
            print(f"Loading cached computed Directions from {cache_path}.")
        return load_computed_directions_cache(
            cache_path,
            model,
            expected_prior_kind=prior_kind,
            expected_coverage=empirical_coverage if prior_kind == "empirical" else None,
            expected_min_count=empirical_min_count if prior_kind == "empirical" else None,
            expected_direction_version="repo_frequency_dipole_optimal_prior_v1",
        )

    if verbose:
        print("Computing prior/geometric artifacts on the fly.")
        print(f"Model unembedding shape: vocab_size={vocab_size:,}, hidden_size={hidden_size:,}")

    if prior_kind == "empirical":
        if tokenizer is None:
            raise ValueError("tokenizer is required when prior_kind='empirical'.")
        if verbose:
            print(
                "Unigram prior: streaming empirical token counts until at least "
                f"{empirical_coverage:.0%} of token IDs have count >= {empirical_min_count}."
            )
        counts = empirical_token_counts(
            tokenizer,
            vocab_size=vocab_size,
            coverage=empirical_coverage,
            min_count=empirical_min_count,
            dataset_name=empirical_dataset_name,
            dataset_config=empirical_dataset_config,
            split=empirical_split,
            text_column=empirical_text_column,
            cache_dir=empirical_cache_dir,
            max_examples=empirical_max_examples,
        )
        p_for_dipole = empirical_prob_from_counts_for_fit(
            counts,
            coverage=empirical_coverage,
            min_count=empirical_min_count,
        )
        log_prior, fit_mask = empirical_logprior_from_counts(
            counts,
            coverage=empirical_coverage,
            min_count=empirical_min_count,
        )
    elif prior_kind == "zipf":
        log_prior = zipf_unigram_logprior(vocab_size, alpha=zipf_alpha)
        fit_mask = torch.ones(vocab_size, dtype=torch.bool)
        p_for_dipole = torch.exp(log_prior).cpu().numpy()
    else:
        raise ValueError(f"Unsupported on-the-fly prior_kind={prior_kind!r}.")

    unigram_prior = torch.exp(log_prior)
    fit_log_prior = log_prior[fit_mask]
    prior_log_centered_fit = fit_log_prior - fit_log_prior.mean()
    prior_log_centered = log_prior - log_prior.mean()
    prior_log_centered_norm_sq = (prior_log_centered @ prior_log_centered).clamp_min(1e-12)
    if verbose:
        entropy = float(-(unigram_prior * log_prior).sum())
        print(
            f"Prior distribution ready; entropy={entropy:.3f} nats, "
            f"fit tokens={int(fit_mask.sum()):,}/{vocab_size:,}."
        )
        print("The prior distribution is full-vocab for PMI/KL, but direction fitting uses only empirical fit tokens.")

    w_u = model.get_output_embeddings().weight.detach().float().cpu()
    if w_u.shape != (vocab_size, hidden_size):
        raise ValueError(f"Unembedding has unexpected shape {tuple(w_u.shape)}.")
    unembedding_centered = w_u - w_u.mean(dim=0, keepdim=True)
    if verbose:
        print("Mean-centered W_U across vocabulary rows.")

    # Match Bayesian_interp_dipole.ipynb exactly: use utils.frequency_dipole
    # with method="optimal_prior" and exclude_zeros=True. That routine fits
    # centered log-frequency and flips d_star to agree with its probability
    # moment dipole d_tilde.
    optimal_prior_result = repo_utils.frequency_dipole(
        w_u.cpu().numpy(),
        p_for_dipole,
        method="optimal_prior",
        ridge_lambda=ridge_lambda,
        exclude_zeros=True,
    )
    d_star = torch.as_tensor(optimal_prior_result["d_star"], dtype=torch.float32)
    d_hat = torch.as_tensor(
        repo_utils.unit_vector(optimal_prior_result["d_tilde"]),
        dtype=torch.float32,
    )
    beta = float(optimal_prior_result["beta_star"])
    r2 = float(optimal_prior_result["r2"])
    ridge_lambda = float(optimal_prior_result["ridge_lambda"])
    cos_dstar_dhat = float(optimal_prior_result["cos_d_star_d_tilde"])
    if verbose:
        print(
            "d_star is computed by repo utils.frequency_dipole(..., method='optimal_prior'), "
            "matching Bayesian_interp_dipole.ipynb. "
            f"ridge_lambda={ridge_lambda:.4g}, fit R^2={r2:.4f}, "
            f"cos(d_star,d_hat)={cos_dstar_dhat:.4f}."
        )

    gram = unembedding_centered.T @ unembedding_centered
    if not (0.0 < dark_fraction <= 1.0):
        raise ValueError(f"dark_fraction must be in (0, 1], got {dark_fraction}.")
    dark_k = max(1, int(math.ceil(hidden_size * dark_fraction)))
    if verbose:
        print(
            f"Dark subspace: eigendecomposing W_U^T W_U and taking the bottom "
            f"{dark_k} directions ({100 * dark_fraction:.1f}% of hidden dims)."
        )
    eigvals, eigvecs = torch.linalg.eigh(gram)
    dark_projector = eigvecs[:, :dark_k].contiguous()
    if verbose:
        bottom = float(eigvals[0])
        cutoff = float(eigvals[dark_k - 1])
        top = float(eigvals[-1])
        print(
            f"Dark basis ready: shape={tuple(dark_projector.shape)}, "
            f"eigval range bottom/cutoff/top={bottom:.4g}/{cutoff:.4g}/{top:.4g}."
        )

    directions = Directions(
        d_hat=d_hat,
        d_star=d_star,
        unigram_logprior=log_prior,
        unigram_prior=unigram_prior,
        dark_projector=dark_projector,
        unembedding_centered=unembedding_centered,
        prior_log_centered=prior_log_centered,
        prior_log_centered_norm_sq=prior_log_centered_norm_sq,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        d_star_r2=r2,
        d_star_beta=beta,
        cos_dstar_dhat=cos_dstar_dhat,
    )
    if cache_path is not None:
        save_computed_directions_cache(directions, cache_path, metadata={
            "prior_kind": prior_kind,
            "zipf_alpha": zipf_alpha,
            "empirical_coverage": empirical_coverage,
            "empirical_min_count": empirical_min_count,
            "empirical_dataset_name": empirical_dataset_name,
            "empirical_dataset_config": empirical_dataset_config,
            "empirical_split": empirical_split,
            "empirical_text_column": empirical_text_column,
            "dark_fraction": dark_fraction,
            "ridge_lambda": ridge_lambda,
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "direction_version": "repo_frequency_dipole_optimal_prior_v1",
        })
        if verbose:
            print(f"Saved computed Directions cache to {cache_path}.")
    return directions


def save_computed_directions_cache(
    directions: Directions,
    path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Persist an on-the-fly Directions bundle for reuse in later notebook runs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "d_hat": directions.d_hat.cpu(),
            "d_star": directions.d_star.cpu(),
            "unigram_logprior": directions.unigram_logprior.cpu(),
            "unigram_prior": directions.unigram_prior.cpu(),
            "dark_projector": directions.dark_projector.cpu(),
            "unembedding_centered": directions.unembedding_centered.cpu(),
            "prior_log_centered": directions.prior_log_centered.cpu(),
            "prior_log_centered_norm_sq": directions.prior_log_centered_norm_sq.cpu(),
            "hidden_size": directions.hidden_size,
            "vocab_size": directions.vocab_size,
            "d_star_r2": directions.d_star_r2,
            "d_star_beta": directions.d_star_beta,
            "cos_dstar_dhat": directions.cos_dstar_dhat,
            "metadata": {} if metadata is None else metadata,
        },
        path,
    )


def _validate_directions_cache_metadata(
    metadata: dict[str, Any],
    *,
    expected_prior_kind: str | None = None,
    expected_coverage: float | None = None,
    expected_min_count: int | None = None,
    expected_direction_version: str | None = None,
) -> None:
    if expected_prior_kind is not None and metadata.get("prior_kind") != expected_prior_kind:
        raise ValueError(
            f"Directions cache prior_kind mismatch: cache has {metadata.get('prior_kind')!r}, "
            f"expected {expected_prior_kind!r}."
        )
    if expected_coverage is not None:
        cached = metadata.get("empirical_coverage")
        if cached is None or not math.isclose(float(cached), float(expected_coverage), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"Directions cache empirical_coverage mismatch: cache has {cached!r}, "
                f"expected {expected_coverage!r}."
            )
    if expected_min_count is not None:
        cached = metadata.get("empirical_min_count")
        if cached is None or int(cached) != int(expected_min_count):
            raise ValueError(
                f"Directions cache empirical_min_count mismatch: cache has {cached!r}, "
                f"expected {expected_min_count!r}."
            )
    if expected_direction_version is not None and metadata.get("direction_version") != expected_direction_version:
        raise ValueError(
            f"Directions cache direction_version mismatch: cache has {metadata.get('direction_version')!r}, "
            f"expected {expected_direction_version!r}. Set FORCE_RECOMPUTE_DIRECTIONS=True to rebuild it."
        )


def load_computed_directions_cache(
    path: str | Path,
    model,
    *,
    expected_prior_kind: str | None = None,
    expected_coverage: float | None = None,
    expected_min_count: int | None = None,
    expected_direction_version: str | None = None,
) -> Directions:
    """Load a Directions cache created by save_computed_directions_cache."""
    hidden_size = _hidden_size(model)
    vocab_size = _vocab_size(model)
    obj = torch.load(Path(path), map_location="cpu")
    metadata = obj.get("metadata", {})
    _validate_directions_cache_metadata(
        metadata,
        expected_prior_kind=expected_prior_kind,
        expected_coverage=expected_coverage,
        expected_min_count=expected_min_count,
        expected_direction_version=expected_direction_version,
    )
    if int(obj["hidden_size"]) != hidden_size or int(obj["vocab_size"]) != vocab_size:
        raise ValueError(
            f"Directions cache shape mismatch: cache hidden/vocab="
            f"{obj['hidden_size']}/{obj['vocab_size']}, model hidden/vocab={hidden_size}/{vocab_size}."
        )
    d_star_r2 = obj.get("d_star_r2")
    d_star_beta = obj.get("d_star_beta")
    cos_dstar_dhat = obj.get("cos_dstar_dhat")
    if d_star_r2 is None or d_star_beta is None or cos_dstar_dhat is None:
        d_hat = _as_1d_tensor(obj["d_hat"], "d_hat", hidden_size)
        d_star = _as_1d_tensor(obj["d_star"], "d_star", hidden_size)
        w_centered = torch.as_tensor(obj["unembedding_centered"], dtype=torch.float32)
        log_prior = _as_prior_logprobs(obj["unigram_logprior"], vocab_size, already_log=True)
        metadata = obj.get("metadata", {})
        coverage = metadata.get("empirical_coverage")
        min_count = metadata.get("empirical_min_count")
        if coverage is not None and min_count is not None:
            probs = torch.exp(log_prior)
            target_n = int(math.ceil(float(coverage) * vocab_size))
            eligible = torch.nonzero(probs > 0, as_tuple=False).flatten()
            if eligible.numel() > target_n:
                keep = eligible[torch.argsort(probs[eligible])[-target_n:]]
            else:
                keep = eligible
            fit_mask = torch.zeros(vocab_size, dtype=torch.bool)
            fit_mask[keep] = True
        else:
            fit_mask = torch.ones(vocab_size, dtype=torch.bool)
        w_fit = w_centered[fit_mask]
        ell = log_prior[fit_mask]
        ell = ell - ell.mean()
        wd = w_fit @ d_star
        d_star_beta = float((wd @ ell) / (wd @ wd).clamp_min(1e-12))
        residual = ell - d_star_beta * wd
        d_star_r2 = 1.0 - float((residual @ residual) / (ell @ ell).clamp_min(1e-12))
        cos_dstar_dhat = float(torch.dot(d_star, d_hat))
    return Directions(
        d_hat=_as_1d_tensor(obj["d_hat"], "d_hat", hidden_size),
        d_star=_as_1d_tensor(obj["d_star"], "d_star", hidden_size),
        unigram_logprior=_as_prior_logprobs(obj["unigram_logprior"], vocab_size, already_log=True),
        unigram_prior=torch.exp(_as_prior_logprobs(obj["unigram_logprior"], vocab_size, already_log=True)),
        dark_projector=_as_dark_projector(obj["dark_projector"], hidden_size),
        unembedding_centered=torch.as_tensor(obj["unembedding_centered"], dtype=torch.float32),
        prior_log_centered=torch.as_tensor(obj["prior_log_centered"], dtype=torch.float32),
        prior_log_centered_norm_sq=torch.as_tensor(obj["prior_log_centered_norm_sq"], dtype=torch.float32),
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        d_star_r2=d_star_r2,
        d_star_beta=d_star_beta,
        cos_dstar_dhat=cos_dstar_dhat,
    )


def _newline_token_ids(tokenizer) -> set[int]:
    ids = set()
    for text in ("\n", "\n\n"):
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if len(encoded) == 1:
            ids.add(int(encoded[0]))
    return ids


@torch.inference_mode()
def generate_answer(model, tokenizer, prompt: str, max_new_tokens: int = 16) -> dict:
    """Greedy decode an answer and stop if a newline token is emitted."""
    device = next(model.parameters()).device
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = encoded["input_ids"].to(device)
    prompt_len = int(input_ids.shape[1])
    generated = input_ids
    newline_ids = _newline_token_ids(tokenizer)
    eos_id = tokenizer.eos_token_id

    for _ in range(max_new_tokens):
        logits = model(generated).logits[:, -1, :]
        next_id = torch.argmax(logits, dim=-1, keepdim=True)
        generated = torch.cat([generated, next_id], dim=1)
        token = int(next_id.item())
        if token in newline_ids or (eos_id is not None and token == eos_id):
            break

    answer_ids = generated[0, prompt_len:].detach().cpu().tolist()
    stop_ids = set(newline_ids)
    if eos_id is not None:
        stop_ids.add(int(eos_id))
    while answer_ids and int(answer_ids[-1]) in stop_ids:
        answer_ids.pop()

    return {
        "generated_text": tokenizer.decode(answer_ids, skip_special_tokens=True).strip(),
        "generated_token_ids": [int(token_id) for token_id in answer_ids],
        "full_input_ids": generated[0].detach().cpu(),
        "prompt_len": prompt_len,
    }


def normalize_answer(s: str) -> str:
    """TriviaQA/SQuAD-style exact-match normalization."""
    text = s.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def label_answer(generated_text: str, aliases: list[str]) -> dict:
    normalized_generated = normalize_answer(generated_text)
    alias_pairs = [(raw_alias, normalize_answer(raw_alias)) for raw_alias in aliases]
    alias_pairs = [(raw_alias, alias) for raw_alias, alias in alias_pairs if alias]
    abstained = normalized_generated in REFUSAL_PATTERNS

    matched_alias = None
    correct_strict = False
    for raw_alias, alias in alias_pairs:
        if normalized_generated == alias:
            matched_alias = raw_alias
            correct_strict = True
            break

    correct_loose = correct_strict
    if not correct_loose and normalized_generated:
        for raw_alias, alias in alias_pairs:
            if alias and alias in normalized_generated:
                matched_alias = raw_alias
                correct_loose = True
                break

    return {
        "correct": bool(correct_loose),
        "correct_strict": bool(correct_strict),
        "correct_loose": bool(correct_loose),
        "matched_alias": matched_alias,
        "abstained": bool(abstained),
    }


def _final_norm_module(model):
    candidates = [
        ("model", "norm"),
        ("model", "final_layernorm"),
        ("model", "final_layer_norm"),
        ("gpt_neox", "final_layer_norm"),
        ("transformer", "ln_f"),
    ]
    for parent_attr, norm_attr in candidates:
        parent = getattr(model, parent_attr, None)
        if parent is not None:
            norm = getattr(parent, norm_attr, None)
            if norm is not None:
                return norm
    for attr in ("norm", "final_layer_norm", "ln_f"):
        norm = getattr(model, attr, None)
        if norm is not None:
            return norm
    raise AttributeError("Could not locate model final norm module.")


def _apply_final_norm(norm_module, hidden_state: torch.Tensor) -> torch.Tensor:
    norm_device = next(norm_module.parameters(), hidden_state).device
    return norm_module(hidden_state.to(norm_device)).float()


def _dark_mass_fraction(y: torch.Tensor, dark_projector: torch.Tensor) -> torch.Tensor:
    projector = dark_projector.to(y.device, dtype=y.dtype)
    if projector.shape[0] == projector.shape[1]:
        projected = y @ projector.T
        numerator = (projected * projected).sum(dim=-1)
    else:
        coeffs = y @ projector
        numerator = (coeffs * coeffs).sum(dim=-1)
    denominator = (y * y).sum(dim=-1).clamp_min(1e-12)
    return numerator / denominator


def _region_for_position(position: int, prompt_len: int, answer_onset_char_token: int | None = None) -> str:
    if position >= prompt_len - 1:
        return "answer"
    if answer_onset_char_token is not None and position >= answer_onset_char_token:
        return "question"
    return "fewshot"


@torch.inference_mode()
def score_sequence(
    model,
    tokenizer,
    directions: Directions,
    full_input_ids: torch.Tensor,
    prompt_len: int,
    *,
    question_id: str | None = None,
    prefix_token_len: int | None = None,
    compute_per_layer: bool = False,
) -> pd.DataFrame:
    """
    Score one generated sequence with one row per prediction position.

    We apply the model's final norm module manually to every hidden state before
    computing geometry. This treats the normalized vector as y. For intermediate
    layers this is a comparability convention, not the architecture's native
    computation path.
    """
    device = next(model.parameters()).device
    input_ids = full_input_ids.to(device=device, dtype=torch.long).view(1, -1)
    outputs = model(input_ids, output_hidden_states=True, use_cache=False)
    logits = outputs.logits[0, :-1].float()
    target_ids = input_ids[0, 1:].detach().cpu()
    n_positions = int(logits.shape[0])

    log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    entropy = -(probs * log_probs).sum(dim=-1)
    realized_logprob = log_probs.gather(1, target_ids.to(device).view(-1, 1)).squeeze(1)
    max_logprob, _ = log_probs.max(dim=-1)
    top_values, top_indices = torch.topk(log_probs, k=min(20, log_probs.shape[-1]), dim=-1)
    top2_margin = top_values[:, 0] - top_values[:, 1]
    top10_cumulative_mass = torch.exp(top_values[:, : min(10, top_values.shape[1])]).sum(dim=-1)
    ranks = (log_probs > realized_logprob[:, None]).sum(dim=-1) + 1

    prior_log = directions.unigram_logprior.to(device)
    prior_prob = directions.unigram_prior.to(device)
    realized_prior_logprob = prior_log[target_ids.to(device)]
    pmi_realized = realized_logprob - realized_prior_logprob
    kl_to_prior = (probs * (log_probs - prior_log[None, :])).sum(dim=-1)
    cross_entropy_to_prior = -(probs * prior_log[None, :]).sum(dim=-1)

    logits_centered = logits - logits.mean(dim=-1, keepdim=True)
    prior_log_centered = directions.prior_log_centered.to(device)
    c_hat = (logits_centered @ prior_log_centered) / directions.prior_log_centered_norm_sq.to(device)

    # HF decoder backbones such as Llama/Gemma/Qwen/GPTNeoX return the final
    # hidden state after the model's final norm. This is the actual lm_head
    # input, matching the convention used in Bayesian_interp_dipole.ipynb.
    # Do not apply the final norm a second time for final-layer geometry.
    y = outputs.hidden_states[-1][0, :-1].float()
    d_hat = directions.d_hat.to(device)
    d_star = directions.d_star.to(device)
    norm_y = torch.linalg.vector_norm(y, dim=-1).clamp_min(1e-12)
    alpha_dhat = (y @ d_hat) / norm_y
    alpha_dstar = (y @ d_star) / norm_y
    proj_dhat = alpha_dhat * norm_y
    dark_mass_fraction = _dark_mass_fraction(y, directions.dark_projector)
    y_bos = y[0]
    cos_to_bos = F.cosine_similarity(y, y_bos[None, :], dim=-1)

    top20_pairs_json = [
        json.dumps([(int(idx), float(val)) for idx, val in zip(row_idx.cpu(), row_val.cpu())])
        for row_idx, row_val in zip(top_indices, top_values)
    ]

    rows: list[dict[str, Any]] = []
    for pos in range(n_positions):
        region = _region_for_position(pos, prompt_len, prefix_token_len)
        row = {
            "question_id": question_id,
            "position": pos,
            "region": region,
            "offset_from_answer_onset": pos - (prompt_len - 1) if region == "answer" else None,
            "token_id_predicted": int(target_ids[pos]),
            "entropy": float(entropy[pos].cpu()),
            "logprob_realized": float(realized_logprob[pos].cpu()),
            "max_logprob": float(max_logprob[pos].cpu()),
            "top1_top2_margin": float(top2_margin[pos].cpu()),
            "rank_of_realized_token": int(ranks[pos].cpu()),
            "top10_cumulative_mass": float(top10_cumulative_mass[pos].cpu()),
            "pmi_realized": float(pmi_realized[pos].cpu()),
            "kl_to_prior": float(kl_to_prior[pos].cpu()),
            "cross_entropy_to_prior": float(cross_entropy_to_prior[pos].cpu()),
            "c_hat": float(c_hat[pos].cpu()),
            "alpha_dhat": float(alpha_dhat[pos].cpu()),
            "alpha_dstar": float(alpha_dstar[pos].cpu()),
            "norm_y": float(norm_y[pos].cpu()),
            "alpha_dhat_x_norm_y": float(proj_dhat[pos].cpu()),
            "dark_mass_fraction": float(dark_mass_fraction[pos].cpu()),
            "cos_to_bos": float(cos_to_bos[pos].cpu()),
            "top20_token_logprobs_json": top20_pairs_json[pos],
        }
        rows.append(row)

    if compute_per_layer:
        norm_module = _final_norm_module(model)
        for layer_idx, hidden_state in enumerate(outputs.hidden_states):
            if layer_idx == len(outputs.hidden_states) - 1:
                y_layer = hidden_state[0, :-1].float()
            else:
                # For intermediate layers this is a comparability convention,
                # not the model's native computation path.
                y_layer = _apply_final_norm(norm_module, hidden_state[0, :-1])
            norm_layer = torch.linalg.vector_norm(y_layer, dim=-1).clamp_min(1e-12)
            alpha_layer = (y_layer @ d_hat) / norm_layer
            for pos, value in enumerate(alpha_layer):
                rows[pos][f"layer{layer_idx}_alpha_dhat"] = float(value.cpu())

    return pd.DataFrame(rows)


def _decision_scalars(score_df: pd.DataFrame, prompt_len: int) -> dict[str, Any]:
    decision_position = prompt_len - 1
    row = score_df.loc[score_df["position"] == decision_position]
    if row.empty:
        raise ValueError(f"No decision row found at position {decision_position}.")
    row_dict = row.iloc[0].to_dict()
    excluded = {"question_id", "position", "region", "offset_from_answer_onset", "top20_token_logprobs_json"}
    return {f"decision_{key}": value for key, value in row_dict.items() if key not in excluded}


def extract_answer_span_trajectories(
    position_df: pd.DataFrame,
    prompt_len: int,
    answer_len_tokens: int,
) -> dict:
    """
    Extract ordered answer-span scalar trajectories from one question's rows.

    Offset 0 is the state predicting the first generated answer token. No
    aggregates are computed here; analysis notebooks can take traj[0],
    traj[-1], np.mean(traj), np.max(traj), etc. as needed.
    """
    answer_len_tokens = int(answer_len_tokens)
    if answer_len_tokens == 0:
        return {"answer_len_tokens": 0, "empty_answer": True, "trajectories": {}}

    missing = [name for name in SPAN_SCALARS if name not in position_df.columns]
    if missing:
        raise KeyError(f"position_df is missing required span scalar columns: {missing}")

    offsets = pd.to_numeric(position_df["offset_from_answer_onset"], errors="coerce")
    mask = (
        (position_df["region"] == "answer")
        & offsets.notna()
        & (offsets >= 0)
        & (offsets < answer_len_tokens)
    )
    span_df = position_df.loc[mask, ["offset_from_answer_onset", *SPAN_SCALARS]].copy()
    span_df["offset_from_answer_onset"] = span_df["offset_from_answer_onset"].astype(int)
    span_df = span_df.sort_values("offset_from_answer_onset")

    expected_offsets = np.arange(answer_len_tokens)
    observed_offsets = span_df["offset_from_answer_onset"].to_numpy()
    if not np.array_equal(observed_offsets, expected_offsets):
        raise ValueError(
            "Answer-span offsets do not match expected range "
            f"0..{answer_len_tokens - 1}; observed {observed_offsets.tolist()}."
        )

    return {
        "answer_len_tokens": answer_len_tokens,
        "empty_answer": False,
        "trajectories": {
            name: span_df[name].astype(float).to_numpy().tolist()
            for name in SPAN_SCALARS
        },
    }


def _sanitize_output_dir(output_dir: str, config: dict) -> Path:
    formatted = output_dir.format(
        MODEL_ID=str(config["MODEL_ID"]).replace("/", "__"),
        NUM_PROMPTS=config["NUM_PROMPTS"],
    )
    return Path(formatted)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _read_json_if_exists(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text())


def load_stage1_outputs(output_dir: str | Path, config: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reload saved Stage 1 parquet outputs from an output directory."""
    if config is not None:
        output_dir = _sanitize_output_dir(str(output_dir), config)
    output_dir = Path(output_dir)
    all_positions_path = output_dir / "all_positions.parquet"
    per_question_path = output_dir / "per_question.parquet"
    config_path = output_dir / "config.json"
    missing = [path for path in (all_positions_path, per_question_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing Stage 1 output files: "
            + ", ".join(str(path) for path in missing)
            + ". Run Stage 1 first or check OUTPUT_DIR."
        )
    if config is not None and config_path.exists():
        saved_config = _read_json_if_exists(config_path, {})
        saved_version = saved_config.get("STAGE1_MEASUREMENT_VERSION")
        current_version = config.get("STAGE1_MEASUREMENT_VERSION", STAGE1_MEASUREMENT_VERSION)
        if saved_version != current_version:
            raise RuntimeError(
                f"Saved Stage 1 outputs have measurement version {saved_version!r}, "
                f"but current code expects {current_version!r}. Recompute outputs or use a new OUTPUT_DIR."
            )
    all_positions_df = pd.read_parquet(all_positions_path)
    per_question_df = pd.read_parquet(per_question_path)
    return all_positions_df, per_question_df


def _trajectory_dict_from_row(row: pd.Series, trajectories: dict[str, Any] | None) -> dict[str, list[float]]:
    if trajectories is not None:
        value = trajectories.get(str(row["question_id"]), {})
        return value.get("trajectories", value) if isinstance(value, dict) else {}

    value = row.get("trajectories", {})
    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, dict) and "trajectories" in value:
        value = value["trajectories"]
    return value if isinstance(value, dict) else {}


def _answer_onset_prior_logprob(
    all_positions_df: pd.DataFrame | None,
    directions: Directions | None,
) -> pd.DataFrame:
    columns = ["question_id", "gold_token_unigram_logprob"]
    if all_positions_df is None or all_positions_df.empty:
        return pd.DataFrame(columns=columns)

    offsets = pd.to_numeric(all_positions_df["offset_from_answer_onset"], errors="coerce")
    onset_df = all_positions_df[
        (all_positions_df["region"] == "answer")
        & offsets.notna()
        & (offsets.astype(float) == 0.0)
    ].copy()
    if onset_df.empty:
        return pd.DataFrame(columns=columns)

    onset_df = onset_df.sort_values(["question_id", "position"]).drop_duplicates("question_id")
    if {"logprob_realized", "pmi_realized"}.issubset(onset_df.columns):
        onset_df["gold_token_unigram_logprob"] = (
            onset_df["logprob_realized"].astype(float) - onset_df["pmi_realized"].astype(float)
        )
    elif directions is not None and "token_id_predicted" in onset_df.columns:
        prior_log = directions.unigram_logprior.detach().cpu().float().numpy()
        token_ids = onset_df["token_id_predicted"].astype(int).to_numpy()
        onset_df["gold_token_unigram_logprob"] = prior_log[token_ids]
    else:
        onset_df["gold_token_unigram_logprob"] = np.nan

    return onset_df[columns]


def _bool_or_false(value: Any) -> bool:
    if pd.isna(value):
        return False
    return bool(value)


def build_analysis_table(
    per_question_df: pd.DataFrame,
    all_positions_df: pd.DataFrame | None = None,
    directions: Directions | None = None,
    trajectories: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build one row per question for Stage 1b analysis.

    For each scalar in SPAN_SCALARS, derive decision, onset_last, span_mean,
    and span_max columns from the saved answer-span trajectory. The main
    returned dataframe excludes abstentions; the second return value keeps
    abstentions for sensitivity checks.
    """
    rows: list[dict[str, Any]] = []
    for _, row in per_question_df.iterrows():
        out = row.to_dict()
        traj_dict = _trajectory_dict_from_row(row, trajectories)
        for scalar in SPAN_SCALARS:
            values = np.asarray(traj_dict.get(scalar, []), dtype=float)
            if values.size == 0:
                out[f"decision_{scalar}"] = out.get(f"decision_{scalar}", np.nan)
                out[f"onset_last_{scalar}"] = np.nan
                out[f"span_mean_{scalar}"] = np.nan
                out[f"span_max_{scalar}"] = np.nan
                continue

            out[f"decision_{scalar}"] = float(values[0])
            out[f"onset_last_{scalar}"] = float(values[-1])
            out[f"span_mean_{scalar}"] = float(np.nanmean(values))
            out[f"span_max_{scalar}"] = float(np.nanmax(values))

        correct_strict = _bool_or_false(row.get("correct_strict", False))
        abstained = _bool_or_false(row.get("abstained", False))
        if (not correct_strict) and (not abstained):
            out["label"] = "hallucinated"
        elif correct_strict:
            out["label"] = "correct"
        else:
            out["label"] = "abstained"
        rows.append(out)

    analysis_df = pd.DataFrame(rows)
    prior_df = _answer_onset_prior_logprob(all_positions_df, directions)
    if not prior_df.empty:
        analysis_df = analysis_df.drop(columns=["gold_token_unigram_logprob"], errors="ignore")
        analysis_df = analysis_df.merge(prior_df, on="question_id", how="left")
    elif "gold_token_unigram_logprob" not in analysis_df.columns:
        analysis_df["gold_token_unigram_logprob"] = np.nan

    abstained_df = analysis_df[analysis_df["label"] == "abstained"].copy()
    main_df = analysis_df[analysis_df["label"] != "abstained"].copy()
    return main_df, abstained_df


def _binary_labels(
    labels: pd.Series,
    positive_label: str,
) -> np.ndarray:
    return (labels.astype(str) == positive_label).astype(int).to_numpy()


def _safe_roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def _bootstrap_auc_ci(
    y_true: np.ndarray,
    scores: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    aucs = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        sample_idx = rng.integers(0, n, size=n)
        auc = _safe_roc_auc(y_true[sample_idx], scores[sample_idx])
        if np.isfinite(auc):
            aucs.append(auc)
    if not aucs:
        return float("nan"), float("nan")
    low, high = np.percentile(aucs, [2.5, 97.5])
    return float(low), float(high)


def compute_auroc_table(
    analysis_df: pd.DataFrame,
    scalar_cols: list[str],
    label_col: str = "label",
    positive_label: str = "hallucinated",
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Compute raw AUROC and bootstrap CIs for scalar hallucination scores.

    AUROC is reported in the raw direction where higher scalar values predict
    positive_label. The direction column makes inverted predictors explicit.
    """
    rows = []
    for col_idx, scalar in enumerate(scalar_cols):
        if scalar not in analysis_df.columns:
            rows.append({
                "scalar": scalar,
                "auroc": np.nan,
                "auroc_ci_low": np.nan,
                "auroc_ci_high": np.nan,
                "auroc_abs": np.nan,
                "direction": "missing",
                "n_positive": 0,
                "n_negative": 0,
                "dropped_count": len(analysis_df),
            })
            continue

        df = analysis_df[[label_col, scalar]].dropna().copy()
        dropped_count = int(len(analysis_df) - len(df))
        y = _binary_labels(df[label_col], positive_label)
        scores = df[scalar].astype(float).to_numpy()
        auc = _safe_roc_auc(y, scores)
        ci_low, ci_high = _bootstrap_auc_ci(y, scores, n_bootstrap, seed + col_idx)
        if np.isfinite(auc):
            direction = "positive" if auc >= 0.5 else "inverted"
        else:
            direction = "undefined"
        rows.append({
            "scalar": scalar,
            "auroc": auc,
            "auroc_ci_low": ci_low,
            "auroc_ci_high": ci_high,
            "auroc_abs": max(auc, 1 - auc) if np.isfinite(auc) else np.nan,
            "direction": direction,
            "n_positive": int(y.sum()),
            "n_negative": int(len(y) - y.sum()),
            "dropped_count": dropped_count,
        })
    return pd.DataFrame(rows)


def incremental_auc_test(
    analysis_df: pd.DataFrame,
    base_features: list[str],
    candidate_features: list[str],
    label_col: str = "label",
    positive_label: str = "hallucinated",
    n_folds: int = 5,
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> dict:
    """
    Cross-validated logistic-regression delta-AUC for candidate features.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    features = list(dict.fromkeys([*base_features, *candidate_features]))
    required_cols = [label_col, *features]
    missing = [col for col in required_cols if col not in analysis_df.columns]
    if missing:
        raise KeyError(f"analysis_df is missing required columns: {missing}")

    df = analysis_df[required_cols].dropna().copy()
    dropped_count = int(len(analysis_df) - len(df))
    y = _binary_labels(df[label_col], positive_label)
    if len(np.unique(y)) < 2:
        raise ValueError("Need at least one positive and one negative example for AUROC.")
    min_class_count = int(np.bincount(y).min())
    if min_class_count < 2:
        raise ValueError("Need at least two examples per class for cross-validation.")
    n_splits = min(int(n_folds), min_class_count)

    base_scores = np.full(len(df), np.nan, dtype=float)
    full_scores = np.full(len(df), np.nan, dtype=float)
    full_coef_rows = []
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, test_idx in cv.split(df[features], y):
        base_model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=seed),
        )
        full_model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=seed),
        )
        base_model.fit(df.iloc[train_idx][base_features], y[train_idx])
        full_model.fit(df.iloc[train_idx][features], y[train_idx])
        base_scores[test_idx] = base_model.predict_proba(df.iloc[test_idx][base_features])[:, 1]
        full_scores[test_idx] = full_model.predict_proba(df.iloc[test_idx][features])[:, 1]
        full_coef_rows.append(full_model.named_steps["logisticregression"].coef_[0])

    auroc_base = float(roc_auc_score(y, base_scores))
    auroc_full = float(roc_auc_score(y, full_scores))
    delta_auc = auroc_full - auroc_base

    rng = np.random.default_rng(seed)
    deltas = []
    n = len(y)
    for _ in range(n_bootstrap):
        sample_idx = rng.integers(0, n, size=n)
        if len(np.unique(y[sample_idx])) < 2:
            continue
        deltas.append(
            float(roc_auc_score(y[sample_idx], full_scores[sample_idx]))
            - float(roc_auc_score(y[sample_idx], base_scores[sample_idx]))
        )
    if deltas:
        ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])
    else:
        ci_low = ci_high = np.nan

    coef_mean = np.vstack(full_coef_rows).mean(axis=0)
    return {
        "auroc_base": auroc_base,
        "auroc_full": auroc_full,
        "delta_auc": float(delta_auc),
        "delta_auc_ci_low": float(ci_low),
        "delta_auc_ci_high": float(ci_high),
        "base_features": base_features,
        "candidate_features": candidate_features,
        "coef_full": {feature: float(coef) for feature, coef in zip(features, coef_mean)},
        "dropped_count": dropped_count,
        "n": int(len(df)),
    }


def entropy_decile_auc(
    analysis_df: pd.DataFrame,
    entropy_col: str,
    candidate_col: str,
    label_col: str = "label",
    positive_label: str = "hallucinated",
    n_bins: int = 5,
    min_bin_size: int = 15,
) -> pd.DataFrame:
    """
    Quantile-bin by entropy, then compute candidate AUROC within each bin.

    Interpretation note: a candidate scalar with strong marginal AUROC but
    flat ~0.5 AUROC across all entropy bins would indicate its signal is fully
    explained by entropy; a candidate with flat marginal AUROC but consistently
    off-0.5 within-bin AUROC would indicate a real effect masked by the entropy
    confound at the marginal level.
    """
    from sklearn.metrics import roc_auc_score

    required = [entropy_col, candidate_col, label_col]
    missing = [col for col in required if col not in analysis_df.columns]
    if missing:
        raise KeyError(f"analysis_df is missing required columns: {missing}")
    df = analysis_df[required].dropna().copy()
    if df.empty:
        return pd.DataFrame(columns=["bin_index", "bin_entropy_range", "n", "n_hallucinated", "auroc", "direction"])

    df["entropy_bin"] = pd.qcut(df[entropy_col], q=n_bins, duplicates="drop")
    rows = []
    for bin_idx, (interval, group) in enumerate(df.groupby("entropy_bin", observed=True)):
        y = _binary_labels(group[label_col], positive_label)
        n_positive = int(y.sum())
        n_negative = int(len(y) - n_positive)
        if n_positive < min_bin_size or n_negative < min_bin_size:
            auc = np.nan
            direction = "insufficient_class_count"
        else:
            auc = float(roc_auc_score(y, group[candidate_col].astype(float).to_numpy()))
            direction = "positive" if auc >= 0.5 else "inverted"
        rows.append({
            "bin_index": bin_idx,
            "bin_entropy_range": str(interval),
            "n": int(len(group)),
            "n_hallucinated": n_positive,
            "auroc": auc,
            "direction": direction,
        })
    return pd.DataFrame(rows)


def check_frequency_confound(
    analysis_df: pd.DataFrame,
    alpha_cols: list[str],
    freq_col: str = "gold_token_unigram_logprob",
) -> pd.DataFrame:
    """
    Correlate alpha features with first answer-token unigram logprob.
    """
    rows = []
    subsets = [("all", analysis_df)]
    if "label" in analysis_df.columns:
        subsets.extend([
            ("correct", analysis_df[analysis_df["label"] == "correct"]),
            ("hallucinated", analysis_df[analysis_df["label"] == "hallucinated"]),
        ])

    for alpha_col in alpha_cols:
        for subset_name, subset_df in subsets:
            if alpha_col not in subset_df.columns or freq_col not in subset_df.columns:
                corr_df = pd.DataFrame()
            else:
                corr_df = subset_df[[alpha_col, freq_col]].dropna()
            if len(corr_df) < 2:
                pearson_r = np.nan
                spearman_r = np.nan
            else:
                pearson_r = corr_df[alpha_col].corr(corr_df[freq_col], method="pearson")
                spearman_r = corr_df[alpha_col].corr(corr_df[freq_col], method="spearman")
            rows.append({
                "alpha_col": alpha_col,
                "subset": subset_name,
                "pearson_r": float(pearson_r) if pd.notna(pearson_r) else np.nan,
                "spearman_r": float(spearman_r) if pd.notna(spearman_r) else np.nan,
                "n": int(len(corr_df)),
            })
    return pd.DataFrame(rows)


def run_stage1(
    config: dict,
    *,
    model=None,
    tokenizer=None,
    directions: Directions | None = None,
    questions: list[dict] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run Stage 1 measurement and save parquet tables plus config sidecar.

    TODO: batch scoring with padding/attention masks after validating tokenizer
    padding side and causal-mask behavior for each model family.
    """
    output_dir = _sanitize_output_dir(config["OUTPUT_DIR"], config)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = dict(config)
    run_config.setdefault("STAGE1_MEASUREMENT_VERSION", STAGE1_MEASUREMENT_VERSION)

    fewshot_k = int(config.get("FEWSHOT_K", 4))
    fewshot_examples = DEFAULT_FEWSHOT_EXAMPLES[:fewshot_k]
    if questions is None:
        questions = load_trivia_qa_questions(int(config["NUM_PROMPTS"]), int(config["SEED"]))
    target_question_ids = [question["question_id"] for question in questions]
    target_question_id_set = set(target_question_ids)
    if model is None or tokenizer is None:
        model, tokenizer = load_model_and_tokenizer(config["MODEL_ID"], config["DTYPE"], config["DEVICE"])
    if directions is None:
        if bool(config.get("COMPUTE_DIRECTIONS_ON_THE_FLY", False)):
            directions = compute_directions_from_model(
                model,
                tokenizer,
                prior_kind=config.get("PRIOR_KIND", "empirical"),
                zipf_alpha=float(config.get("ZIPF_ALPHA", 1.0)),
                empirical_coverage=float(config.get("EMPIRICAL_COVERAGE", 0.60)),
                empirical_min_count=int(config.get("EMPIRICAL_MIN_COUNT", 2)),
                empirical_dataset_name=config.get("EMPIRICAL_DATASET_NAME", "monology/pile-uncopyrighted"),
                empirical_dataset_config=config.get("EMPIRICAL_DATASET_CONFIG"),
                empirical_split=config.get("EMPIRICAL_SPLIT", "train"),
                empirical_text_column=config.get("EMPIRICAL_TEXT_COLUMN", "text"),
                empirical_cache_dir=config.get("EMPIRICAL_CACHE_DIR", "data/token_stats"),
                empirical_max_examples=config.get("EMPIRICAL_MAX_EXAMPLES"),
                dark_fraction=float(config.get("DARK_FRACTION", 0.05)),
                ridge_lambda=config.get("RIDGE_LAMBDA"),
                directions_cache_path=config.get("DIRECTIONS_CACHE_PATH"),
                force_recompute=bool(config.get("FORCE_RECOMPUTE_DIRECTIONS", False)),
                verbose=True,
            )
        else:
            cache_path = config.get("DIRECTIONS_CACHE_PATH")
            if not cache_path:
                raise RuntimeError(
                    "COMPUTE_DIRECTIONS_ON_THE_FLY is False, but DIRECTIONS_CACHE_PATH is not set. "
                    "Set COMPUTE_DIRECTIONS_ON_THE_FLY=True to compute the empirical prior/directions, "
                    "or point DIRECTIONS_CACHE_PATH to a previously computed cache."
                )
            cache_path = Path(cache_path)
            if not cache_path.exists():
                raise RuntimeError(
                    "COMPUTE_DIRECTIONS_ON_THE_FLY is False, but the requested empirical prior/direction "
                    f"cache does not exist: {cache_path}. Expected prior_kind={config.get('PRIOR_KIND', 'empirical')!r}, "
                    f"coverage={config.get('EMPIRICAL_COVERAGE', 0.60)!r}, "
                    f"min_count={config.get('EMPIRICAL_MIN_COUNT', 2)!r}. "
                    "Set COMPUTE_DIRECTIONS_ON_THE_FLY=True to calculate and store it first."
                )
            directions = load_computed_directions_cache(
                cache_path,
                model,
                expected_prior_kind=config.get("PRIOR_KIND", "empirical"),
                expected_coverage=float(config.get("EMPIRICAL_COVERAGE", 0.60)),
                expected_min_count=int(config.get("EMPIRICAL_MIN_COUNT", 2)),
                expected_direction_version="repo_frequency_dipole_optimal_prior_v1",
            )

    resume_existing = bool(config.get("RESUME_PRECOMPUTED_RESULTS", True))
    all_positions_path = output_dir / "all_positions.parquet"
    per_question_path = output_dir / "per_question.parquet"
    trajectories_path = output_dir / "answer_span_trajectories.json"
    failures_path = output_dir / "failures.json"

    if resume_existing and all_positions_path.exists() and per_question_path.exists():
        saved_config = _read_json_if_exists(output_dir / "config.json", {})
        saved_version = saved_config.get("STAGE1_MEASUREMENT_VERSION")
        current_version = run_config["STAGE1_MEASUREMENT_VERSION"]
        if saved_version != current_version:
            raise RuntimeError(
                f"Refusing to resume stale Stage 1 outputs in {output_dir}: saved measurement "
                f"version is {saved_version!r}, current version is {current_version!r}. "
                "Use a new OUTPUT_DIR or remove old outputs before recomputing."
            )
        existing_all_positions_df = pd.read_parquet(all_positions_path)
        existing_per_question_df = pd.read_parquet(per_question_path)
        existing_all_positions_df = existing_all_positions_df[
            existing_all_positions_df["question_id"].astype(str).isin(target_question_id_set)
        ].copy()
        existing_per_question_df = existing_per_question_df[
            existing_per_question_df["question_id"].astype(str).isin(target_question_id_set)
        ].copy()
        existing_question_ids = set(existing_per_question_df["question_id"].astype(str))
        answer_span_trajectories = _read_json_if_exists(trajectories_path, {})
        answer_span_trajectories = {
            str(qid): value
            for qid, value in answer_span_trajectories.items()
            if str(qid) in target_question_id_set
        }
        failures = _read_json_if_exists(failures_path, [])
        print(
            f"Resuming from {output_dir}: found {len(existing_question_ids)} already-scored "
            f"questions out of requested {len(questions)}."
        )
    else:
        existing_all_positions_df = pd.DataFrame()
        existing_per_question_df = pd.DataFrame()
        existing_question_ids = set()
        answer_span_trajectories = {}
        failures = []

    questions_to_run = [
        question for question in questions
        if question["question_id"] not in existing_question_ids
    ]
    if questions_to_run:
        print(f"Scoring {len(questions_to_run)} new questions; {len(existing_question_ids)} will be reused.")
    else:
        print("All requested questions already have saved results; no new generation/scoring needed.")

    all_position_frames = []
    per_question_rows = []

    for idx, question in enumerate(tqdm(questions_to_run, desc="Stage 1 TriviaQA"), start=1):
        if idx == 1 or idx % 25 == 0:
            print(f"Scoring new question {idx}/{len(questions_to_run)}")
        qid = question["question_id"]
        try:
            prompt, prefix_char_len = build_fewshot_prompt(question["question"], fewshot_examples)
            prefix_token_len = len(tokenizer(prompt[:prefix_char_len], add_special_tokens=True)["input_ids"])
            generation = generate_answer(model, tokenizer, prompt)
            label = label_answer(generation["generated_text"], question["aliases"])
            score_df = score_sequence(
                model,
                tokenizer,
                directions,
                generation["full_input_ids"],
                generation["prompt_len"],
                question_id=qid,
                prefix_token_len=prefix_token_len,
            )
            for key, value in label.items():
                score_df[key] = value
            score_df["generated_text"] = generation["generated_text"]
            all_position_frames.append(score_df)
            answer_len_tokens = len(generation["generated_token_ids"])
            trajectories = extract_answer_span_trajectories(
                score_df,
                generation["prompt_len"],
                answer_len_tokens,
            )
            answer_span_trajectories[qid] = trajectories

            question_row = {
                "question_id": qid,
                "question": question["question"],
                "aliases": question["aliases"],
                "prompt": prompt,
                "prefix_char_len": prefix_char_len,
                "prompt_len": generation["prompt_len"],
                "answer_len_tokens": answer_len_tokens,
                "empty_answer": trajectories["empty_answer"],
                "generated_text": generation["generated_text"],
                **label,
                **_decision_scalars(score_df, generation["prompt_len"]),
            }
            per_question_rows.append(question_row)
        except Exception as exc:  # do not silently skip; record and continue
            failures.append({"question_id": qid, "exception": repr(exc)})
            print(f"[failure] question_id={qid}: {exc!r}")

    new_all_positions_df = pd.concat(all_position_frames, ignore_index=True) if all_position_frames else pd.DataFrame()
    new_per_question_df = pd.DataFrame(per_question_rows)
    all_positions_df = pd.concat(
        [existing_all_positions_df, new_all_positions_df],
        ignore_index=True,
    )
    per_question_df = pd.concat(
        [existing_per_question_df, new_per_question_df],
        ignore_index=True,
    )

    if not per_question_df.empty:
        order = {qid: idx for idx, qid in enumerate(target_question_ids)}
        per_question_df["_question_order"] = per_question_df["question_id"].astype(str).map(order)
        per_question_df = per_question_df.sort_values("_question_order").drop(columns="_question_order")
    if not all_positions_df.empty:
        order = {qid: idx for idx, qid in enumerate(target_question_ids)}
        all_positions_df["_question_order"] = all_positions_df["question_id"].astype(str).map(order)
        all_positions_df = (
            all_positions_df
            .sort_values(["_question_order", "position"])
            .drop(columns="_question_order")
        )

    all_positions_df.to_parquet(output_dir / "all_positions.parquet", index=False)
    per_question_df.to_parquet(output_dir / "per_question.parquet", index=False)
    _write_json(output_dir / "config.json", run_config)
    _write_json(output_dir / "answer_span_trajectories.json", answer_span_trajectories)
    _write_json(output_dir / "failures.json", failures)
    print(f"Completed Stage 1 with {len(failures)} failures. Outputs saved to {output_dir}.")
    if failures:
        print(f"Failures written to {output_dir / 'failures.json'}")

    return all_positions_df, per_question_df
