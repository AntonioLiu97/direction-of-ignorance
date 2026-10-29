from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import numpy as np


def _safe_tag(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "__", text).strip("_") or "tokenizer"


def _resolve_hf_token() -> str | None:
    """Return a usable HF token for gated datasets.

    A stale ``HF_TOKEN`` env var can shadow the token written by ``hf auth login``.
    Prefer the on-disk login token when present, then fall back to hub discovery.
    """
    for path in (
        Path.home() / ".cache" / "huggingface" / "token",
        Path.home() / ".huggingface" / "token",
    ):
        try:
            token = path.read_text().strip()
        except OSError:
            continue
        if token:
            return token
    try:
        from huggingface_hub import get_token
    except ImportError:
        return None
    return get_token()


def tokenizer_cache_tag(tokenizer: Any, *, vocab_size: int | None = None) -> str:
    """Stable cache tag for models that share the same tokenizer.

    The tag is purely content-based (vocabulary size plus a short hash of the
    tokenizer serialization), so models with identical tokenizer internals —
    e.g. Llama-3.1 and Llama-3.2 — share one empirical frequency cache.
    """
    if vocab_size is None:
        vocab_size = int(getattr(tokenizer, "vocab_size", len(tokenizer)))

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


def _counts_to_prob_at_coverage(counts: np.ndarray, coverage: float) -> np.ndarray:
    """Return probabilities using only the most frequent tokens up to coverage."""
    positive_ids = np.flatnonzero(counts > 0)
    if positive_ids.size == 0:
        raise RuntimeError("No positive token counts are available.")

    target_seen = int(np.ceil(coverage * counts.size))
    if positive_ids.size > target_seen:
        # Keep exactly the highest-count token IDs. This makes lower requested
        # coverage behave like a frequency threshold on a richer cached count set.
        order = np.argsort(counts[positive_ids], kind="stable")
        keep_ids = positive_ids[order[-target_seen:]]
        filtered = np.zeros_like(counts)
        filtered[keep_ids] = counts[keep_ids]
    else:
        filtered = counts

    total = filtered.sum()
    if total <= 0:
        raise RuntimeError("No tokens remain after applying the coverage threshold.")
    return filtered.astype(np.float64) / float(total)


def contrastive_token_prob_from_counts(
    foreground_counts: np.ndarray,
    background_counts: np.ndarray,
    *,
    coverage: float,
    epsilon: float = 1e-12,
    min_foreground_count: int = 1,
) -> np.ndarray:
    """Restrict the foreground unigram to tokens distinctive vs a background corpus.

    Score each token by ``log((p_fg + eps) / (p_bg + eps))``, keep the top
    ``coverage`` fraction of the vocabulary by that score (among tokens with
    foreground count >= ``min_foreground_count``), then renormalize the
    foreground probabilities on the retained support. Tokens outside the
    support are zero, matching the usual coverage mask used by dipole fits.
    """
    if not 0 < coverage <= 1:
        raise ValueError("coverage must be in the interval (0, 1].")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    if min_foreground_count < 1:
        raise ValueError("min_foreground_count must be >= 1.")

    fg = np.asarray(foreground_counts, dtype=np.float64)
    bg = np.asarray(background_counts, dtype=np.float64)
    if fg.ndim != 1 or bg.shape != fg.shape:
        raise ValueError("foreground_counts and background_counts must be 1D and aligned.")

    fg_total = float(fg.sum())
    bg_total = float(bg.sum())
    if fg_total <= 0 or bg_total <= 0:
        raise RuntimeError("Both foreground and background corpora need positive token counts.")

    p_fg = fg / fg_total
    p_bg = bg / bg_total
    score = np.log(p_fg + epsilon) - np.log(p_bg + epsilon)

    eligible = np.flatnonzero(fg >= min_foreground_count)
    if eligible.size == 0:
        raise RuntimeError("No tokens meet the foreground count threshold.")

    target = int(np.ceil(coverage * fg.size))
    if eligible.size > target:
        # Highest contrast score first; stable for ties.
        order = np.argsort(score[eligible], kind="stable")
        keep_ids = eligible[order[-target:]]
    else:
        keep_ids = eligible

    probs = np.zeros(fg.size, dtype=np.float64)
    probs[keep_ids] = fg[keep_ids]
    total = float(probs.sum())
    if total <= 0:
        raise RuntimeError("No foreground mass remains after contrastive selection.")
    return probs / total


def load_cached_token_counts(cache_path: str | Path) -> np.ndarray:
    """Load raw token counts from an ``empirical_token_prob`` cache file."""
    cache_path = Path(cache_path)
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing token-count cache: {cache_path}")
    with np.load(cache_path) as data:
        if "counts" not in data.files:
            raise KeyError(f"Cache {cache_path} has no 'counts' array.")
        return np.asarray(data["counts"], dtype=np.int64)


def empirical_token_prob(
    tokenizer: Any,
    *,
    vocab_size: int | None = None,
    coverage: float = 0.70,
    dataset_name: str = "monology/pile-uncopyrighted",
    dataset_config: str | None = None,
    data_dir: str | None = None,
    split: str = "train",
    text_column: str = "text",
    add_special_tokens: bool = False,
    cache_path: str | Path | None = None,
    max_examples: int | None = None,
    report_every: int = 1_000,
) -> np.ndarray:
    """Estimate token probabilities by streaming text through a tokenizer.

    The scan terminates once at least ``coverage`` of token IDs in the target
    vocabulary have appeared at least once. If ``cache_path`` exists, the scan
    resumes from the cached counts and consumed dataset offset. The cache keeps
    the highest available coverage, but the returned vector is thresholded to
    the requested coverage by retaining only the most frequent observed tokens
    and renormalizing. This means requesting lower coverage than the cache has
    will ignore lower-frequency tokens rather than using the full cache.

    ``monology/pile-uncopyrighted`` is the default because it is a much larger
    Pile-compatible stream that works with recent ``datasets`` versions. The
    original ``EleutherAI/pile`` dataset script now fails with
    "Dataset scripts are no longer supported".

    Code corpora such as ``bigcode/starcoderdata`` / ``bigcode/the-stack-dedup``
    typically expose file text in a ``content`` column and select a language
    slice with ``data_dir`` rather than ``dataset_config``.
    """
    if not 0 < coverage <= 1:
        raise ValueError("coverage must be in the interval (0, 1].")

    if vocab_size is None:
        vocab_size = int(getattr(tokenizer, "vocab_size", len(tokenizer)))
    vocab_size = int(vocab_size)

    target_seen = int(np.ceil(coverage * vocab_size))
    counts = np.zeros(vocab_size, dtype=np.int64)
    seen = np.zeros(vocab_size, dtype=bool)
    n_tokens = 0
    n_consumed_examples = 0
    n_counted_examples = 0

    cache_path = Path(cache_path) if cache_path is not None else None
    if cache_path is not None and cache_path.exists():
        data = np.load(cache_path)
        cached_counts = np.asarray(data["counts"], dtype=np.int64)
        if cached_counts.shape != (vocab_size,):
            raise ValueError(
                f"Cached counts have shape {cached_counts.shape}, expected {(vocab_size,)}."
            )
        counts = cached_counts.copy()
        seen = counts > 0
        n_tokens = int(data["n_tokens"])
        n_consumed_examples = int(
            data["n_consumed_examples"]
            if "n_consumed_examples" in data.files
            else data["n_examples"]
            if "n_examples" in data.files
            else 0
        )
        n_counted_examples = int(
            data["n_counted_examples"]
            if "n_counted_examples" in data.files
            else data["n_examples"]
            if "n_examples" in data.files
            else 0
        )

        if seen.sum() >= target_seen:
            return _counts_to_prob_at_coverage(counts, coverage)

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "empirical_token_prob requires the Hugging Face `datasets` package."
        ) from exc

    # Pass an explicit token for gated BigCode datasets. Avoid token=True alone:
    # a stale HF_TOKEN env var can override the cached `hf auth login` credential.
    load_kwargs: dict[str, Any] = {"split": split, "streaming": True}
    hf_token = _resolve_hf_token()
    if hf_token:
        load_kwargs["token"] = hf_token
    if data_dir is not None:
        load_kwargs["data_dir"] = data_dir
    if dataset_config is None:
        dataset = load_dataset(dataset_name, **load_kwargs)
    else:
        dataset = load_dataset(dataset_name, dataset_config, **load_kwargs)

    def save_cache() -> None:
        if cache_path is None or n_tokens == 0:
            return
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            probs=counts.astype(np.float64) / float(n_tokens),
            counts=counts,
            n_tokens=n_tokens,
            n_consumed_examples=n_consumed_examples,
            n_counted_examples=n_counted_examples,
            coverage=float((counts > 0).sum() / vocab_size),
            dataset_name=dataset_name,
            dataset_config="" if dataset_config is None else dataset_config,
            data_dir="" if data_dir is None else data_dir,
            split=split,
            text_column=text_column,
        )

    for row_idx, example in enumerate(dataset):
        if row_idx < n_consumed_examples:
            continue
        n_consumed_examples = row_idx + 1

        text = example.get(text_column)
        if not text:
            continue

        ids = tokenizer.encode(str(text), add_special_tokens=add_special_tokens)
        if not ids:
            continue

        ids_arr = np.asarray(ids, dtype=np.int64)
        ids_arr = ids_arr[(0 <= ids_arr) & (ids_arr < vocab_size)]
        if ids_arr.size == 0:
            continue

        counts += np.bincount(ids_arr, minlength=vocab_size)
        seen[ids_arr] = True

        n_tokens += int(ids_arr.size)
        n_counted_examples += 1

        n_seen = int(seen.sum())
        if report_every and n_counted_examples % report_every == 0:
            print(
                f"counted_examples={n_counted_examples:,} tokens={n_tokens:,} "
                f"coverage={n_seen / vocab_size:.1%}"
            )
            save_cache()

        if n_seen >= target_seen:
            break
        if max_examples is not None and n_counted_examples >= max_examples:
            break

    if n_tokens == 0:
        raise RuntimeError("No tokens were counted; check dataset/text_column settings.")
    save_cache()

    n_seen = int((counts > 0).sum())
    if n_seen < target_seen:
        raise RuntimeError(
            f"Stopped at {n_seen / vocab_size:.1%} coverage after "
            f"{n_counted_examples:,} counted examples; target was {coverage:.1%}. "
            f"Saved partial counts to {cache_path}."
        )

    return _counts_to_prob_at_coverage(counts, coverage)
