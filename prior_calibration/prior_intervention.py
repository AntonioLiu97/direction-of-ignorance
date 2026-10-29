"""Fit the prior direction d_prior and rescale the prior loading at inference time.

The intervention edits the final hidden state y (after the last norm, i.e. the
input of the unembedding / lm_head):

    y  ->  y + (alpha - 1) * (y . d) d,        |d| = 1

so alpha = 1 is the unmodified model, alpha < 1 attenuates the unigram prior
and alpha > 1 amplifies it. The projection is computed in float32 and cast back
to the model dtype; alpha = 1 returns the input untouched.
"""
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'scripts'))
import utils  # noqa: E402
from pile_token_stats import tokenizer_cache_tag  # noqa: E402

TOKEN_STATS = REPO / 'data' / 'token_stats'


def pile_counts_path(tokenizer, vocab_size):
    """Locate the cached Pile token counts shipped for this tokenizer."""
    tag = tokenizer_cache_tag(tokenizer, vocab_size=vocab_size)
    path = TOKEN_STATS / f'{tag}__monology__pile-uncopyrighted__counts.npz'
    if not path.exists():
        raise FileNotFoundError(
            f'No cached Pile counts for tokenizer {tag}. Build them with '
            'scripts/pile_token_stats.py (streams monology/pile-uncopyrighted).')
    return path


def unigram_prior(counts, coverage=0.7):
    """Empirical unigram distribution restricted to the most frequent `coverage` of the vocabulary."""
    counts = np.asarray(counts)
    n = int(np.ceil(coverage * len(counts)))
    observed = np.flatnonzero(counts > 0)
    if len(observed) < n:
        raise ValueError(f'Only {len(observed)} tokens observed; coverage {coverage} needs {n}')
    ids = observed[np.argsort(counts[observed], kind='stable')[-n:]]
    p = np.zeros(len(counts), dtype=np.float64)
    p[ids] = counts[ids]
    return p / p.sum()


def fit_prior_direction(unembedding, counts, coverage=0.7):
    """Least-squares fit of log p(w) on the unembedding rows; returns the unit direction and fit stats."""
    W = np.asarray(unembedding, dtype=np.float32)
    if W.shape[0] != len(counts):
        raise ValueError(f'Unembedding has {W.shape[0]} rows but counts cover {len(counts)} tokens')
    fit = utils.frequency_dipole(W, unigram_prior(counts, coverage), method='optimal_prior', exclude_zeros=True)
    return dict(direction=fit['d_star'], beta=float(fit['beta_star']), r2=float(fit['r2']),
                n_tokens=int(fit['n_tokens']), coverage=coverage)


def prior_scaling_hook(direction, alpha):
    """Forward pre-hook for the unembedding module implementing the alpha rescaling."""
    d = torch.as_tensor(np.asarray(direction), dtype=torch.float32)
    d = d / d.norm()

    def hook(module, args):
        if alpha == 1:
            return args
        y = args[0]
        z = y.float()
        local = d.to(z.device)
        return ((z + (alpha - 1) * (z @ local).unsqueeze(-1) * local).to(y.dtype),) + args[1:]
    return hook


@contextmanager
def scale_prior(model, direction, alpha):
    """Temporarily rescale the prior loading of a Hugging Face causal LM by `alpha`."""
    handle = model.get_output_embeddings().register_forward_pre_hook(prior_scaling_hook(direction, alpha))
    try:
        yield model
    finally:
        handle.remove()
