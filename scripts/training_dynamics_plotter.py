"""Shared training-dynamics figure for output and input embeddings."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LABEL_MAP = {
    "pythia-70m": "Pythia 70M",
    "pythia-160m": "Pythia 160M",
    "pythia-410m": "Pythia 410M",
    "pythia-1b": "Pythia 1B",
    "pythia-1.4b": "Pythia 1.4B",
    "pythia-2.8b": "Pythia 2.8B",
    "pythia-6.9b": "Pythia 6.9B",
    "Llama-3.2-1B": "Llama 1B",
    "Llama-3.1-8B": "Llama 8B",
    "Llama-3.2-3B": "Llama 3B",
    "Llama-3.1-70B": "Llama 70B",
    "Llama-3.1-405B": "Llama 405B",
    "Qwen2.5-1.5B": "Qwen 1.5B",
    "Qwen2.5-3B": "Qwen 3B",
    "gemma-3-1b-pt": "Gemma 1B",
    "gemma-3-4b-pt": "Gemma 4B",
}

FONTSIZE = 11

NKL_YLABEL = (
    "$\\dfrac{\\mathrm{KL}(P_{\\mathrm{unigram}}\\,\\|\\,P_{\\mathrm{prior}})}"
    "{\\mathrm{KL}(P_{\\mathrm{unigram}}\\,\\|\\,\\mathrm{Unif})}$"
)


def _plot_final_strip(
    side_axes,
    final_points,
    keys,
    *,
    title="final checkpoints",
    show_init_pairs=True,
    show_init_markers=False,
    untrained_band=None,
    untrained_baseline=None,
    untrained_baseline_label=None,
    final_marker_label=None,
):
    """Draw final-checkpoint markers in the narrow companion axes.

    By default, a cache carrying emulated init-point metrics (``<key>_init``)
    is shown as an init -> final pair joined by a segment. Set
    ``show_init_pairs=False`` to drop those vertical trajectories;
    ``show_init_markers=True`` then still shows the reinitialized points as
    open markers, laterally offset from their trained counterparts but without
    the connecting segment. ``untrained_band`` gives a per-row (lo, hi) range
    of untrained checkpointed models, continued across the strip, and
    ``final_marker_label`` annotates the final markers as a group.
    """
    n = len(final_points)
    dx = 0.25
    for row, key in enumerate(keys):
        sax = side_axes[row]
        if untrained_band is not None:
            band_lo, band_hi = (float(v) for v in untrained_band[row])
            sax.axhspan(band_lo, band_hi, color="gray", alpha=0.3, lw=0, zorder=0)
            # Sit just below the lowest untrained marker (band floor and any
            # emulated-init points), so the tag does not collide with a model
            # whose init falls under the Pythia band (e.g. Llama 70B).
            lowest_untrained = band_lo
            if show_init_pairs or show_init_markers:
                init_key = f"{key}_init"
                for _, _, data in final_points:
                    if init_key in data.files:
                        lowest_untrained = min(
                            lowest_untrained,
                            float(np.asarray(data[init_key])),
                        )
            sax.text(
                0.97,
                lowest_untrained - 0.08,
                "untrained models",
                transform=sax.get_yaxis_transform(),
                ha="right",
                va="top",
                fontsize=FONTSIZE,
                color="dimgray",
            )
        if untrained_baseline is not None:
            baseline_y = float(np.asarray(untrained_baseline[row]))
            sax.axhline(
                baseline_y,
                color="gray",
                lw=1.4,
                ls="--",
                alpha=0.55,
                zorder=0,
            )
            if untrained_baseline_label is not None:
                # Below the line: the baseline sits at the top of the y-range,
                # so a label above it would be clipped by the axes frame.
                sax.text(
                    0.97,
                    baseline_y,
                    untrained_baseline_label,
                    transform=sax.get_yaxis_transform(),
                    ha="right",
                    va="top",
                    fontsize=FONTSIZE - 1,
                    color="gray",
                )
        for i, (label, style, data) in enumerate(final_points):
            color, marker = style if isinstance(style, tuple) else (style, "*")
            # Stars have less ink than diamonds/squares at equal ms.
            ms = 13 if marker == "*" else 9
            y_final = float(np.asarray(data[key]))
            init_key = f"{key}_init"
            show_init = (show_init_pairs or show_init_markers) and init_key in data.files
            if show_init:
                y_init = float(np.asarray(data[init_key]))
                # Paired mode splits init/final laterally around i; the
                # marker-only mode stacks them in one vertical column.
                x_init, x_final = (i - dx, i + dx) if show_init_pairs else (i, i)
                if show_init_pairs:
                    sax.plot(
                        [x_init, x_final], [y_init, y_final], color=color, lw=1.2,
                    )
                sax.plot(
                    x_init, y_init, marker=marker, ms=ms,
                    mfc="white", mec=color, mew=1.2, ls="none",
                )
            else:
                x_final = i
            sax.plot(
                x_final, y_final, marker=marker, ms=ms,
                color=color, mec="black", mew=0.6, ls="none",
            )
        # set_ylim(bottom=0) on the main panel disables y autoscale for the
        # shared row, so widen the limit if a final point falls outside it.
        value_keys = (
            (key, f"{key}_init")
            if (show_init_pairs or show_init_markers)
            else (key,)
        )
        vals = [
            float(np.asarray(data[k]))
            for _, _, data in final_points
            for k in value_keys
            if k in data.files
        ]
        if final_marker_label is not None:
            # Same lateral placement as the "uniform dist." label, sitting
            # right above the cluster of trained final markers (init points
            # excluded so the label hugs the trained cluster).
            final_vals = [
                float(np.asarray(data[key])) for _, _, data in final_points
            ]
            sax.text(
                0.97,
                max(final_vals) + 0.08,
                final_marker_label,
                transform=sax.get_yaxis_transform(),
                ha="right",
                va="bottom",
                fontsize=FONTSIZE,
                color="black",
            )
        if untrained_band is not None:
            vals.append(float(untrained_band[row][1]))
        if untrained_baseline is not None:
            vals.append(float(np.asarray(untrained_baseline[row])))
        _, top = sax.get_ylim()
        if max(vals) > top:
            sax.set_ylim(top=1.08 * max(vals))
        sax.set_xlim(-0.7, n - 0.3)
        sax.tick_params(labelleft=False, labelsize=FONTSIZE)
        sax.grid(True, alpha=0.3, axis="both")
    side_axes[0].set_title(title, fontsize=FONTSIZE)
    side_axes[-1].set_xticks(range(n))
    side_axes[-1].set_xticklabels(
        [label for label, _, _ in final_points],
        rotation=45,
        ha="right",
        fontsize=FONTSIZE,
    )


def model_specs_from_figure_models(figure_models, cmap_name: str = "viridis"):
    """Color-code models by size (light → dark) for overlays across panels."""
    cmap = plt.get_cmap(cmap_name)
    tags = [m.split("/")[-1] for m in figure_models]
    n = len(tags)
    return [
        (tag, LABEL_MAP.get(tag, tag), cmap(0.05 + 0.95 * i / max(n - 1, 1)))
        for i, tag in enumerate(tags)
    ]


def load_cached_curves(
    model_specs,
    cache_dir: Path,
    *,
    require_keys: tuple[str, ...] = ("nkl", "rank_fraction"),
):
    """Load npz caches; skip missing/stale files and report them."""
    loaded = []
    for tag, label, color in model_specs:
        path = Path(cache_dir) / f"{tag}.npz"
        if not path.exists():
            print(f"[missing] {path}")
            continue
        data = np.load(path)
        missing = [k for k in require_keys if k not in data.files]
        if missing:
            print(f"[stale] {path} missing keys {missing}")
            continue
        loaded.append((label, color, data))

    if not loaded:
        raise RuntimeError(f"No usable caches in {cache_dir} for keys {require_keys}.")

    shown = {lab for lab, _, _ in loaded}
    expected = {lab for _, lab, _ in model_specs}
    if shown != expected:
        print(f"[warn] plotted {sorted(shown)}; suite is {sorted(expected)}")
    return loaded


def plot_training_dynamics(
    loaded,
    *,
    nkl_key: str = "nkl",
    rank_key: str = "rank_fraction",
    null_key: str = "hm_rank_fraction",
    title_suffix: str = "",
    outfile: str | Path = "fig3_training_dynamics_pythia.pdf",
    figsize: tuple[float, float] = (8, 4),
    dpi: int = 200,
    show_uniform_baseline: bool = False,
    export_dir: str | Path = "plot_export",
    final_points=None,
):
    """
    Stacked (a) normalized KL and (b) normalized spectral rank vs training step.

    Style is shared by output- and input-embedding figures; only the cache keys,
    title suffix, and outfile differ.

    final_points: optional list of (label, style, data) for models with only a
    final checkpoint (no training trajectory). Shown as single markers in a
    narrow companion strip to the right of each panel, sharing the y-axis, so
    they are not placed at a fictitious training step on the log-x axis.
    style is either a color or a (color, marker) tuple (marker defaults to "*").
    """
    if final_points:
        fig, axes_grid = plt.subplots(
            2,
            2,
            figsize=(figsize[0] * 1.2, figsize[1]),
            dpi=dpi,
            sharex="col",
            sharey="row",
            gridspec_kw={"width_ratios": [4, 1], "wspace": 0.06},
            layout="constrained",
        )
        axes = axes_grid[:, 0]
        side_axes = axes_grid[:, 1]
    else:
        fig, axes = plt.subplots(2, 1, figsize=figsize, dpi=dpi, sharex=True)
        side_axes = None

    ax = axes[0]
    for label, color, data in loaded:
        ax.plot(
            data["steps"] + 1,
            data[nkl_key],
            "o-",
            ms=3.5,
            lw=1.2,
            color=color,
            label=label,
        )
    if show_uniform_baseline:
        ax.axhline(1.0, color="gray", lw=0.8, ls="--", label="uniform-prior baseline")
    ax.set_xscale("log")
    # nKL = KL(P_unigram || P_prior) / KL(P_unigram || Uniform), where
    # P_unigram is the Pile token-frequency distribution and P_prior is the
    # softmax prior encoded along d_prior.
    ax.set_ylabel(NKL_YLABEL, fontsize=FONTSIZE)
    ax.set_ylim(bottom=0)
    ax.set_title(rf"$\bf{{(a)}}$ Prior recovery{title_suffix}", fontsize=FONTSIZE)
    ax.legend(frameon=False, fontsize=FONTSIZE)
    ax.tick_params(labelsize=FONTSIZE)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for label, color, data in loaded:
        ax.plot(
            data["steps"] + 1,
            data[rank_key],
            "o-",
            ms=3.5,
            lw=1.2,
            color=color,
            label=label,
        )
    # Null: expected rank fraction of a noise-fit direction at initialization
    # (step 0) = rank fraction of the spectrum's harmonic mean. Shown as a band
    # spanning the smallest to largest model; falls back to the 1/2 reference
    # when caches predate the hm_rank_fraction key.
    # Stored as a scalar (step-0 value); older caches kept a per-step array.
    init_nulls = [
        float(np.atleast_1d(data[null_key])[0])
        for _, _, data in loaded
        if null_key in data.files
    ]
    if init_nulls:
        ax.axhspan(min(init_nulls), max(init_nulls), color="gray", alpha=0.35, lw=0)
        null_y = 0.5 * (min(init_nulls) + max(init_nulls))
    else:
        ax.axhline(0.5, color="gray", lw=0.8, ls="--")
        null_y = 0.5
    # Annotate the null band directly instead of using a legend (models are
    # already labeled in panel a).
    ax.text(
        0.98,
        null_y,
        "random direction rank",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="center",
        fontsize=FONTSIZE,
        color="black",
    )
    # Not inverted: lower rank fraction = more prominent (aligned with panel a,
    # where lower nKL = better prior recovery).
    ax.set_xlabel("Training step", fontsize=FONTSIZE)
    # Fraction of PCA eigenvalues above d_prior's Rayleigh quotient:
    # 0 = top of spectrum, ~0.5 = indistinguishable from a random direction.
    ax.set_ylabel(
        "$\\dfrac{\\mathrm{rank}(\\boldsymbol{d}_{\\mathrm{prior}})}{d_{\\mathrm{model}}}$",
        fontsize=FONTSIZE,
    )
    ax.set_title(rf"$\bf{{(b)}}$ Spectral prominence{title_suffix}", fontsize=FONTSIZE)
    ax.tick_params(labelsize=FONTSIZE)
    ax.grid(True, alpha=0.3)

    if side_axes is not None:
        _plot_final_strip(side_axes, final_points, (nkl_key, rank_key))

    if side_axes is None:
        fig.tight_layout()
    export_path = Path(export_dir)
    export_path.mkdir(parents=True, exist_ok=True)
    out = export_path / outfile
    fig.savefig(out, bbox_inches="tight")
    plt.show()
    return fig


def plot_prior_recovery(
    loaded,
    *,
    nkl_key: str = "nkl",
    outfile: str | Path = "fig3_prior_recovery_main.pdf",
    figsize: tuple[float, float] = (7, 2),
    dpi: int = 200,
    export_dir: str | Path = "plot_export",
    final_points=None,
    yscale: str = "linear",
):
    """
    Single-panel main-paper figure: normalized KL vs training step, with an
    optional final-checkpoint panel for models without intermediate checkpoints.
    The untrained state is represented by a translucent band carrying the
    untrained (first-checkpoint) Pythia levels across both panels, spanning
    the highest to the lowest step-0 value, instead of a separate vertical
    init-to-final segment for every model.

    Prior recovery is the headline metric: unlike spectral rank it is invariant
    to reparameterizations of the hidden space (e.g. folding the final norm
    gain into the unembedding), so it compares cleanly across model families.
    """
    if final_points:
        fig, axes_row = plt.subplots(
            1,
            2,
            figsize=(figsize[0] * 1.2, figsize[1]),
            dpi=dpi,
            sharey=True,
            # No constrained layout: it would push the panels apart to make
            # room for the strip's rotated tick labels, which can safely
            # extend under the main panel instead.
            gridspec_kw={"width_ratios": [2, 1], "wspace": 0.05},
        )
        ax = axes_row[0]
        side_axes = axes_row[1:]
    else:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        side_axes = None

    for label, color, data in loaded:
        ax.plot(
            data["steps"] + 1,
            data[nkl_key],
            "o-",
            ms=3.5,
            lw=1.2,
            color=color,
            label=label,
        )
    ax.set_xscale("log")

    # Continue the untrained (first-checkpoint) levels across panel (a) as a
    # translucent band spanning the highest to the lowest step-0 value; the
    # matching band is drawn across panel (b) below so the checkpoint-free
    # models can be compared with the untrained Pythia suite.
    init_vals = [float(np.asarray(data[nkl_key])[0]) for _, _, data in loaded]
    untrained_band = (min(init_vals), max(init_vals))
    ax.axhspan(*untrained_band, color="gray", alpha=0.3, lw=0, zorder=0)
    ax.set_xlabel("Training step", fontsize=FONTSIZE)
    ax.set_ylabel(NKL_YLABEL, fontsize=FONTSIZE)
    if yscale == "log":
        ax.set_yscale("log")
    else:
        # Fixed headroom so open init markers and strip labels are not clipped.
        ax.set_ylim(0.0, 1.05)
    ax.set_title("Pythia checkpoints", fontsize=FONTSIZE)
    # Pin the legend to the empty mid-right region: "best" placement can push
    # it up into the untrained band.
    ax.legend(frameon=False, fontsize=FONTSIZE, loc="center right")
    ax.tick_params(labelsize=FONTSIZE)
    ax.grid(True, alpha=0.3)

    if side_axes is not None:
        _plot_final_strip(
            side_axes,
            final_points,
            (nkl_key,),
            title="models w/o checkpoints",
            show_init_pairs=False,
            show_init_markers=True,
            untrained_band=(untrained_band,),
            final_marker_label="trained models",
        )
        if yscale != "log":
            # Re-assert after the strip, which may widen the shared y-limit.
            ax.set_ylim(0.0, 1.05)

    if side_axes is None:
        fig.tight_layout()
    export_path = Path(export_dir)
    export_path.mkdir(parents=True, exist_ok=True)
    out = export_path / outfile
    fig.savefig(out, bbox_inches="tight")
    plt.show()
    return fig
