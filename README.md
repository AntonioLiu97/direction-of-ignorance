# The Direction of Ignorance: LLMs Encode and Temper a Unigram Prior

Code and data for the paper by Toni J.B. Liu, Gurbir Arora, Jiajun Bao, Mitchell Ostrow, Yizhou Liu,
Raphaël Sarfati, Nicolas Boullé, and Christopher J. Earls.

Language models encode the unigram token prior as a single direction $\boldsymbol d_{\mathrm{prior}}$
in the unembedding space. The final hidden state decomposes as a context term plus a prior term with
loading factor $\lambda$, which grows when the context is uninformative. Rescaling that loading at
inference time ($y \to y + (\alpha-1)(y\cdot d)\,d$) causally changes behavior; on GSM8K, attenuating
the prior ($\alpha<1$) corrects errors for some models.

## Layout

```text
scripts/                  core library: fitting d_prior, lambda, Pile token statistics, WikiText loader
notebook_plotters/        one notebook per paper figure / table (saved outputs included)
prior_calibration/
  prior_intervention.py   fit d_prior and rescale the prior loading in any Hugging Face model
  run_gsm8k_sweep.py      GSM8K alpha sweep used in Section 5
  manifests/              the fixed 1,536-problem GSM8K training subset
  helpers/                unit-aware rescoring, plotting helpers for Section 5
  build_scaling_paired_table.py   appendix Table 2 (CPU only)
  data/                   summary statistics behind the Section 5 and 6 figures
data/
  token_stats/            Pile token counts per tokenizer (used to fit d_prior)
  wikitext103_400w/       the indexed WikiText-103 passages used for lambda measurements
```

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
huggingface-cli login   # Llama and Gemma checkpoints are gated
```

Run outputs (GSM8K generations for every model and alpha, per-passage lambda results, and small
summary caches) are attached to the GitHub release rather than tracked in git. To replot without GPUs,
download and extract them at the repository root:

```bash
wget https://github.com/AntonioLiu97/direction-of-ignorance/releases/download/v1.0/direction-of-ignorance-results.tar.gz
tar -xzf direction-of-ignorance-results.tar.gz   # ~340 MB compressed, ~1 GB extracted
```

## Using the intervention

```python
import sys; sys.path.insert(0, "prior_calibration")   # from the repository root
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import prior_intervention as pi

name = "meta-llama/Llama-3.2-3B-Instruct"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype="bfloat16", device_map="cuda")

W = model.get_output_embeddings().weight.float().cpu().numpy()
counts = np.load(pi.pile_counts_path(tok, W.shape[0]))["counts"]
fit = pi.fit_prior_direction(W, counts, coverage=0.7)        # d_prior, beta, R^2

with pi.scale_prior(model, fit["direction"], alpha=0.65):  # attenuate the prior
    out = model.generate(**tok("...", return_tensors="pt").to("cuda"), max_new_tokens=256)
```

## Reproducing the GSM8K sweeps (Section 5)

```bash
# Main-text model (Figures 4 and 5): lm-eval gsm8k-cot-llama prompt, 8-shot, greedy, 1,024 new tokens
python prior_calibration/run_gsm8k_sweep.py --model meta-llama/Llama-3.2-3B-Instruct \
    --protocol main --alphas $(seq 0 0.05 1.5) --out runs/llama32_3b

# Other models (Appendix Figure 12, Table 3)
python prior_calibration/run_gsm8k_sweep.py --model google/gemma-3-12b-it --protocol multimodel --out runs/gemma3_12b
```

The script writes `alpha_<value>/predictions.jsonl` in the same format as the released runs, and
resumes interrupted runs. Use `--limit 5` for a quick check. For Llama-3.2-3B-Instruct it reproduces
the released direction and generations exactly (checked on the first problems at α = 0.65 and 1).
The 1,536 problems come from the GSM8K *training* split, and the best α is chosen post hoc, so the
sweeps are exploratory rather than a held-out test-set result.

## Figures and tables

"Archive" means the notebook replots from the release archive on a CPU. "GPU" means the notebook
first runs the models to rebuild hidden-state caches (tens of GB), then plots.

| Paper item | Notebook / script | Needs |
|---|---|---|
| Figure 1 (geometry of d_prior), Table 1, Figures 7 and 15 | `fig1_dipole_geometry.ipynb` | GPU |
| Figure 2 (prior recovery during training) | `fig4_training_dynamics_pythia.ipynb` | GPU |
| Figure 3 (lambda in natural text), Figure 8 | `fig2_multi_model_prior_reliance_wikitext.ipynb` | GPU |
| Figure 4 (α sweep, Llama-3.2-3B-Instruct) | `fig4_2_alpha_sweep.ipynb` | Archive |
| Figure 5 and Figure 14 (forking paths) | `fig4_forking_paths.ipynb` | Archive |
| Figure 6 (prior reliance vs. model scale), Figure 11 | `scaling_law.ipynb` | GPU |
| Table 2 (paired prior reliance after post-training) | `prior_calibration/build_scaling_paired_table.py` | Archive |
| Figures 9 and 10 (two frequency branches) | `fig1_2_multi_dipole_geometry.ipynb` | GPU |
| Figure 12 and Table 3 (nine-model GSM8K sweeps) | `GSM8K_all_models_sweep.ipynb` | Archive |
| Figure 13 (λ intervention and KL) | `fig7_lambda_intervention.ipynb` | GPU |

Run notebooks from `notebook_plotters/`; figures are written to `notebook_plotters/plot_export/` and
`prior_calibration/plot_export/`. Figure 1 was assembled from the panel exported by the notebook.
The forking-path figures use the Inter and Source Code Pro fonts; without them matplotlib falls back to
DejaVu Sans and the layout differs slightly.

## Citation

```bibtex
@article{liu2026direction,
  title  = {The Direction of Ignorance: {LLMs} Encode and Temper a Unigram Prior},
  author = {Liu, Toni J.B. and Arora, Gurbir and Bao, Jiajun and Ostrow, Mitchell and Liu, Yizhou and
            Sarfati, Rapha{\"e}l and Boull{\'e}, Nicolas and Earls, Christopher J.},
  year   = {2026}
}
```

## License

MIT; see `LICENSE`.
