"""Audit retained passage summaries and build the appendix paired table (CPU only)."""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
PC = ROOT / 'prior_calibration'
N, SEED, REPS = 200, 20260923, 10000


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


REFERENCE = read(PC / 'scaling_pairs_bos_cov07_v1/manifest.json')
HASHES = [p['sha256'] for p in REFERENCE['passages']]
assert len(HASHES) == N and len(set(HASHES)) == N


def load_run(run, key):
    folder = PC / run
    manifest = read(folder / 'manifest.json')
    assert [p['sha256'] for p in manifest['passages']] == HASHES
    assert manifest['coverage'] == .7 and manifest['interval'] == [250, 300]
    spec = manifest.get('models', {}).get(key, {})
    sequences = spec.get('token_ids', manifest.get('token_ids'))
    dest = folder / key
    assert read(dest / 'complete.json')['n'] == N
    assert read(dest / 'identity_gate.json')['passed']
    vals, hashes = [], []
    for i in range(N):
        path = dest / 'results' / f'{i:04}.npz'
        with np.load(path) as z:
            assert str(z['text_sha256']) == HASHES[i]
            assert np.array_equal(z['token_ids'], sequences[i])
            lam = z['lambda_values'].astype(float)
            assert lam.shape == (300,) and np.isfinite(lam).all()
            vals.append(float(lam[250:300].mean()))
        hashes.append(digest(path))
    return dict(values=vals, tokens=sequences, source=f'{run}/{key}',
                dtype=read(dest / 'runtime.json').get('dtype', 'unknown'),
                probability_sha256=spec.get('probability_sha256', manifest.get('probability_sha256')),
                model_id=spec.get('model_id', manifest.get('adapter')),
                result_sha256=hashes, pairing='per-passage hashes and token IDs verified')


def load_cache(model, reference_run, reference_key):
    """Legacy caches lack token IDs: preserve that provenance limitation explicitly."""
    path = ROOT / 'data/fig2_cache' / f'panelA_{model.replace("/", "__")}_N200_T300.npz'
    with np.load(path) as z:
        assert bool(z['BOS_token']) and np.isclose(z['vocab_coverage'], .7)
        vals = (z['proj_mat'][:, 250:300] / float(z['beta_star'])).mean(axis=1)
        beta = float(z['beta_star'])
    gate = read(PC / reference_run / reference_key / 'reference_gate.json')
    assert gate['passed'] and np.isclose(gate['original_beta'], beta)
    return dict(values=vals.tolist(), tokens=None, source=str(path.relative_to(ROOT)),
                dtype='legacy cache (not independently recorded)', model_id=model,
                cache_sha256=digest(path), pairing='legacy original WikiText index order; no cached token IDs')


def analyze(base, variant, indices):
    b, v = np.asarray(base['values']), np.asarray(variant['values'])
    assert b.shape == v.shape == (N,) and np.isfinite(b).all() and np.isfinite(v).all()
    if base['tokens'] is not None:
        assert base['tokens'] == variant['tokens'], 'Unmatched tokens'
        assert base['probability_sha256'] == variant['probability_sha256'], 'Unmatched frequency reference'
    rb, rv = abs(b.mean()), abs(v.mean())
    assert rb > 0
    bs, vs = b[indices].mean(axis=1), v[indices].mean(axis=1)
    # For stable nonzero signs, the magnitude contrast is locally linear.
    stable = min(abs(b.mean()) / stats.sem(b), abs(v.mean()) / stats.sem(v)) >= 3
    flips = int(np.count_nonzero(np.sign(bs) != np.sign(b.mean())) +
                np.count_nonzero(np.sign(vs) != np.sign(v.mean())))
    assert stable and flips == 0, 'Near-zero signed mean: needs nonregular inference; do not publish t approximation'
    differences = np.sign(b.mean()) * b - np.sign(v.mean()) * v
    test = stats.ttest_1samp(differences, 0, alternative='two-sided')
    assert np.isclose(differences.mean(), rb - rv)
    draws = 100 * (1 - np.abs(vs) / np.abs(bs))
    return dict(base_reliance=float(rb), variant_reliance=float(rv),
                base_ci=np.percentile(np.abs(bs), [2.5, 97.5]).tolist(),
                variant_ci=np.percentile(np.abs(vs), [2.5, 97.5]).tolist(),
                reduction_pct=float(100 * (1 - rv / rb)),
                reduction_ci=np.percentile(draws, [2.5, 97.5]).tolist(),
                p_raw=float(test.pvalue), t=float(test.statistic), df=N-1,
                signed_base_mean=float(b.mean()), signed_variant_mean=float(v.mean()),
                sign_flip_count=flips, base_provenance=base, variant_provenance=variant)


def build():
    pairs = 'scaling_pairs_bos_cov07_v1'
    remaining = 'scaling_remaining_pairs_cov07_v1'
    simple = 'scaling_simplerl_cov07_v1'
    additions = 'scaling_llama3_additions_cov07_v1'
    gemma = 'scaling_gemma1_reasoning_cov07_v1'
    cloud = 'scaling_pairs_bos_cov07_cloud_v1'
    bases = {
        'Llama 3.2 1B': load_cache('meta-llama/Llama-3.2-1B', remaining, 'llama1_grpo'),
        'Llama 3.2 3B': load_run(pairs, 'llama3b_base'),
        'Llama 3.1 8B': load_run(pairs, 'llama8_base'),
        'Llama 3.1 70B': load_run(cloud, 'llama70_base'),
        'Gemma 3 1B': load_run(gemma, 'gemma1_base'),
        'Qwen 2.5 1.5B': load_run(pairs, 'qwen1p5_base'),
        'Qwen 2.5 3B': load_cache('Qwen/Qwen2.5-3B', remaining, 'qwen3_smallthinker'),
    }
    refit = read(PC / 'data/gemma4_scaling_refit_cov70.json')
    assert refit['original_fit_reproduced'] and refit['coverage'] == .7
    path = ROOT / 'data/fig2_cache/panelA_google__gemma-3-4b-pt_N200_T300.npz'
    with np.load(path) as z:
        assert np.isclose(float(z['beta_star']), refit['original_beta']) and bool(z['BOS_token'])
        values = (z['proj_mat'][:, 250:300] / float(z['beta_star'])).mean(axis=1)
    bases['Gemma 3 4B'] = dict(values=values.tolist(), tokens=None, source=str(path.relative_to(ROOT)),
        dtype='legacy cache (not independently recorded)', cache_sha256=digest(path),
        pairing='legacy original WikiText index order; no cached token IDs')
    specifications = [
        ('Llama 3.2 1B', 'Llama 1B GRPO', remaining, 'llama1_grpo'),
        ('Llama 3.2 3B', 'DeepHermes 3B', pairs, 'llama3b_reasoning'),
        ('Llama 3.2 3B', 'FineMath 3B', additions, 'llama3_finemath'),
        ('Llama 3.2 3B', 'ThinkSFT 3B', additions, 'llama3_thinksft'),
        ('Llama 3.1 8B', 'SimpleRL Llama 8B', simple, 'llama8_simplerl'),
        ('Llama 3.1 8B', 'R1-Distill Llama 8B', pairs, 'llama8_reasoning'),
        ('Llama 3.1 70B', 'R1-Distill Llama 70B', cloud, 'llama70_reasoning'),
        ('Gemma 3 1B', 'Gemma 1B reasoning (GRPO)', gemma, 'gemma1_reasoning'),
        ('Gemma 3 4B', 'Gemma 4B reasoning (GRPO)', 'scaling_gemma4_exploratory_cov07_v1', 'reasoning'),
        ('Qwen 2.5 1.5B', 'Qwen 2.5 Math 1.5B', pairs, 'qwen1p5_math'),
        ('Qwen 2.5 1.5B', 'R1-Distill Qwen 1.5B', pairs, 'qwen1p5_reasoning'),
        ('Qwen 2.5 1.5B', 'SimpleRL Qwen 1.5B', simple, 'qwen1p5_simplerl'),
        ('Qwen 2.5 3B', 'SmallThinker 3B', remaining, 'qwen3_smallthinker'),
    ]
    indices = np.random.default_rng(SEED).integers(0, N, size=(REPS, N))
    rows = []
    for name, label, run, key in specifications:
        base, variant = bases[name], load_run(run, key)
        if key == 'llama8_simplerl':
            base = load_run('scaling_llama8_base_fp32_cov07_v1', 'llama8_base')
        if key == 'reasoning':
            variant['values'] = next(x['passage_means'] for x in refit['models'] if x['key'] == 'reasoning')
            variant['refit_source'] = 'prior_calibration/data/gemma4_scaling_refit_cov70.json'
        row = dict(base_model=name, variant=label, key=key, **analyze(base, variant, indices))
        rows.append(row)
    order = np.argsort([r['p_raw'] for r in rows])
    adjusted = np.maximum.accumulate([(len(rows)-rank)*rows[j]['p_raw'] for rank,j in enumerate(order)])
    for j, p in zip(order, adjusted):
        rows[j]['p_holm'] = float(min(p, 1))
    report = dict(n=N, window=[250,300], coverage=.7, bootstrap_seed=SEED,
                  bootstrap_replicates=REPS, passage_hashes=HASHES, rows=rows)
    out = ROOT / 'notebook_plotters/plot_export'
    out.mkdir(exist_ok=True)
    (out / 'scaling_paired_significance.json').write_text(json.dumps(report, indent=2)+'\n')
    lines = ['| Base model | Reasoning/math variant | Base reliance | Variant reliance | Reliance reduction (%) | p |',
             '|---|---|---:|---:|---:|---:|']
    def fmt(value, ci, digits):
        return f'{value:.{digits}f} [{ci[0]:.{digits}f}, {ci[1]:.{digits}f}]'
    for r in rows:
        lines.append('| '+' | '.join([r['base_model'], r['variant'],
            fmt(r['base_reliance'],r['base_ci'],4), fmt(r['variant_reliance'],r['variant_ci'],4),
            fmt(r['reduction_pct'],r['reduction_ci'],1), f"{r['p_holm']:.2e}"])+ ' |')
    markdown = '\n'.join(lines)
    (out / 'scaling_paired_significance.md').write_text(markdown+'\n')
    print(markdown)
    return report


if __name__ == '__main__':
    build()
