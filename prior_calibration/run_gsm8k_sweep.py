"""GSM8K alpha sweep with the prior-loading intervention (Section 5).

8-shot chain-of-thought GSM8K through lm-evaluation-harness, greedy decoding,
1,024 new tokens, on the fixed 1,536-problem training subset in
manifests/gsm8k_train_1536.json. Each alpha writes one predictions.jsonl row per
problem to <out>/alpha_<alpha>/, the layout read by helpers/rescore_units.py and
the Section 5 notebooks. Interrupted runs resume from the rows already written.

Example:
    python prior_calibration/run_gsm8k_sweep.py --model meta-llama/Llama-3.2-3B-Instruct \\
        --protocol main --alphas 1 0.65 --out runs/llama32_3b

Protocols:
    main        Llama-3.2-3B-Instruct runs in the main text: lm-eval's gsm8k-cot-llama
                prompt and stop sequences, BOS added, native chat template.
    multimodel  Nine-model sweep (Appendix): shared stop sequences, 4,096-token context,
                BOS added only for Gemma; use --no-chat-template for base/math models,
                which switches to the plain "Question: ... Answer:" format.
"""
import argparse
import copy
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prior_intervention as pi  # noqa: E402

MANIFEST = HERE / 'manifests' / 'gsm8k_train_1536.json'
TASK = 'gsm8k_train_subset'
CHAT_DATE = '08 Sep 2026'
MULTIMODEL_STOP = ['\nQuestion:', '<|im_end|>', '<|endoftext|>', '<end_of_turn>', '<|eot_id|>']


def sha256(x):
    return hashlib.sha256(x if isinstance(x, bytes) else x.encode()).hexdigest()


def save(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, default=str) + '\n')
    tmp.replace(path)


def task_config(manifest, out, protocol, chat):
    cfg = copy.deepcopy(manifest['task_config'])
    data = out / 'selected_train.jsonl'
    data.write_text(''.join(json.dumps(x) + '\n' for x in manifest['examples']))
    cfg.update(task=TASK, dataset_path='json', dataset_name=None,
               dataset_kwargs={'data_files': {'train': str(data)}}, test_split='train')
    if protocol == 'multimodel':
        if not chat:
            cfg.update(doc_to_text='Question: {{question}}\nAnswer:', target_delimiter=' ', fewshot_delimiter='\n\n')
        cfg['generation_kwargs'] = dict(do_sample=False, temperature=0, max_gen_toks=1024, until=MULTIMODEL_STOP)
    cfg['generation_kwargs']['max_gen_toks'] = 1024
    return cfg


def load_model(args):
    from lm_eval.models.huggingface import HFLM
    from transformers import AutoConfig
    kwargs = dict(pretrained=args.model, revision=args.revision, dtype='bfloat16', batch_size=1,
                  chat_template_args={'date_string': CHAT_DATE})
    if args.protocol == 'main':
        kwargs.update(add_bos_token=True)
    else:
        kwargs.update(add_bos_token='gemma' in args.model.lower(), max_length=4096, attn_implementation='sdpa')
    if args.parallelize:
        kwargs.update(parallelize=True)
    config = AutoConfig.from_pretrained(args.model, revision=args.revision)
    if 'Gemma3ForConditionalGeneration' in (config.architectures or []):
        from transformers import Gemma3ForConditionalGeneration

        class GemmaHFLM(HFLM):
            AUTO_MODEL_CLASS = Gemma3ForConditionalGeneration
        return GemmaHFLM(**kwargs)
    return HFLM(**kwargs)


def prior_direction(lm, out, coverage, path=None):
    if path:
        return np.load(path)
    if (out / 'direction.npy').exists():
        return np.load(out / 'direction.npy')
    model = lm.model
    weights = model.get_output_embeddings().weight.detach().float().cpu().numpy()
    counts_path = pi.pile_counts_path(lm.tokenizer, weights.shape[0])
    fit = pi.fit_prior_direction(weights, np.load(counts_path)['counts'], coverage)
    np.save(out / 'direction.npy', fit['direction'])
    save(out / 'direction.json', dict({k: v for k, v in fit.items() if k != 'direction'},
                                      counts=counts_path.name, direction_sha256=sha256((out / 'direction.npy').read_bytes())))
    return fit['direction']


def score(response, target, cfg):
    from lm_eval.filters.extraction import RegexFilter

    def clean(s):
        for pattern in cfg['metric_list'][0]['regexes_to_ignore']:
            s = re.sub(pattern, '', s)
        return s.lower()
    scores = {}
    for f in cfg['filter_list']:
        rule = f['filter'][0]
        value = RegexFilter(regex_pattern=rule['regex_pattern'], group_select=rule['group_select']).apply([[response]], [{}])[0][0]
        scores[f['name']] = dict(answer=value, correct=int(clean(value) == clean(target)), parse_failure=value == '[invalid]')
    return scores


def summarize(rows):
    summary = dict(n=len(rows))
    for key in ['strict-match', 'flexible-extract']:
        correct = sum(r['scores'][key]['correct'] for r in rows)
        summary[key] = dict(correct=correct, accuracy=correct / len(rows),
                            parse_failures=sum(r['scores'][key]['parse_failure'] for r in rows))
    summary['truncations'] = sum(r['truncated_at_limit'] for r in rows)
    return summary


def run_alpha(lm, cfg, manifest, direction, alpha, out, args):
    from lm_eval import simple_evaluate
    from tqdm import tqdm
    dest = out / f'alpha_{alpha:g}'
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / 'predictions.jsonl'
    rows = [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
    done = {r['doc_id']: r for r in rows}
    generate = lm.generate_until

    def generate_logged(requests, **kwargs):
        outputs = []
        with path.open('a') as stream, tqdm(total=len(requests), initial=len(done), desc=f'alpha={alpha:g}') as bar:
            for request in requests:
                i, prompt_hash = int(request.doc_id), sha256(request.args[0])
                if i in done:
                    assert done[i]['prompt_sha256'] == prompt_hash, 'Prompt changed since the saved row was written'
                    outputs.append(done[i]['response'])
                    continue
                response = generate([request], disable_tqdm=True)[0]
                doc = manifest['examples'][i]
                n_tokens = len(lm.tokenizer.encode(response, add_special_tokens=False))
                row = dict(doc_id=i, train_index=doc['train_index'], question_sha256=doc['question_sha256'],
                           prompt_sha256=prompt_hash, alpha=alpha, response=response, generated_tokens=n_tokens,
                           truncated_at_limit=n_tokens >= 1024,
                           scores=score(response, doc['answer'].split('####')[-1].strip(), cfg))
                stream.write(json.dumps(row) + '\n')
                stream.flush()
                done[i] = row
                outputs.append(response)
                bar.update(1)
        return outputs

    lm.generate_until = generate_logged
    try:
        with pi.scale_prior(lm.model, direction, alpha):
            simple_evaluate(model=lm, tasks=[cfg], num_fewshot=8, apply_chat_template=args.chat_template,
                            fewshot_as_multiturn=args.chat_template, limit=args.limit, log_samples=False,
                            random_seed=42, numpy_random_seed=42, torch_random_seed=42, fewshot_random_seed=42)
    finally:
        lm.generate_until = generate
    rows = sorted(done.values(), key=lambda r: r['doc_id'])
    summary = summarize(rows)
    save(dest / 'complete.json' if len(rows) == len(manifest['examples']) else dest / 'partial.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', required=True, help='Hugging Face model id or local path')
    parser.add_argument('--revision', default=None)
    parser.add_argument('--alphas', type=float, nargs='+', default=[i / 20 for i in range(8, 25)])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--protocol', choices=['main', 'multimodel'], default='multimodel')
    parser.add_argument('--no-chat-template', dest='chat_template', action='store_false')
    parser.add_argument('--coverage', type=float, default=0.7)
    parser.add_argument('--direction', type=Path, help='Use a saved d_prior (.npy) instead of refitting')
    parser.add_argument('--limit', type=int, help='Only the first N problems (smoke tests)')
    parser.add_argument('--parallelize', action='store_true', help='Split the model across visible GPUs')
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    args = parser.parse_args()
    if args.protocol == 'main' and not args.chat_template:
        parser.error('--no-chat-template is only defined for --protocol multimodel')

    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text())
    cfg = task_config(manifest, args.out, args.protocol, args.chat_template)
    lm = load_model(args)
    direction = prior_direction(lm, args.out, args.coverage, args.direction)
    save(args.out / 'configuration.json', dict(vars(args), out=str(args.out), manifest=str(args.manifest),
                                               direction=str(args.direction) if args.direction else None,
                                               manifest_sha256=sha256(args.manifest.read_bytes())))
    for alpha in args.alphas:
        summary = run_alpha(lm, cfg, manifest, direction, alpha, args.out, args)
        print(f"alpha={alpha:g}  flexible accuracy {summary['flexible-extract']['accuracy']:.4f} (n={summary['n']})", flush=True)


if __name__ == '__main__':
    main()
