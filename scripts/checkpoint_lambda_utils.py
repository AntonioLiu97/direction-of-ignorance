"""Shared utilities for checkpoint-wise prior-loading analyses."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from huggingface_hub import list_repo_refs
from tqdm.auto import tqdm


@dataclass(frozen=True)
class StageCheckpoint:
    """One immutable Hugging Face checkpoint from a named training stage."""

    step: int
    tokens_b: int
    branch: str
    revision: str


def select_linear_stage_checkpoints(
    model_name: str,
    count: int,
    *,
    stage: str = "stage1",
    min_step: int = 0,
) -> list[StageCheckpoint]:
    """Select nearest unique checkpoints on a linear tokens-seen grid.

    OLMo checkpoint branches encode both optimizer steps and cumulative tokens,
    for example ``stage1-step10000-tokens84B``. Token spacing is used because
    optimizer-step scales differ across model sizes.
    """
    if count < 2:
        raise ValueError("count must be at least 2 to span a training stage.")
    if min_step < 0:
        raise ValueError("min_step must be non-negative.")

    pattern = re.compile(
        rf"^{re.escape(stage)}-step(?P<step>\d+)-tokens(?P<tokens>\d+)B$"
    )
    by_tokens: dict[int, StageCheckpoint] = {}
    for branch in list_repo_refs(model_name).branches:
        match = pattern.fullmatch(branch.name)
        if match is None:
            continue
        candidate = StageCheckpoint(
            step=int(match.group("step")),
            tokens_b=int(match.group("tokens")),
            branch=branch.name,
            revision=branch.target_commit,
        )
        if candidate.step < min_step:
            continue
        # Some repositories expose two snapshots rounded to the same token
        # count. Keep the later optimizer step deterministically.
        previous = by_tokens.get(candidate.tokens_b)
        if previous is None or candidate.step > previous.step:
            by_tokens[candidate.tokens_b] = candidate

    available = sorted(by_tokens.values(), key=lambda item: item.tokens_b)
    if len(available) < count:
        raise RuntimeError(
            f"{model_name} exposes only {len(available)} {stage} checkpoints "
            f"at or above step {min_step}."
        )

    targets = np.linspace(available[0].tokens_b, available[-1].tokens_b, count)
    remaining = available.copy()
    selected: list[StageCheckpoint] = []
    for target in targets:
        index = min(
            range(len(remaining)),
            key=lambda i: (abs(remaining[i].tokens_b - target), remaining[i].tokens_b),
        )
        selected.append(remaining.pop(index))
    return sorted(selected, key=lambda item: item.tokens_b)


def make_forward_backbone(model) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a forward function yielding final pre-LM-head hidden states."""
    backbone = None
    for attribute in ("gpt_neox", "model", "transformer"):
        candidate = getattr(model, attribute, None)
        if candidate is not None and candidate is not model:
            backbone = candidate
            break
    if backbone is None:
        raise TypeError(f"Could not identify the backbone of {type(model).__name__}.")

    def forward(input_ids: torch.Tensor) -> torch.Tensor:
        output = backbone(
            input_ids=input_ids,
            use_cache=False,
            return_dict=True,
        )
        if not hasattr(output, "last_hidden_state"):
            raise TypeError(
                f"{type(backbone).__name__} did not return last_hidden_state."
            )
        return output.last_hidden_state

    return forward


def module_execution_device(module) -> torch.device:
    """Resolve the execution device of a resident or Accelerate-offloaded module."""
    weight = getattr(module, "weight", None)
    if weight is None:
        raise TypeError(f"{type(module).__name__} has no weight parameter.")
    if weight.device.type != "meta":
        return weight.device

    hook = getattr(module, "_hf_hook", None)
    execution_device = getattr(hook, "execution_device", None)
    if execution_device is None:
        raise RuntimeError(
            f"{type(module).__name__} is on meta with no Accelerate execution device."
        )
    if isinstance(execution_device, int):
        return torch.device(f"cuda:{execution_device}")
    return torch.device(execution_device)


def output_embedding_numpy(model) -> np.ndarray:
    """Materialize W_out from resident weights or Accelerate's offload map."""
    output_embeddings = model.get_output_embeddings()
    weight = output_embeddings.weight
    if weight.device.type != "meta":
        return weight.detach().float().cpu().numpy()

    hook = getattr(output_embeddings, "_hf_hook", None)
    weights_map = getattr(hook, "weights_map", None)
    if weights_map is None:
        raise RuntimeError(
            "Output embedding is a meta tensor but has no Accelerate offload map."
        )
    try:
        offloaded_weight = weights_map["weight"]
    except (KeyError, TypeError) as error:
        raise RuntimeError(
            "Could not materialize output embedding from Accelerate's offload map."
        ) from error
    return offloaded_weight.detach().float().cpu().numpy()


def prefix_token_ids(
    tokenizer,
    *,
    use_bos: bool,
    bos_fallback: str | None = "eos",
) -> list[int]:
    """Resolve a one-token sequence prefix without architecture assumptions."""
    if not use_bos:
        return []
    token_id = tokenizer.bos_token_id
    if token_id is None and bos_fallback == "eos":
        token_id = tokenizer.eos_token_id
    if token_id is None:
        raise ValueError(
            "Tokenizer has no BOS token"
            + (" or EOS fallback." if bos_fallback == "eos" else ".")
        )
    return [int(token_id)]


def cache_coherent_latents(
    model,
    tokenizer,
    destination: str | Path,
    *,
    num_passages: int,
    num_tokens: int,
    load_text: Callable[[int], str],
    use_bos: bool = True,
    bos_fallback: str | None = "eos",
    description: str = "latent y",
) -> None:
    """Atomically cache final prediction states as passage × position × hidden."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    hidden_size = int(model.config.hidden_size)
    shape = (num_passages, num_tokens, hidden_size)
    partial = destination.with_name(destination.stem + ".partial.npy")
    partial.unlink(missing_ok=True)
    writer = np.lib.format.open_memmap(
        partial,
        mode="w+",
        dtype=np.float32,
        shape=shape,
    )
    forward_backbone = make_forward_backbone(model)
    model_device = module_execution_device(model.get_input_embeddings())
    prefix = prefix_token_ids(
        tokenizer,
        use_bos=use_bos,
        bos_fallback=bos_fallback,
    )
    passage_length = num_tokens - len(prefix)

    try:
        for row in tqdm(range(num_passages), desc=description):
            passage_ids = tokenizer(
                load_text(row),
                add_special_tokens=False,
            )["input_ids"][:passage_length]
            if len(passage_ids) != passage_length:
                raise ValueError(
                    f"Passage {row} has {len(passage_ids)} tokens; "
                    f"expected {passage_length}."
                )
            input_ids = torch.tensor(
                [prefix + passage_ids],
                dtype=torch.long,
                device=model_device,
            )
            with torch.inference_mode():
                hidden = forward_backbone(input_ids)[0]
            writer[row] = hidden.float().cpu().numpy()
        writer.flush()
    except Exception:
        del writer
        partial.unlink(missing_ok=True)
        raise

    del writer
    os.replace(partial, destination)


def projection_from_latents(
    source: str | Path,
    direction: np.ndarray,
    *,
    num_passages: int,
    num_tokens: int,
) -> np.ndarray:
    """Project a memory-mapped latent cache without loading it all into RAM."""
    source = Path(source)
    hidden = np.load(source, mmap_mode="r")
    expected = (num_passages, num_tokens)
    if hidden.shape[:2] != expected:
        raise ValueError(
            f"{source.name} has shape {hidden.shape}; expected {expected} x hidden."
        )
    direction = np.asarray(direction, dtype=np.float64)
    norm = np.linalg.norm(direction)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("direction must have finite, nonzero norm.")
    direction = direction / norm
    projection = np.empty(expected, dtype=np.float64)
    for row in range(num_passages):
        projection[row] = np.asarray(hidden[row], dtype=np.float64) @ direction
    return projection
