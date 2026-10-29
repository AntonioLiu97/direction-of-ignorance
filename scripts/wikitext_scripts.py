"""Load indexed WikiText-103 attribution scripts from ../data/wikitext103_400w/."""

from __future__ import annotations

import json
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "wikitext103_400w"


def manifest_path(data_root: Path | None = None) -> Path:
    root = DATA_ROOT if data_root is None else Path(data_root)
    return root / "manifest.json"


def load_manifest(data_root: Path | None = None) -> dict:
    path = manifest_path(data_root)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run scripts/build_wikitext103_corpus.py first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def script_path(index: int, data_root: Path | None = None) -> Path:
    manifest = load_manifest(data_root)
    count = manifest["count"]
    if index < 0 or index >= count:
        raise IndexError(f"script index {index} out of range [0, {count})")
    root = DATA_ROOT if data_root is None else Path(data_root)
    return root / manifest["scripts"][index]["file"]


def load_script(index: int, data_root: Path | None = None) -> str:
    """Load full passage text for script index (400 words by default at build time)."""
    return script_path(index, data_root).read_text(encoding="utf-8")


def truncate_to_tokens(text: str, tokenizer, tok_num: int) -> tuple[str, int]:
    """Truncate text to the first tok_num tokens under the given tokenizer."""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"][:tok_num]
    return tokenizer.decode(ids, skip_special_tokens=True), len(ids)


def load_script_for_model(
    index: int,
    tokenizer,
    tok_num: int = 200,
    data_root: Path | None = None,
) -> str:
    """Load script index and truncate to tok_num tokens for the attribution model."""
    text, _ = truncate_to_tokens(load_script(index, data_root), tokenizer, tok_num)
    return text
