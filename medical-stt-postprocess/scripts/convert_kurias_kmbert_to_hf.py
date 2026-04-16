#!/usr/bin/env python3
"""KU-RIAS pytorch_model.bin + bert_config.json + kmbert_vocab.txt → HF BertForMaskedLM."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from transformers import BertConfig, BertForMaskedLM, BertTokenizer
from transformers.models.bert.tokenization_bert import load_vocab


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    ind = args.input_dir.resolve()
    out = args.out_dir.resolve()
    bin_path = ind / "pytorch_model.bin"
    cfg_path = ind / "bert_config.json"
    vocab_path = ind / "kmbert_vocab.txt"
    for path in (bin_path, cfg_path, vocab_path):
        if not path.is_file():
            raise SystemExit(f"없음: {path}")
    out.mkdir(parents=True, exist_ok=True)
    config = BertConfig(**json.loads(cfg_path.read_text(encoding="utf-8")))
    state = torch.load(bin_path, map_location="cpu", weights_only=True)
    state = {
        k: v
        for k, v in state.items()
        if not k.startswith("bert.pooler") and not k.startswith("cls.seq_relationship")
    }
    bias = state.get("cls.predictions.bias")
    if bias is None:
        raise SystemExit("cls.predictions.bias 없음")
    state["cls.predictions.decoder.bias"] = bias
    model = BertForMaskedLM(config)
    model.load_state_dict(state, strict=True)
    shutil.copy(vocab_path, out / "vocab.txt")
    vocab_dict = load_vocab(str(out / "vocab.txt"))
    tokenizer = BertTokenizer(vocab=vocab_dict, do_lower_case=False)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    cfgw = json.loads((out / "config.json").read_text(encoding="utf-8"))
    cfgw["model_type"] = "bert"
    (out / "config.json").write_text(json.dumps(cfgw, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"저장: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
