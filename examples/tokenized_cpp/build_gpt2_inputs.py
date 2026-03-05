#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file
from tokenizers import Tokenizer


EOS_TOKEN_ID = 50256
MAX_LENGTH = 64


PROMPTS: list[str] = [
    "The weather is calm today, so I will focus on model debugging.",
    "A small batch helps me inspect each token prediction step by step.",
    "When attention scores explode, check scaling and mask logic first.",
    "Tokenization quality can strongly affect downstream generation quality.",
    "I will trace one layer at a time and compare every tensor shape.",
    "Gradient flow becomes easier to analyze with clear residual paths.",
    "If a tensor is misaligned, the next matrix multiply fails immediately.",
    "Profiling before optimization prevents wasted effort on the wrong bottleneck.",
    "The GPU kernel is fast only when memory access is also efficient.",
    "Consistent preprocessing makes training and inference behavior easier to match.",
    "Today I will build GPT2 input tensors first and verify each stage.",
    "Saving token IDs and embeddings together makes debugging much easier.",
    "When sentence lengths differ, padding masks must be handled correctly.",
    "Position embeddings help the model understand token order in a sequence.",
    "The initial input x0 is the sum of token and position embeddings.",
    "Saving tensors before layer normalization simplifies intermediate checks.",
    "Documenting weight application order improves reproducible experiments.",
    "Sentences with numbers and symbols also test tokenizer behavior: 42%!?",
    "A bilingual prompt set is useful for stress-testing byte-level BPE.",
    "After building inputs, I can apply wte, wpe, and each block weights.",
]


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    data_dir = Path(__file__).resolve().parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_path = repo_root / "gpt2" / "tokenizer.json"
    weights_path = repo_root / "gpt2" / "model.safetensors"

    prompts_path = data_dir / "prompts_20.txt"
    st_path = data_dir / "gpt2_inputs_20.safetensors"
    meta_path = data_dir / "gpt2_inputs_20.meta.json"

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.enable_truncation(max_length=MAX_LENGTH)
    tokenizer.enable_padding(
        length=MAX_LENGTH,
        pad_id=EOS_TOKEN_ID,
        pad_token="<|endoftext|>",
    )

    encodings = tokenizer.encode_batch(PROMPTS)
    input_ids = np.asarray([enc.ids for enc in encodings], dtype=np.int64)
    attention_mask = np.asarray([enc.attention_mask for enc in encodings], dtype=np.int64)
    token_lengths = attention_mask.sum(axis=1, dtype=np.int64)

    weights = load_file(str(weights_path))
    wte = weights["wte.weight"]  # [vocab, embd]
    wpe = weights["wpe.weight"]  # [n_positions, embd]

    batch_size, seq_len = input_ids.shape
    position_ids = np.tile(np.arange(seq_len, dtype=np.int64), (batch_size, 1))

    token_embeddings = wte[input_ids]  # [B, T, C]
    position_embeddings = wpe[position_ids]  # [B, T, C]
    x0 = token_embeddings + position_embeddings
    x0_masked = x0 * attention_mask[..., None].astype(x0.dtype)

    # Save in safetensors so C++ code can load without zip-based containers.
    save_file(
        {
            "input_ids": input_ids.astype(np.float32),
            "attention_mask": attention_mask.astype(np.float32),
            "position_ids": position_ids.astype(np.float32),
            "token_lengths": token_lengths.astype(np.float32),
            "token_embeddings": token_embeddings.astype(np.float32),
            "position_embeddings": position_embeddings.astype(np.float32),
            "x0": x0.astype(np.float32),
            "x0_masked": x0_masked.astype(np.float32),
        },
        str(st_path),
        metadata={
            "format": "gpt2_inputs",
            "num_samples": str(batch_size),
            "max_length": str(seq_len),
            "n_embd": str(int(wte.shape[1])),
            "eos_token_id": str(EOS_TOKEN_ID),
        },
    )

    prompts_path.write_text(
        "\n".join(f"{idx + 1:02d}. {text}" for idx, text in enumerate(PROMPTS)) + "\n",
        encoding="utf-8",
    )

    meta = {
        "num_samples": int(batch_size),
        "max_length": int(seq_len),
        "n_embd": int(wte.shape[1]),
        "eos_token_id": EOS_TOKEN_ID,
        "files": {
            "prompts": str(prompts_path),
            "input_safetensors": str(st_path),
        },
        "arrays": {
            "input_ids": list(input_ids.shape),
            "attention_mask": list(attention_mask.shape),
            "position_ids": list(position_ids.shape),
            "token_lengths": list(token_lengths.shape),
            "token_embeddings": list(token_embeddings.shape),
            "position_embeddings": list(position_embeddings.shape),
            "x0": list(x0.shape),
            "x0_masked": list(x0_masked.shape),
        },
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"saved prompts: {prompts_path}")
    print(f"saved input:   {st_path}")
    print(f"saved meta:    {meta_path}")
    print(f"input_ids shape={input_ids.shape}, x0 shape={x0.shape}")


if __name__ == "__main__":
    main()
