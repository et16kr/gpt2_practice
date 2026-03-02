#!/usr/bin/env python3
from __future__ import annotations

import torch
from transformers import AutoTokenizer


MODEL_ID = "openai-community/gpt2"


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"model_id={MODEL_ID}")
    print(f"vocab_size={tokenizer.vocab_size}")
    print(f"eos_token={tokenizer.eos_token}, eos_token_id={tokenizer.eos_token_id}")
    print(f"pad_token={tokenizer.pad_token}, pad_token_id={tokenizer.pad_token_id}")
    print()

    # 1) Single text encode/decode
    text = "Hello GPT-2. This is a tokenizer demo."
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    print("[Single Encode]")
    print("text:", text)
    print("input_ids shape:", tuple(input_ids.shape))
    print("input_ids:", input_ids.tolist())
    print("attention_mask:", attention_mask.tolist())
    print("tokens:", tokenizer.convert_ids_to_tokens(input_ids[0].tolist()))
    print("decoded:", tokenizer.decode(input_ids[0], skip_special_tokens=False))
    print()

    # 2) Batch encode with padding/truncation
    batch_texts = [
        "first sample",
        "second sample is longer than the first one",
    ]
    batch = tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        max_length=32,
        return_tensors="pt",
    )
    print("[Batch Encode]")
    print("input_ids shape:", tuple(batch["input_ids"].shape))
    print("attention_mask shape:", tuple(batch["attention_mask"].shape))
    print("input_ids:")
    print(batch["input_ids"])
    print("attention_mask:")
    print(batch["attention_mask"])
    print()

    # 3) LM training target shift (x=t[:-1], y=t[1:])
    train_ids = tokenizer.encode("A short training sample.", add_special_tokens=False)
    train_ids.append(tokenizer.eos_token_id)
    train_tensor = torch.tensor(train_ids, dtype=torch.long).unsqueeze(0)  # [1, T]
    x = train_tensor[:, :-1]
    y = train_tensor[:, 1:]
    print("[LM Shift Example]")
    print("full ids:", train_tensor.tolist())
    print("x ids:", x.tolist())
    print("y ids:", y.tolist())


if __name__ == "__main__":
    main()
