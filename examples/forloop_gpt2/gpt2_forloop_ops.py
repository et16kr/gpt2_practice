#!/usr/bin/env python3
from __future__ import annotations

import math
from typing import Any

import torch


def add_tensors(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape == b.shape
    out = torch.empty_like(a)
    a_flat = a.contiguous().view(-1)
    b_flat = b.contiguous().view(-1)
    out_flat = out.view(-1)
    for i in range(out_flat.numel()):
        out_flat[i] = a_flat[i] + b_flat[i]
    return out


def embedding_lookup(input_ids: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    # input_ids: [B, T], table: [V, C] -> out: [B, T, C]
    bsz, seqlen = input_ids.shape
    vocab, channels = table.shape
    out = torch.empty((bsz, seqlen, channels), dtype=table.dtype)
    for b in range(bsz):
        for t in range(seqlen):
            token_id = int(input_ids[b, t].item())
            assert 0 <= token_id < vocab
            for c in range(channels):
                out[b, t, c] = table[token_id, c]
    return out


def add_token_and_position_embeddings(
    token_emb: torch.Tensor, pos_emb: torch.Tensor
) -> torch.Tensor:
    # token_emb: [B, T, C], pos_emb: [T, C]
    bsz, seqlen, channels = token_emb.shape
    assert pos_emb.shape == (seqlen, channels)
    out = torch.empty_like(token_emb)
    for b in range(bsz):
        for t in range(seqlen):
            for c in range(channels):
                out[b, t, c] = token_emb[b, t, c] + pos_emb[t, c]
    return out


def layer_norm_last_dim(
    x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float
) -> torch.Tensor:
    # x: [B, T, C], gamma/beta: [C]
    bsz, seqlen, channels = x.shape
    assert gamma.shape == (channels,)
    assert beta.shape == (channels,)
    out = torch.empty_like(x)
    for b in range(bsz):
        for t in range(seqlen):
            mean = 0.0
            for c in range(channels):
                mean += float(x[b, t, c])
            mean /= channels

            var = 0.0
            for c in range(channels):
                diff = float(x[b, t, c]) - mean
                var += diff * diff
            var /= channels

            inv_std = 1.0 / math.sqrt(var + eps)
            for c in range(channels):
                norm = (float(x[b, t, c]) - mean) * inv_std
                out[b, t, c] = norm * float(gamma[c]) + float(beta[c])
    return out


def linear_last_dim(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None
) -> torch.Tensor:
    # x: [B, T, Cin], weight: [Cout, Cin], bias: [Cout] or None
    bsz, seqlen, cin = x.shape
    cout, cin_w = weight.shape
    assert cin == cin_w
    if bias is not None:
        assert bias.shape == (cout,)

    out = torch.empty((bsz, seqlen, cout), dtype=x.dtype)
    for b in range(bsz):
        for t in range(seqlen):
            for o in range(cout):
                acc = 0.0 if bias is None else float(bias[o])
                for i in range(cin):
                    acc += float(x[b, t, i]) * float(weight[o, i])
                out[b, t, o] = acc
    return out


def gelu_new_scalar(x: float) -> float:
    return 0.5 * x * (
        1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x))
    )


def gelu_new(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    x_flat = x.contiguous().view(-1)
    out_flat = out.view(-1)
    for i in range(x_flat.numel()):
        out_flat[i] = gelu_new_scalar(float(x_flat[i]))
    return out


def split_qkv_and_heads(
    qkv: torch.Tensor, n_head: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # qkv: [B, T, 3C] -> q/k/v: [B, H, T, Dh]
    bsz, seqlen, three_c = qkv.shape
    assert three_c % 3 == 0
    channels = three_c // 3
    assert channels % n_head == 0
    dh = channels // n_head

    q = torch.empty((bsz, n_head, seqlen, dh), dtype=qkv.dtype)
    k = torch.empty((bsz, n_head, seqlen, dh), dtype=qkv.dtype)
    v = torch.empty((bsz, n_head, seqlen, dh), dtype=qkv.dtype)

    for b in range(bsz):
        for t in range(seqlen):
            for h in range(n_head):
                for d in range(dh):
                    idx = h * dh + d
                    q[b, h, t, d] = qkv[b, t, idx]
                    k[b, h, t, d] = qkv[b, t, channels + idx]
                    v[b, h, t, d] = qkv[b, t, 2 * channels + idx]
    return q, k, v


def softmax_1d(values: list[float]) -> list[float]:
    max_v = max(values)
    exps = [math.exp(v - max_v) for v in values]
    denom = sum(exps)
    return [e / denom for e in exps]


def causal_scaled_dot_product_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    # q/k/v: [B, H, T, Dh] -> out: [B, H, T, Dh]
    bsz, n_head, seqlen, dh = q.shape
    assert k.shape == (bsz, n_head, seqlen, dh)
    assert v.shape == (bsz, n_head, seqlen, dh)
    scale = 1.0 / math.sqrt(dh)

    # 1) score = q @ k^T with causal mask
    scores = torch.empty((bsz, n_head, seqlen, seqlen), dtype=q.dtype)
    for b in range(bsz):
        for h in range(n_head):
            for i in range(seqlen):
                for j in range(seqlen):
                    if j > i:
                        scores[b, h, i, j] = -1.0e9
                    else:
                        dot = 0.0
                        for d in range(dh):
                            dot += float(q[b, h, i, d]) * float(k[b, h, j, d])
                        scores[b, h, i, j] = dot * scale

    # 2) softmax over last dim
    probs = torch.empty_like(scores)
    for b in range(bsz):
        for h in range(n_head):
            for i in range(seqlen):
                row = [float(scores[b, h, i, j]) for j in range(seqlen)]
                row_probs = softmax_1d(row)
                for j in range(seqlen):
                    probs[b, h, i, j] = row_probs[j]

    # 3) probs @ v
    out = torch.empty((bsz, n_head, seqlen, dh), dtype=q.dtype)
    for b in range(bsz):
        for h in range(n_head):
            for i in range(seqlen):
                for d in range(dh):
                    acc = 0.0
                    for j in range(seqlen):
                        acc += float(probs[b, h, i, j]) * float(v[b, h, j, d])
                    out[b, h, i, d] = acc
    return out


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    # x: [B, H, T, Dh] -> [B, T, C]
    bsz, n_head, seqlen, dh = x.shape
    channels = n_head * dh
    out = torch.empty((bsz, seqlen, channels), dtype=x.dtype)
    for b in range(bsz):
        for t in range(seqlen):
            for h in range(n_head):
                for d in range(dh):
                    c = h * dh + d
                    out[b, t, c] = x[b, h, t, d]
    return out


def cross_entropy_next_token(logits: torch.Tensor, labels: torch.Tensor) -> float:
    # logits: [B, T, V], labels: [B, T]
    bsz, seqlen, vocab = logits.shape
    assert labels.shape == (bsz, seqlen)

    total = 0.0
    count = 0
    for b in range(bsz):
        for t in range(seqlen - 1):
            target = int(labels[b, t + 1].item())
            assert 0 <= target < vocab

            max_logit = -float("inf")
            for v in range(vocab):
                lv = float(logits[b, t, v])
                if lv > max_logit:
                    max_logit = lv

            denom = 0.0
            for v in range(vocab):
                denom += math.exp(float(logits[b, t, v]) - max_logit)
            logsumexp = max_logit + math.log(denom)

            nll = -(float(logits[b, t, target]) - logsumexp)
            total += nll
            count += 1

    return total / count


def argmax_1d(vec: torch.Tensor) -> int:
    assert vec.dim() == 1
    best_idx = 0
    best_val = float(vec[0])
    for i in range(1, vec.numel()):
        cur = float(vec[i])
        if cur > best_val:
            best_val = cur
            best_idx = i
    return best_idx


def transformer_block(
    x: torch.Tensor, block: dict[str, Any], n_head: int, eps: float
) -> tuple[torch.Tensor, tuple[int, ...], tuple[int, ...]]:
    # Pre-LN attention
    h1 = layer_norm_last_dim(x, block["ln1_g"], block["ln1_b"], eps)
    qkv = linear_last_dim(h1, block["w_qkv"], block["b_qkv"])
    q, k, v = split_qkv_and_heads(qkv, n_head)
    attn_ctx = causal_scaled_dot_product_attention(q, k, v)
    attn_merge = merge_heads(attn_ctx)
    attn_out = linear_last_dim(attn_merge, block["w_attn_out"], block["b_attn_out"])
    resid1 = add_tensors(x, attn_out)

    # Pre-LN MLP
    h2 = layer_norm_last_dim(resid1, block["ln2_g"], block["ln2_b"], eps)
    fc = linear_last_dim(h2, block["w_fc"], block["b_fc"])
    act = gelu_new(fc)
    mlp_out = linear_last_dim(act, block["w_proj"], block["b_proj"])
    out = add_tensors(resid1, mlp_out)
    return out, tuple(q.shape), tuple(attn_ctx.shape)


def run_demo() -> None:
    torch.manual_seed(7)

    # Small shapes for a runnable loop-based demo.
    bsz = 1
    seqlen = 6
    vocab = 64
    channels = 8
    n_head = 2
    n_layer = 3  # set to 12 to mimic config.json
    ffn_hidden = 4 * channels
    eps = 1e-5

    assert channels % n_head == 0

    input_ids = torch.tensor([[1, 5, 7, 9, 3, 2]], dtype=torch.long)
    labels = input_ids.clone()

    # Parameters
    wte = torch.randn(vocab, channels, dtype=torch.float32) * 0.02
    wpe = torch.randn(seqlen, channels, dtype=torch.float32) * 0.02

    lnf_g = torch.ones(channels, dtype=torch.float32)
    lnf_b = torch.zeros(channels, dtype=torch.float32)

    lm_head = torch.randn(vocab, channels, dtype=torch.float32) * 0.02

    # Per-layer parameters (Transformer repeats this block n_layer times)
    blocks: list[dict[str, Any]] = []
    for _ in range(n_layer):
        block: dict[str, Any] = {
            "ln1_g": torch.ones(channels, dtype=torch.float32),
            "ln1_b": torch.zeros(channels, dtype=torch.float32),
            "ln2_g": torch.ones(channels, dtype=torch.float32),
            "ln2_b": torch.zeros(channels, dtype=torch.float32),
            "w_qkv": torch.randn(3 * channels, channels, dtype=torch.float32) * 0.02,
            "b_qkv": torch.zeros(3 * channels, dtype=torch.float32),
            "w_attn_out": torch.randn(channels, channels, dtype=torch.float32) * 0.02,
            "b_attn_out": torch.zeros(channels, dtype=torch.float32),
            "w_fc": torch.randn(ffn_hidden, channels, dtype=torch.float32) * 0.02,
            "b_fc": torch.zeros(ffn_hidden, dtype=torch.float32),
            "w_proj": torch.randn(channels, ffn_hidden, dtype=torch.float32) * 0.02,
            "b_proj": torch.zeros(channels, dtype=torch.float32),
        }
        blocks.append(block)

    # Embedding
    tok_emb = embedding_lookup(input_ids, wte)
    x = add_token_and_position_embeddings(tok_emb, wpe)

    # Repeated transformer blocks
    q_shape = ()
    attn_shape = ()
    for layer_idx in range(n_layer):
        x, q_shape, attn_shape = transformer_block(x, blocks[layer_idx], n_head, eps)

    # Final LN + LM head
    final = layer_norm_last_dim(x, lnf_g, lnf_b, eps)
    logits = linear_last_dim(final, lm_head, None)
    loss = cross_entropy_next_token(logits, labels)

    last_logits = logits[0, seqlen - 1, :]
    next_token = argmax_1d(last_logits)

    print("[for-loop GPT-2 ops demo]")
    print("n_layer (demo):", n_layer)
    print("input_ids shape:", tuple(input_ids.shape))
    print("x shape:", tuple(x.shape))
    print("q shape:", q_shape)
    print("attention context shape:", attn_shape)
    print("logits shape:", tuple(logits.shape))
    print("next-token CE loss:", f"{loss:.6f}")
    print("greedy next token id:", next_token)


if __name__ == "__main__":
    run_demo()
