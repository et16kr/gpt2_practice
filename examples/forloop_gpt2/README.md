# GPT-2 For-Loop Ops Example

CUDA 커널 작성 전에 인덱싱/메모리 접근을 눈으로 추적하기 위한 참고 코드입니다.

## Run

```bash
python3 examples/forloop_gpt2/gpt2_forloop_ops.py
```

기본값은 `n_layer=3`이며, 코드에서 값을 바꾸면 블록 반복 횟수를 늘릴 수 있습니다.
`config.json`과 동일하게 보려면 `n_layer=12`로 변경하세요.

## 포함 연산

- 토큰 임베딩 조회 (`embedding_lookup`)
- 토큰/위치 임베딩 합 (`add_token_and_position_embeddings`)
- LayerNorm (`layer_norm_last_dim`)
- Linear (`linear_last_dim`)
- QKV 분리 + head 변환 (`split_qkv_and_heads`)
- Causal Scaled Dot-Product Attention (`causal_scaled_dot_product_attention`)
- Head 병합 (`merge_heads`)
- GELU new (`gelu_new`)
- Next-token cross entropy (`cross_entropy_next_token`)

모든 핵심 계산은 Python `for` loop 기반으로 작성되어 매우 느리지만, CUDA 커널 단위로 분해해 보기 좋게 구성되어 있습니다.
