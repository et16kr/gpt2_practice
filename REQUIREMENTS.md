# GPT-2 구현 연습 요구사항 문서

## 1. 문서 목적
이 문서는 `config.json`을 기반으로 GPT-2(소형, 12-layer) 모델을 학습/추론 가능한 형태로 직접 구현하기 위한 최소 요구사항을 정의한다.  
대상 독자는 "GPT-2 동작 원리를 아직 모르는 개발자"다.

## 2. 참고 모델
- 모델: `openai-community/gpt2`
- 링크: https://huggingface.co/openai-community/gpt2
- 제공 파일: `config.json` (이미 확보됨)

## 3. 범위
- 포함 범위: `config.json` 파싱, GPT-2 블록(마스킹된 self-attention + MLP) 구현, 순전파(forward)/손실(loss) 계산, 텍스트 생성(greedy/sample)
- 제외 범위(초기 버전): 대규모 분산 학습, RLHF, instruction tuning, 고급 최적화

## 4. `config.json` 기반 핵심 스펙
- `vocab_size`: 50257
- `n_positions` / `n_ctx`: 1024
- `n_embd`: 768
- `n_layer`: 12
- `n_head`: 12
- `activation_function`: `gelu_new`
- `attn_pdrop`, `embd_pdrop`, `resid_pdrop`: 0.1
- `layer_norm_epsilon`: 1e-5
- `initializer_range`: 0.02
- `bos_token_id`, `eos_token_id`: 50256

## 5. 기능 요구사항(Functional Requirements)
- FR-01: `config.json`을 읽어 모델 하이퍼파라미터를 초기화해야 한다.
- FR-02: 토큰 임베딩(`wte`)과 위치 임베딩(`wpe`)을 더해 입력 표현을 만들어야 한다.
- FR-03: 각 Transformer block은 다음 순서를 따라야 한다: `LayerNorm -> Causal Multi-Head Self-Attention -> Residual Add -> LayerNorm -> MLP(FC -> GELU -> FC) -> Residual Add`
- FR-04: self-attention은 미래 토큰을 보지 못하도록 causal mask를 적용해야 한다.
- FR-05: 최종 hidden states를 vocab projection(`lm_head`)으로 변환해 logits를 출력해야 한다.
- FR-06: labels가 주어지면 next-token prediction cross-entropy loss를 계산해야 한다.
- FR-07: 생성 모드에서 `max_new_tokens`, `temperature`, `top_k`, `do_sample` 옵션을 지원해야 한다.
- FR-08: 모델 가중치 저장/로딩 인터페이스를 제공해야 한다.
- FR-09: Hugging Face GPT-2 토크나이저(`openai-community/gpt2`)로 문자열 <-> 토큰 ID 변환을 지원해야 한다.

## 6. 비기능 요구사항(Non-Functional Requirements)
- NFR-01: 랜덤 시드 고정 옵션 제공(재현성).
- NFR-02: 배치/시퀀스 길이에 대해 텐서 shape 검증(assert) 포함.
- NFR-03: 단일 GPU 또는 CPU에서 동작 가능해야 한다.
- NFR-04: 코드 구조를 모듈화하여 블록 단위 테스트가 가능해야 한다.

## 7. 권장 디렉터리 구조
```text
gpt2_practice/
  config.json
  README.md
  REQUIREMENTS.md
  examples/
    tokenizer/
    forloop_gpt2/
    tokenized_cpp/
    cpp_cuda_extension/
  src/
    model.py
    layers.py
    generate.py
    train.py
    config.py
  tests/
    test_shapes.py
    test_masking.py
    test_generation.py
```

## 8. 동작 원리 요약
- 입력 문장을 토크나이즈해서 정수 토큰 시퀀스로 변환한다.
- 토큰 임베딩 + 위치 임베딩을 더해 시퀀스 표현을 만든다.
- 12개 Transformer block을 통과시키며 문맥 정보를 누적한다.
- 각 위치의 hidden state를 vocab logits로 바꿔 "다음 토큰 확률"을 얻는다.
- 생성 시에는 새 토큰을 한 개씩 붙이면서 반복한다(autoregressive).

## 9. 구현 흐름 요약 (Pseudo Code 최소화)
- 초기화: `wte`, `wpe`, `n_layer`개의 block, `ln_f`, `lm_head`를 만든다.
- forward: `x = token_embedding + position_embedding` 후 block들을 순차 실행한다.
- block: `h = x + attention(ln1(x))`, `y = h + mlp(ln2(h))`.
- attention: `qkv -> head split -> causal score -> softmax -> weighted sum -> output projection`.
- loss: `shifted logits/labels`로 next-token cross-entropy를 계산한다.
- generation: 마지막 logits에서 다음 토큰을 뽑고 입력 뒤에 붙이는 과정을 반복한다.

상세 구현 참고 코드는 아래 예제를 기준으로 한다.
- `for-loop 연산 레퍼런스`: `examples/forloop_gpt2/gpt2_forloop_ops.py`
- `토크나이저 레퍼런스`: `examples/tokenizer/tokenizer_demo.py`
- `토크나이저 입력셋`: `examples/tokenized_cpp/data/prompts.txt`
- `Python -> C++/CUDA 연동 레퍼런스`: `examples/cpp_cuda_extension/demo.py`

## 10. 검증/완료 기준(Acceptance Criteria)
- AC-01: 임의 입력 `[B, T]`에 대해 출력 logits shape가 `[B, T, 50257]`이어야 한다.
- AC-02: causal mask 테스트에서 위치 `t`가 `t+1..T-1` 정보에 접근하지 않아야 한다.
- AC-03: 1-step 학습 시 loss가 계산되고 역전파(backprop)가 실패하지 않아야 한다.
- AC-04: 짧은 프롬프트 입력 시 `max_new_tokens`만큼 토큰이 생성되어야 한다.

## 11. 구현 순서(권장)
- 1단계: config loader + shape-only forward 골격 구현
- 2단계: attention/MLP 정확 구현 + unit test
- 3단계: loss 계산 + 1-step train loop
- 4단계: generate 함수(greedy/sample) 구현
- 5단계: Hugging Face 가중치 로딩 호환(선택)

## 12. 토크나이저 사용 가이드 (초심자용)

### 12.1 설치
```bash
pip install torch transformers tokenizers
```

### 12.2 토크나이저 로딩
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

# GPT-2는 기본 pad_token이 없어서 배치 padding이 필요하면 eos를 pad로 재사용
tokenizer.pad_token = tokenizer.eos_token
```

### 12.3 문자열 -> 토큰 ID (인코딩)
```python
text = "Hello, GPT-2!"
enc = tokenizer(
    text,
    return_tensors="pt",
    add_special_tokens=False,  # GPT-2에서는 보통 직접 제어
)

input_ids = enc["input_ids"]         # shape: [1, T]
attention_mask = enc["attention_mask"]  # shape: [1, T], 실제 토큰=1
```

### 12.4 토큰 ID -> 문자열 (디코딩)
```python
decoded = tokenizer.decode(input_ids[0], skip_special_tokens=False)
```

### 12.5 배치 인코딩 (여러 문장)
```python
texts = ["first sample", "second sample"]
batch = tokenizer(
    texts,
    padding=True,          # 같은 길이로 맞춤
    truncation=True,       # 길면 자름
    max_length=1024,       # n_ctx/n_positions와 맞춤
    return_tensors="pt",
)

input_ids = batch["input_ids"]          # [B, T]
attention_mask = batch["attention_mask"]  # [B, T]
```

### 12.6 학습용 전처리 핵심 규칙
- 규칙-01: GPT-2 컨텍스트 길이는 최대 1024 토큰이다. (`T <= 1024`)
- 규칙-02: 샘플 경계 구분을 위해 문서 끝에 `eos_token_id(50256)`를 붙인다.
- 규칙-03: next-token 학습은 `x=tokens[:-1]`, `y=tokens[1:]`로 만든다.
- 규칙-04: padding을 쓴다면 `attention_mask == 0` 위치는 loss에서 제외한다(예: label = -100).

### 12.7 토크나이저 전처리 구현 참고
- `examples/tokenizer/tokenizer_demo.py`에서 인코딩, 배치 패딩, LM shift(`x=t[:-1], y=t[1:]`) 흐름을 그대로 확인할 수 있다.

### 12.8 생성 시 토크나이저 연결
```python
prompt = "Once upon a time"
prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]

generated_ids = generate(prompt_ids, max_new_tokens=50, do_sample=True)
generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
```

## 13. `model.safetensors` 파싱 설계 (C++ 기준)

### 13.1 핵심 개념
- 파일 시작 `8byte`는 텐서 데이터 길이가 아니라 `JSON header 길이`다.
- 그 다음 `N byte`(N=header 길이)가 텐서 설계도(JSON)다.
- JSON에는 텐서별 `dtype`, `shape`, `data_offsets=[begin,end]`가 들어 있다.
- 실제 텐서 데이터 시작 위치는 `data_base = 8 + N`이다.
- 텐서 바이트 범위는 `data_base + begin`부터 `data_base + end` 직전까지다.

### 13.2 파싱 절차
- 1단계: 파일 오픈 후 첫 8바이트를 little-endian `uint64`로 읽는다.
- 2단계: 읽은 값 `N`만큼 JSON 헤더를 읽고 파싱한다.
- 3단계: `__metadata__`를 제외한 모든 텐서 항목을 테이블(맵)에 저장한다.
- 4단계: 텐서 요청 시 키 이름으로 `dtype/shape/offset`를 조회한다.
- 5단계: `data_base + begin`으로 포인터(또는 파일 오프셋)를 계산해 읽는다.

### 13.3 구현 시 주의사항
- 주의-01: `8byte + size` 패턴은 파일당 1회다. 텐서마다 반복되는 구조가 아니다.
- 주의-02: `data_offsets`는 파일 시작 기준이 아니라 데이터 버퍼 기준 오프셋이다.
- 주의-03: 오프셋 범위가 파일 길이를 넘지 않는지 반드시 검증한다.
- 주의-04: GPT-2의 Q/K/V는 보통 `c_attn` 하나에 합쳐져 있으므로 슬라이스 규칙을 별도로 둔다.
- 주의-05: `lm_head.weight`가 파일에 없을 수 있으므로 tie-weight(`wte.weight`) 규칙을 설계에 포함한다.

### 13.4 GPT-2 키 구조 참고
- 최상위: `wte.weight`, `wpe.weight`, `ln_f.weight`, `ln_f.bias`
- 레이어: `h.{i}.ln_1.*`, `h.{i}.attn.c_attn.*`, `h.{i}.attn.c_proj.*`, `h.{i}.ln_2.*`, `h.{i}.mlp.c_fc.*`, `h.{i}.mlp.c_proj.*`
- `config.json` 기준 레이어 인덱스는 `i=0..11`이다.
