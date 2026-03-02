# Tokenizer Example

Simple GPT-2 tokenizer reference using Hugging Face.

## Run

```bash
python3 examples/tokenizer/tokenizer_demo.py
```

## What it shows

- Loading `openai-community/gpt2` tokenizer
- Handling missing `pad_token` in GPT-2
- Single encode/decode
- Batch tokenize with padding/truncation
- LM shift example (`x=t[:-1]`, `y=t[1:]`)
