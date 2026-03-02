# Examples

This directory contains reference snippets you can read and run while building GPT-2 yourself.

## Structure

- `tokenizer/tokenizer_demo.py`: Hugging Face GPT-2 tokenizer basics (encode/decode, batch, LM shift)
- `cpp_cuda_extension/demo.py`: Python calling a custom C++/CUDA op through a PyTorch extension
- `forloop_gpt2/gpt2_forloop_ops.py`: loop-based GPT-2 core ops for CUDA kernel indexing reference
- `tokenized_cpp/data/prompts.txt`: pre-made sentence/question inputs for tokenizer tests

## Run

```bash
python3 examples/tokenizer/tokenizer_demo.py
python3 examples/forloop_gpt2/gpt2_forloop_ops.py
python3 examples/cpp_cuda_extension/demo.py
```

## Notes

- Install dependencies first: `pip install torch transformers tokenizers`
- `forloop_gpt2` is intentionally slow because all core ops use explicit loops.
- `cpp_cuda_extension` builds a C++ extension at runtime.
- If CUDA is available and a CUDA toolkit is installed, it also builds and runs a CUDA kernel.
- If CUDA is not available, the sample still runs in CPU mode.
