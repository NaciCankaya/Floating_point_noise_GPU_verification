Same model, all INT4, just different quant methods.

Example code:



```python
llm = LLM(
    model=MODEL_NAME,
    quantization=QUANTIZATION,          # "gptq"
    dtype="float16",                    # <— force FP16 activations
    tensor_parallel_size=TENSOR_PARALLEL_SIZE,
    max_model_len=MAX_MODEL_LEN,
    gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    trust_remote_code=True,
    seed=SEED,
    enforce_eager=True,
    enable_prefix_caching=False,
)
```

