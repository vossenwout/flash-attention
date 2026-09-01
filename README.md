# Flash Attention

Flash attention implementation from scratch. Purpose is to gain a better understanding in what attention is and also learn Triton :)

For implementation look at:
`/triton-kernels/flash_attn_kernel.py`

## Benchmark

I compared my flash_attn_kernel with the naive_attn kernel found in the same file.

### Setup

- GPU: NVIDIA Tesla T4
- PyTorch CUDA: 12.4
- Head dimension: `d = 128`
- Data type: `float16`

### Results

| Sequence length | Naive PyTorch | Triton FlashAttention | Speedup |
|---:|---:|---:|---:|
| 4,096 | 1.075 ms | 0.912 ms | **1.18×** |
| 16,184 | 16.558 ms | 13.793 ms | **1.20×** |
| 48,552 | 179.591 ms | 126.678 ms | **1.42×** |

The relative benefit increases at larger sequence lengths. The naive implementation explicitly computes and stores the full `N × N` attention matrix:
