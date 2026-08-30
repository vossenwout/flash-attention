import torch
import triton
import triton.language as tl
from torch import Tensor

"""
Softmax kernel as warm up for flash attention :)
"""
M, N = 1000, 80

DEVICE = triton.runtime.driver.active.get_active_torch_device()
A = torch.randn((M, N), device=DEVICE, dtype=torch.float32)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 32}, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 64}, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 32}, num_warps=8),
        triton.Config({"BLOCK_SIZE_M": 64}, num_warps=8),
    ],
    key=["M", "N"],
)
@triton.jit
def _softmax(
    a_ptr,
    b_ptr,
    stride_m,
    stride_n,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    # load input block
    pid_m = tl.program_id(0)

    offset_am = tl.arange(0, BLOCK_SIZE_M) + BLOCK_SIZE_M * pid_m
    offset_an = tl.arange(0, triton.next_power_of_2(N))

    a_block_ptrs = a_ptr + (
        (offset_am * stride_m)[:, None] + (offset_an * stride_n)[None, :]
    )
    mask = (offset_am < M)[:, None] & (offset_an < N)[None, :]

    a_block = tl.load(a_block_ptrs, mask=mask, other=float("-inf"))
    tl.static_assert(a_block.shape == (BLOCK_SIZE_M, triton.next_power_of_2(N)))

    # softmax
    row_max = tl.max(a_block, axis=1, keep_dims=True)
    tl.static_assert(row_max.shape == (BLOCK_SIZE_M, 1))

    exp_block = tl.exp(a_block - row_max)
    tl.static_assert(a_block.shape == (BLOCK_SIZE_M, triton.next_power_of_2(N)))

    sum_block = tl.sum(exp_block, axis=1, keep_dims=True)
    tl.static_assert(sum_block.shape == (BLOCK_SIZE_M, 1))

    sm_block = exp_block / sum_block
    tl.static_assert(sm_block.shape == (BLOCK_SIZE_M, triton.next_power_of_2(N)))

    # store results
    offsets_bm = tl.arange(0, BLOCK_SIZE_M) + BLOCK_SIZE_M * pid_m
    offsets_bn = tl.arange(0, triton.next_power_of_2(N))

    b_block_ptrs = b_ptr + (
        (offsets_bm * stride_m)[:, None] + (offsets_bn * stride_n)[None, :]
    )
    mask_b = (offsets_bm < M)[:, None] & (offsets_bn < N)[None, :]

    tl.store(b_block_ptrs, value=sm_block, mask=mask_b)


def softmax(A: Tensor) -> Tensor:
    M, N = A.shape
    B = torch.empty((M, N), dtype=A.dtype, device=A.device)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_SIZE_M"]),)
    _softmax[grid](
        a_ptr=A,
        b_ptr=B,
        stride_m=A.stride(0),
        stride_n=A.stride(1),
        M=M,
        N=N,
    )
    return B


torch_sm = (A - A.max(dim=1, keepdim=True).values).softmax(dim=1)
triton_sm = softmax(A)
torch.testing.assert_close(torch_sm, triton_sm, rtol=1e-5, atol=1e-6)

torch_ms = triton.testing.do_bench(lambda: torch.softmax(A, dim=1))
triton_ms = triton.testing.do_bench(lambda: softmax(A))

print(f"Triton: {triton_ms:.3f} ms")
print(f"PyTorch: {torch_ms:.3f} ms")
