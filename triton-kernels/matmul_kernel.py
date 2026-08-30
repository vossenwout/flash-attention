import torch
from torch import Tensor
import triton
import triton.language as tl
from triton.runtime import driver

"""
Matmul kernel I made to learn triton. My first kernel ever :)
"""

torch.manual_seed(0)


DEVICE = triton.runtime.driver.active.get_active_torch_device()

# Vars
M = 1000
K = 300
N = 500

A = torch.randn((M, K), device=DEVICE)
B = torch.randn((K, N), device=DEVICE)

naive_matmul = A @ B

# 'max_shared_mem': 65536,
# 'max_num_regs': 65536,
# 'multiprocessor_count': 40,
# 'warpSize': 32,
# 'sm_clock_rate': 1590000,
# 'mem_clock_rate': 5001000,
# 'mem_bus_width': 256

device_props = driver.active.utils.get_device_properties(DEVICE.index)  # ty: ignore


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32}, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32}, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64},
            num_warps=8,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _matmul(
    a_ptr,
    b_ptr,
    c_ptr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # pid's moeten er ook nog in komen
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # wrap around but need to mask later?
    offset_am = tl.arange(0, BLOCK_SIZE_M) + pid_m * BLOCK_SIZE_M
    offset_bn = tl.arange(0, BLOCK_SIZE_N) + pid_n * BLOCK_SIZE_N

    tl.static_assert(offset_am.shape == (BLOCK_SIZE_M,))
    tl.static_assert(offset_bn.shape == (BLOCK_SIZE_N,))

    c_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(triton.cdiv(K, BLOCK_SIZE_K)):
        # each iteration move k
        offset_k = tl.arange(0, BLOCK_SIZE_K) + (k * BLOCK_SIZE_K)

        # mask invalid
        mask_a = (offset_am < M)[:, None] & (offset_k < K)[None, :]
        mask_b = (offset_k < K)[:, None] & (offset_bn < N)[None, :]

        tl.static_assert(mask_a.shape == (BLOCK_SIZE_M, BLOCK_SIZE_K))
        tl.static_assert(mask_b.shape == (BLOCK_SIZE_K, BLOCK_SIZE_N))

        a_block_ptrs = a_ptr + (
            (offset_am * stride_am)[:, None] + ((offset_k * stride_ak)[None, :])
        )
        b_block_ptrs = b_ptr + (
            (offset_k * stride_bk)[:, None] + ((offset_bn * stride_bn)[None, :])
        )

        tl.static_assert(a_block_ptrs.shape == (BLOCK_SIZE_M, BLOCK_SIZE_K))
        tl.static_assert(b_block_ptrs.shape == (BLOCK_SIZE_K, BLOCK_SIZE_N))

        a_block = tl.load(a_block_ptrs, mask=mask_a, other=0)
        b_block = tl.load(b_block_ptrs, mask=mask_b, other=0)

        tl.static_assert(a_block.shape == (BLOCK_SIZE_M, BLOCK_SIZE_K))
        tl.static_assert(b_block.shape == (BLOCK_SIZE_K, BLOCK_SIZE_N))

        c_acc += tl.dot(a_block, b_block)

    offset_cm = tl.arange(0, BLOCK_SIZE_M) + (pid_m * BLOCK_SIZE_M)
    offset_cn = tl.arange(0, BLOCK_SIZE_N) + (pid_n * BLOCK_SIZE_N)

    mask_c = (offset_cm < M)[:, None] & (offset_cn < N)[None, :]
    c_block_ptrs = c_ptr + (
        (offset_cm * stride_cm)[:, None] + (offset_cn * stride_cn)[None, :]
    )

    tl.static_assert(c_block_ptrs.shape == (BLOCK_SIZE_M, BLOCK_SIZE_N))

    tl.store(c_block_ptrs, value=c_acc, mask=mask_c)


def matmul(A: Tensor, B: Tensor) -> Tensor:
    M, K = A.shape
    K_B, N = B.shape
    assert K == K_B, f"A and B have incompatible shapes in K dim: {K} vs {K_B}"

    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE_M"]),
        triton.cdiv(N, meta["BLOCK_SIZE_N"]),
    )

    _matmul[grid](
        a_ptr=A,
        b_ptr=B,
        c_ptr=C,
        stride_am=A.stride(0),
        stride_ak=A.stride(1),
        stride_bk=B.stride(0),
        stride_bn=B.stride(1),
        stride_cm=C.stride(0),
        stride_cn=C.stride(1),
        M=M,
        N=N,
        K=K,
    )
    return C


triton_matmul = matmul(A, B)
torch.testing.assert_close(
    triton_matmul,
    naive_matmul,
    rtol=1e-5,
    atol=1e-6,
)

torch_ms = triton.testing.do_bench(lambda: A @ B)
triton_ms = triton.testing.do_bench(lambda: matmul(A, B))

print(f"Triton: {triton_ms:.3f} ms")
print(f"PyTorch: {torch_ms:.3f} ms")


def tflops(ms: float, M: int, N: int, K: int) -> float:
    return 2.0 * M * N * K / (ms * 1e-3) / 1e12


print(f"Triton: {triton_ms:.3f} ms, {tflops(triton_ms, M, N, K):.2f} TFLOPS")
print(f"PyTorch: {torch_ms:.3f} ms, {tflops(torch_ms, M, N, K):.2f} TFLOPS")
