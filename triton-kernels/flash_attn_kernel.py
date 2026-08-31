from torch import Tensor
import triton
import triton.language as tl
import torch

"""
Flash attention implementation in triton + comparison against pytorch
"""

DEVICE = triton.runtime.driver.active.get_active_torch_device()

# 'max_shared_mem': 65536,
# 'max_num_regs': 65536,
# 'multiprocessor_count': 40,
# 'warpSize': 32,
# 'sm_clock_rate': 1590000,
# 'mem_clock_rate': 5001000,
# 'mem_bus_width': 256

DEVICE_PROPS = triton.runtime.driver.active.utils.get_device_properties(DEVICE.index)  # ty: ignore

# Params
# N = 1000
N = 8192 * 6
d = 128

# Matrices
Q, K, V = (
    torch.randn((N, d), device=DEVICE, dtype=torch.float16),
    torch.randn((N, d), device=DEVICE, dtype=torch.float16),
    torch.randn((N, d), device=DEVICE, dtype=torch.float16),
)


def naive_attn(Q: Tensor, K: Tensor, V: Tensor) -> Tensor:
    scale = d**-0.5
    S = (Q @ K.transpose(-2, -1)) * scale
    assert S.shape == (N, N)
    P = S.softmax(dim=1)
    assert P.shape == (N, N)
    O = P @ V
    assert O.shape == (N, d)
    return O


@triton.autotune(
    configs=[
        triton.Config(
            {"B_r": 16, "B_c": 16},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"B_r": 16, "B_c": 32},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"B_r": 32, "B_c": 32},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"B_r": 32, "B_c": 64},
            num_warps=8,
            num_stages=3,
        ),
    ],
    key=["N", "d"],
)
@triton.jit
def _flash_attn(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    stride_qn,
    stride_qd,
    stride_kn,
    stride_kd,
    stride_vn,
    stride_vd,
    stride_on,
    stride_od,
    sm_scale,
    N: tl.constexpr,
    d: tl.constexpr,
    B_c: tl.constexpr,
    B_r: tl.constexpr,
):
    pid = tl.program_id(0)

    # Load Q_i
    offset_qn = tl.arange(0, B_r) + (pid * B_r)
    offset_qd = tl.arange(0, triton.next_power_of_2(d))
    mask_q = (offset_qn < N)[:, None] & (offset_qd < d)[None, :]
    block_ptrs_q = q_ptr + (
        (offset_qn * stride_qn)[:, None] + (offset_qd * stride_qd)[None, :]
    )
    Q_i = tl.load(block_ptrs_q, mask=mask_q, other=0)
    tl.static_assert(Q_i.shape == (B_r, triton.next_power_of_2(d)))

    # init
    l_i = tl.zeros((B_r, 1), dtype=tl.float32)
    m_i = tl.full((B_r, 1), -float("inf"), dtype=tl.float32)
    O_i = tl.zeros((B_r, d), dtype=tl.float32)

    for j in range(0, N, B_c):
        # Load K_j
        offset_kn = j + tl.arange(0, B_c)
        offset_kd = tl.arange(0, triton.next_power_of_2(d))
        mask_k = (offset_kn < N)[:, None] & (offset_kd < d)[None, :]
        block_ptrs_k = k_ptr + (
            (offset_kn * stride_kn)[:, None] + (offset_kd * stride_kd)[None, :]
        )
        K_j = tl.load(block_ptrs_k, mask=mask_k, other=0)
        tl.static_assert(K_j.shape == (B_c, triton.next_power_of_2(d)))

        # Load V_j
        offset_vn = j + tl.arange(0, B_c)
        offset_vd = tl.arange(0, triton.next_power_of_2(d))
        mask_v = (offset_vn < N)[:, None] & (offset_vd < d)[None, :]
        block_ptrs_v = v_ptr + (
            (offset_vn * stride_vn)[:, None] + (offset_vd * stride_vd)[None, :]
        )
        V_j = tl.load(block_ptrs_v, mask=mask_v, other=0)
        tl.static_assert(V_j.shape == (B_c, triton.next_power_of_2(d)))

        # Compute attention
        S_ij = tl.dot(Q_i, tl.trans(K_j))
        S_ij *= sm_scale
        tl.static_assert(S_ij.shape == (B_r, B_c))

        S_ij = tl.where(
            (offset_kn < N)[None, :], S_ij, float("-inf")
        )  # masking necessary

        m_ij = tl.max(S_ij, axis=1, keep_dims=True)
        tl.static_assert(m_ij.shape == (B_r, 1))

        P_ij = tl.exp(S_ij - m_ij)
        tl.static_assert(P_ij.shape == (B_r, B_c))

        l_ij = tl.sum(P_ij, axis=1, keep_dims=True)
        tl.static_assert(m_ij.shape == (B_r, 1))

        m_i_new = tl.maximum(m_i, m_ij)
        tl.static_assert(m_i_new.shape == (B_r, 1))

        l_i_new = (tl.exp(m_i - m_i_new) * l_i) + (tl.exp(m_ij - m_i_new) * l_ij)
        tl.static_assert(l_i_new.shape == (B_r, 1))

        _norm_o_i = l_i * tl.exp(m_i - m_i_new) * O_i
        tl.static_assert(_norm_o_i.shape == (B_r, d))

        _pv = tl.exp(m_ij - m_i_new) * tl.dot(P_ij.to(tl.float16), V_j)
        tl.static_assert(_pv.shape == (B_r, d))

        _new_o_i = (1 / l_i_new) * (_norm_o_i + _pv)
        tl.static_assert(O_i.shape == (B_r, d))

        O_i = _new_o_i
        l_i = l_i_new
        m_i = m_i_new

    # Store O_i to HBM
    offset_on = tl.arange(0, B_r) + (pid * B_r)
    offset_od = tl.arange(0, triton.next_power_of_2(d))
    mask_o = (offset_on < N)[:, None] & (offset_od < d)[None, :]
    block_ptrs_o = o_ptr + (
        (offset_on * stride_on)[:, None] + (offset_od * stride_od)[None, :]
    )
    tl.store(block_ptrs_o, value=O_i, mask=mask_o)


def flash_attn(Q: Tensor, K: Tensor, V: Tensor) -> Tensor:
    N, d = Q.shape
    Nk, dk = K.shape
    Nv, dv = V.shape
    assert (N, d) == (Nk, dk) and (N, d) == (Nv, dv)

    # allocate on HBM
    O = torch.zeros((N, d), device=Q.device, dtype=Q.dtype)

    grid = lambda meta: (triton.cdiv(N, meta["B_r"]),)

    _flash_attn[grid](
        q_ptr=Q,
        k_ptr=K,
        v_ptr=V,
        o_ptr=O,
        stride_qn=Q.stride(0),
        stride_qd=Q.stride(1),
        stride_kn=K.stride(0),
        stride_kd=K.stride(1),
        stride_vn=V.stride(0),
        stride_vd=V.stride(1),
        stride_on=O.stride(0),
        stride_od=O.stride(1),
        sm_scale=d**-0.5,
        N=N,
        d=d,
    )
    return O


torch_attn = naive_attn(Q=Q, K=K, V=V)
triton_attn = flash_attn(Q=Q, K=K, V=V)
torch.testing.assert_close(torch_attn, triton_attn, rtol=1e-2, atol=1e-2)

torch_ms = triton.testing.do_bench(lambda: naive_attn(Q, K, V))
triton_ms = triton.testing.do_bench(lambda: flash_attn(Q, K, V))

print(f"Naive PyTorch: {torch_ms:.3f} ms")
print(f"Flash Triton:  {triton_ms:.3f} ms")
print(f"Speedup:       {torch_ms / triton_ms:.2f}x")
