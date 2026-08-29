# ruff: noqa: E741
import torch
from torch import Tensor
import math
import torch.nn.functional as F

# Seed
torch.manual_seed(0)

# Params
N = 1000
d = 128
M = 100000

# Matrices
Q, K, V = torch.randn((N, d)), torch.randn((N, d)), torch.randn((N, d))


def naive_attn(Q: Tensor, K: Tensor, V: Tensor) -> Tensor:
    S = Q @ K.transpose(-2, -1)
    assert S.shape == (N, N)
    P = S.softmax(dim=1)
    assert P.shape == (N, N)
    O = P @ V
    assert O.shape == (N, d)
    return O


def flash_attn(Q: Tensor, K: Tensor, V: Tensor, N: int, d: int, M: int) -> Tensor:
    # Block sizes
    B_c = math.ceil(M / (4 * d))
    B_r = min(B_c, d)

    # Total blocks
    T_r = math.ceil(N / B_r)
    T_c = math.ceil(N / B_c)

    # Outputs
    O = torch.zeros((N, d))
    l = torch.zeros((N, 1))
    m = torch.full((N, 1), float("-inf"))

    # Splitting
    Q_split = list(torch.split(Q, B_r))
    K_split = list(torch.split(K, B_c))
    V_split = list(torch.split(V, B_c))

    O_split = list(torch.split(O, B_r))
    l_split = list(torch.split(l, B_r))
    m_split = list(torch.split(m, B_r))

    # (should also pad last)
    def pad(t: list[Tensor], block_size: int):
        to_pad = block_size - t[-1].shape[0]
        if to_pad > 0:
            t[-1] = F.pad(t[-1], (0, 0, 0, to_pad))

    pad(Q_split, B_r)
    pad(K_split, B_c)
    pad(V_split, B_c)

    pad(O_split, B_r)
    pad(l_split, B_r)
    pad(m_split, B_r)

    for j in range(T_c):
        # load on SRAM
        K_j, V_j = K_split[j], V_split[j]
        assert K_j.shape == (B_c, d)
        assert V_j.shape == (B_c, d)

        for i in range(T_r):
            # load on SRAM
            Q_i, O_i, l_i, m_i = Q_split[i], O_split[i], l_split[i], m_split[i]
            assert Q_i.shape == (B_r, d)
            assert O_i.shape == (B_r, d)
            assert l_i.shape == (B_r, 1)
            assert m_i.shape == (B_r, 1)

            # compute things
            S_ij = Q_i @ K_j.transpose(-2, -1)
            assert S_ij.shape == (B_r, B_c)

            m_ij = S_ij.max(dim=1, keepdim=True).values
            assert m_ij.shape == (B_r, 1)

            P_ij = (S_ij - m_ij).exp()
            assert P_ij.shape == (B_r, B_c)

            l_ij = P_ij.sum(dim=1, keepdim=True)
            assert l_ij.shape == (B_r, 1)

            m_i_new = torch.max(m_i, m_ij)
            assert l_ij.shape == (B_r, 1)

            l_i_new = (torch.exp(m_i - m_i_new) * l_i) + (
                torch.exp(m_ij - m_i_new) * l_ij
            )
            assert l_i_new.shape == (B_r, 1)

            # write to HBM
            # TODO: diag function very ineffecient better to rely on broadcasting
            _norm_o_i = torch.diag(l_i.squeeze(1)) @ torch.exp(m_i - m_i_new) * O_i
            assert _norm_o_i.shape == (B_r, d)

            _pv = torch.exp(m_ij - m_i_new) * P_ij @ V_j
            assert _pv.shape == (B_r, d)

            _new_o_i = torch.diag(1 / l_i_new.squeeze(1)) @ (_norm_o_i + _pv)
            assert _new_o_i.shape == (B_r, d)

            O_split[i] = _new_o_i
            l_split[i] = l_i_new
            m_split[i] = m_i_new

    O = torch.cat(O_split)

    # mask off
    mask = torch.ones(O.size(0), dtype=torch.bool)
    mask[Q.size(0) :] = False
    O = O[mask]
    assert O.shape == (Q.size(0), d)

    return O


naive_attn_res = naive_attn(Q=Q, K=K, V=V)
flash_attn_res = flash_attn(Q=Q, K=K, V=V, N=N, d=d, M=M)

assert naive_attn_res.shape == flash_attn_res.shape
torch.testing.assert_close(flash_attn_res, naive_attn_res, rtol=1e-5, atol=1e-6)
