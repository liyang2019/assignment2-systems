import torch
import torch.nn.functional as F
from einops import einsum
import triton
import triton.language as tl


class FlashAttention2PyTorch(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        is_causal: bool = False,
    ):
        ctx.scale = Q.shape[-1] ** (-0.5)
        ctx.is_causal = is_causal
        Bq = 16
        Bk = 16
        Tq = Q.shape[-2] // Bq
        Tk = K.shape[-2] // Bk

        O = Q.new_zeros(Q.shape)
        L = Q.new_zeros(Q.shape[:-1], dtype=torch.float32)
        for i in range(Tq):
            Qi = Q[..., i * Bq : (i + 1) * Bq, :]
            Oi = Qi.new_zeros(Qi.shape)
            li = Qi.new_zeros(Qi.shape[:-1])
            mi = Qi.new_full(Qi.shape[:-1], float("-inf"))
            for j in range(Tk):
                Kj = K[..., j * Bk : (j + 1) * Bk, :]
                Vj = V[..., j * Bk : (j + 1) * Bk, :]
                Sij = einsum(Qi, Kj, "... q d, ... k d -> ... q k") * ctx.scale
                prev_mi = mi
                mi = torch.maximum(mi, Sij.max(dim=-1)[0])
                Pij = torch.exp(Sij - mi.unsqueeze(-1))
                exp_dmi = torch.exp(prev_mi - mi)
                li = exp_dmi * li + Pij.sum(dim=-1)
                Oi = exp_dmi.unsqueeze(-1) * Oi + Pij @ Vj
            O[..., i * Bq : (i + 1) * Bq, :] = Oi / li.unsqueeze(-1)
            L[..., i * Bq : (i + 1) * Bq] = mi + torch.log(li)
        ctx.save_for_backward(Q, K, V, O, L)
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        S = einsum(Q, K, '... q d, ... k d -> ... q k') * ctx.scale
        if ctx.is_causal:
            S += torch.triu(Q.new_full((ctx.Nq, ctx.Nk), -float("inf")), diagonal=1)
        P = torch.exp(S - L[..., None])
        dV = einsum(P, dO, '... q k, ... q d -> ... k d')
        dP = einsum(dO, V, '... q d, ... k d -> ... q k')
        D = einsum(O, dO, '... q d, ... q d -> ... q')
        dS = P * (dP - D[..., None])
        dQ = einsum(dS, K, '... q k, ... k d -> ... q d') * ctx.scale
        dK = einsum(dS, Q, '... q k, ... q d -> ... k d') * ctx.scale
        return dQ, dK, dV, None

@triton.jit
def flash_fwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )

    Q = tl.load(Q_block_ptr, boundary_check=(1, 0), padding_option="zero")
    O = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    L = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    M = tl.full((Q_TILE_SIZE,), value=float('-inf'), dtype=tl.float32)

    i = query_tile_index
    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        K = tl.load(K_block_ptr, boundary_check=(1, 0), padding_option="zero")
        V = tl.load(V_block_ptr, boundary_check=(1, 0), padding_option="zero")
        S = tl.dot(Q, K.T) * scale
        if is_causal:
            q_pos = i * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            k_pos = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            S = tl.where(q_pos[:, None] >= k_pos[None, :], S, -1e6)
        prev_M = M
        M = tl.maximum(M, tl.max(S, axis=-1))
        P = tl.exp(S - M[:, None])
        L *= tl.exp(prev_M - M)
        L += tl.sum(P, axis=-1)
        O *= tl.exp(prev_M - M)[:, None]
        O = tl.dot(P.to(V.dtype), V, acc=O)
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
    tl.store(O_block_ptr, O.to(O_block_ptr.type.element_ty) / L[:, None])
    tl.store(L_block_ptr, M + tl.log(L))


class FlashAttention2Triton(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=True):

        O_shape = Q.shape
        ctx.Nq = Q.shape[-2]
        ctx.Nk = K.shape[-2]
        ctx.D = K.shape[-1]
        ctx.Bq = 16
        ctx.Bk = 16
        ctx.scale = ctx.D**(-0.5)
        ctx.B = Q.shape[0]
        ctx.is_causal = is_causal

        Q = Q.view(-1, ctx.Nq, ctx.D)
        K = K.view(-1, ctx.Nk, ctx.D)
        V = V.view(-1, ctx.Nk, ctx.D)

        O = Q.new_zeros(Q.shape)
        L = Q.new_zeros(Q.shape[:-1])

        flash_fwd_kernel[(triton.cdiv(ctx.Nq, ctx.Bq), ctx.B)](
            Q, K, V, O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            ctx.Nq, ctx.Nk,
            ctx.scale,
            ctx.D,
            ctx.Bq,
            ctx.Bk,
            is_causal,
        )

        ctx.save_for_backward(Q, K, V, O, L)

        return O.view(O_shape)

    @torch.compile
    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        S = einsum(Q, K, '... q d, ... k d -> ... q k') * ctx.scale
        if ctx.is_causal:
            S += torch.triu(Q.new_full((ctx.Nq, ctx.Nk), -float("inf")), diagonal=1)
        P = torch.exp(S - L[..., None])
        dV = einsum(P, dO, '... q k, ... q d -> ... k d')
        dP = einsum(dO, V, '... q d, ... k d -> ... q k')
        D = einsum(O, dO, '... q d, ... q d -> ... q')
        dS = P * (dP - D[..., None])
        dQ = einsum(dS, K, '... q k, ... k d -> ... q d') * ctx.scale
        dK = einsum(dS, Q, '... q k, ... q d -> ... k d') * ctx.scale
        return dQ, dK, dV, None


def test_timing_flash_forward_backward():
    n_heads = 16
    d_head = 64
    sequence_length = 16384
    q, k, v = torch.randn(
        3,
        n_heads,
        sequence_length,
        d_head,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )

    flash = torch.compile(FlashAttention2PyTorch.apply)
    # flash = torch.compile(FlashAttention2Triton.apply)

    def flash_forward_backward():
        o = flash(q, k, v, True)
        loss = o.sum()
        # loss.backward()

    results = triton.testing.do_bench(flash_forward_backward, rep=10000, warmup=1000)
    print(results)


if __name__ == "__main__":
    # Q = torch.randn(2, 4, 32, 16).to("cuda")
    # K = torch.randn(2, 4, 32, 16).to("cuda")
    # V = torch.randn(2, 4, 32, 16).to("cuda")
    # O_pt = FlashAttention2PyTorch.apply(Q, K, V)
    # O_tr = FlashAttention2Triton.apply(Q, K, V)
    # print(O_pt - O_tr)
    test_timing_flash_forward_backward()
