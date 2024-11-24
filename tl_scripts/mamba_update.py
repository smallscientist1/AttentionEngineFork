import torch
import torch.nn as nn
from tvm import tl
import tvm.tl.language as T
from einops import rearrange, repeat

def selective_scan_update_fwd(batch, nheads, ngroups, headdim, dstate, block_M, block_Dstate):
    dtype = "float16"
    accum_dtype = "float"
    p = 1.44269504
    assert dstate == block_Dstate
    @T.prim_func
    def main(
        state: T.Buffer((batch, nheads, headdim, dstate), dtype),
        x: T.Buffer((batch, nheads, headdim), dtype),
        dt: T.Buffer((batch, nheads, headdim), dtype),
        A: T.Buffer((nheads, headdim, dstate), dtype),
        B: T.Buffer((batch, ngroups, dstate), dtype),
        C: T.Buffer((batch, ngroups, dstate), dtype),
        Output: T.Buffer((batch, nheads, headdim), dtype)
    ):
        with T.Kernel(T.ceildiv(headdim, block_M), batch, nheads, threads=128) as (bx, by, bz):
            state_shared = T.alloc_shared((block_M, block_Dstate), dtype)
            state_local = T.alloc_fragment((block_M, block_Dstate), accum_dtype)
            # new_state_local = T.alloc_fragment((block_M, block_Dstate), accum_dtype)
            x_shared = T.alloc_shared((block_M), dtype)
            x_local = T.alloc_fragment((block_M), accum_dtype)
            dt_shared = T.alloc_shared((block_M), dtype)
            dt_local = T.alloc_fragment((block_M), accum_dtype)
            A_shared = T.alloc_shared((block_M, block_Dstate), dtype)
            A_local = T.alloc_fragment((block_M, block_Dstate), accum_dtype)
            dA_local = T.alloc_fragment((block_M, block_Dstate), accum_dtype)
            B_shared = T.alloc_shared((block_Dstate), dtype)
            C_shared = T.alloc_shared((block_Dstate), dtype)
            C_local = T.alloc_fragment((block_Dstate), accum_dtype)
            B_local = T.alloc_fragment((block_Dstate), accum_dtype)
            dB_local = T.alloc_fragment((block_M, block_Dstate), accum_dtype)
            state_sum_local = T.alloc_fragment((block_M), accum_dtype)

            batch_idx = by
            head_idx = bz
            m_idx = bx

            # T.annotate_layout({
            #     new_state_local: tl.layout.make_swizzled_layout(state_shared),
            # })

            T.copy(state[batch_idx, head_idx, m_idx * block_M : (m_idx + 1) * block_M, :], state_shared)
            T.copy(state_shared, state_local)
            T.copy(x[batch_idx, head_idx, m_idx * block_M : (m_idx + 1) * block_M], x_shared)
            T.copy(x_shared, x_local)
            # Not TIE_HDIM
            T.copy(dt[batch_idx, head_idx, m_idx * block_M : (m_idx + 1) * block_M], dt_shared)
            T.copy(dt_shared, dt_local)
            T.copy(A[head_idx, m_idx * block_M : (m_idx + 1) * block_M, :], A_shared)
            T.copy(A_shared, A_local)
            for i, j in T.Parallel(block_M, block_Dstate):
                dA_local[i, j] = T.exp2(A_local[i, j] * dt_local[i] * p)
            T.copy(B[batch_idx, bz // (nheads // ngroups), :], B_shared)
            T.copy(B_shared, B_local)
            T.copy(C[batch_idx, bz // (nheads // ngroups), :], C_shared)
            T.copy(C_shared, C_local)
            for i, j in T.Parallel(block_M, block_Dstate):
                dB_local[i, j] = B_local[j] * dt_local[i]
            for i, j in T.Parallel(block_M, block_Dstate):
                state_local[i, j] *= dA_local[i, j]
            for i, j in T.Parallel(block_M, block_Dstate):
                state_local[i, j] += dB_local[i, j] * x_local[i]
            for i, j in T.Parallel(block_M, block_Dstate):
                state_local[i, j] *= C_local[j]
            T.reduce_sum(state_local, state_sum_local, dim=1)
            T.copy(state_sum_local, Output[batch_idx, head_idx, m_idx * block_M : (m_idx + 1) * block_M])

    return main

def selective_state_update_ref(state, x, dt, A, B, C, D=None, z=None, dt_bias=None, dt_softplus=False):
    """
    Argument:
        state: (batch, dim, dstate) or (batch, nheads, dim, dstate)
        x: (batch, dim) or (batch, nheads, dim)
        dt: (batch, dim) or (batch, nheads, dim)
        A: (dim, dstate) or (nheads, dim, dstate)
        B: (batch, dstate) or (batch, ngroups, dstate)
        C: (batch, dstate) or (batch, ngroups, dstate)
        D: (dim,) or (nheads, dim)
        z: (batch, dim) or (batch, nheads, dim)
        dt_bias: (dim,) or (nheads, dim)
    Return:
        out: (batch, dim) or (batch, nheads, dim)
    """
    has_heads = state.dim() > 3
    if state.dim() == 3:
        state = state.unsqueeze(1)
    if x.dim() == 2:
        x = x.unsqueeze(1)
    if dt.dim() == 2:
        dt = dt.unsqueeze(1)
    if A.dim() == 2:
        A = A.unsqueeze(0)
    if B.dim() == 2:
        B = B.unsqueeze(1)
    if C.dim() == 2:
        C = C.unsqueeze(1)
    if D is not None and D.dim() == 1:
        D = D.unsqueeze(0)
    if z is not None and z.dim() == 2:
        z = z.unsqueeze(1)
    if dt_bias is not None and dt_bias.dim() == 1:
        dt_bias = dt_bias.unsqueeze(0)
    batch, nheads, dim, dstate = state.shape
    assert x.shape == (batch, nheads, dim)
    assert dt.shape == x.shape
    assert A.shape == (nheads, dim, dstate)
    ngroups = B.shape[1]
    assert nheads % ngroups == 0, "nheads must be divisible by ngroups"
    assert B.shape == (batch, ngroups, dstate)
    assert C.shape == B.shape
    if D is not None:
        assert D.shape == (nheads, dim)
    if z is not None:
        assert z.shape == x.shape
    if dt_bias is not None:
        assert dt_bias.shape == (nheads, dim)
        dt = dt + dt_bias
    dt = F.softplus(dt) if dt_softplus else dt
    dA = torch.exp(rearrange(dt, "b h d -> b h d 1") * A)  # (batch, nheads, dim, dstate)
    B = repeat(B, "b g n -> b (g h) n", h=nheads // ngroups)  # (batch, nheads, dstate)
    C = repeat(C, "b g n -> b (g h) n", h=nheads // ngroups)  # (batch, nheads, dstate)
    dB = rearrange(dt, "b h d -> b h d 1") * rearrange(B, "b h n -> b h 1 n")  # (batch, nheads, dim, dstate)
    state_ = state * dA + dB * rearrange(x, "b h d -> b h d 1")  # (batch, dim, dstate
    out = torch.einsum("bhdn,bhn->bhd", state_.to(C.dtype), C)
    if D is not None:
        out += (x * D).to(out.dtype)
    out = (out if z is None else out * F.silu(z)).to(x.dtype)
    if not has_heads:
        out = out.squeeze(1)
    return out

def selective_state_update_triton(state, x, dt, A, B, C):
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
    out = selective_state_update(state, x, dt, A, B, C)
    return out

if __name__ == "__main__":
    batch, nheads, ngroups, headdim, dstate, block_M, block_Dstate = 128, 64, 1, 64, 128, 64, 128
    program = selective_scan_update_fwd(batch, nheads, ngroups, headdim, dstate, block_M, block_Dstate)
    mod, params = tl.lower(program)

    mod = tl.Profiler(mod, params, [6], tl.TensorSupplyType.Normal)
    mod.assert_allclose(selective_state_update_ref, rtol=0.01, atol=0.01)
    print("All checks pass.")

    latency = mod.do_bench(selective_state_update_triton, n_warmup=10, n_repeat=10)
    print("{:.4f} ms".format(latency))
    latency = mod.do_bench(mod.func, n_warmup=10, n_repeat=10, profiler="auto")
    print("{:.4f} ms".format(latency))