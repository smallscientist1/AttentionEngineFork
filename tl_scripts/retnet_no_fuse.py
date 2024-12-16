import argparse
import torch
from tvm import tl
import tvm.tl.language as T
from functools import partial

# def ref_program(Q, K, V, mask):
#     qk = torch.einsum('bqhd,bkhd->bhqk', Q, K)
#     qkm = qk * mask
#     r = qkm.detach().abs().sum(dim=-1, keepdim=True).clamp(min=1.0)
#     o = torch.einsum('bhqk,bkhd->bqhd', qkm/r, V)
#     return o.to(dtype=torch.float16)

def ref_program0(Q, K, mask):
    qk = torch.einsum('bqhd,bkhd->bhqk', Q, K)
    qkm = qk * mask
    return qkm.to(dtype=torch.float16)

def retnet0(batch, heads, seq_len, dim_qk, dim_v, block_M, block_N, block_K):
    dtype = "float16"
    accum_dtype = "float"

    @T.prim_func
    def main(
        Q: T.Buffer([batch, seq_len, heads, dim_qk], dtype), 
        K: T.Buffer([batch, seq_len, heads, dim_qk], dtype), 
        mask: T.Buffer([heads, seq_len, seq_len], dtype),
        Output: T.Buffer([batch, heads, seq_len, seq_len], dtype)):
        with T.Kernel(T.ceildiv(seq_len, block_M), T.ceildiv(seq_len, block_N), heads * batch, threads=128) as (bx, by, bz):
            Q_shared = T.alloc_shared((block_M, block_K), dtype)
            K_shared = T.alloc_shared((block_N, block_K), dtype)
            O_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            mask_local = T.alloc_shared((block_M, block_N), dtype)

            bid = bz // heads
            hid = bz % heads

            T.clear(O_local)
            for k in T.Pipelined(T.ceildiv(dim_qk, block_K), num_stages=2):
                T.copy(Q[bid, bx * block_M : (bx + 1) * block_M, hid, k * block_K : (k + 1) * block_K], Q_shared)
                T.copy(K[bid, by * block_N : (by + 1) * block_N, hid, k * block_K : (k + 1) * block_K], K_shared)
                T.gemm(Q_shared, K_shared, O_local, transpose_B=True)
            T.copy(mask[hid, bx * block_M : (bx + 1) * block_M, by * block_N : (by + 1) * block_N], mask_local)
            for i, j in T.Parallel(block_M, block_N):
                O_local[i, j] *= mask_local[i, j]
            T.copy(O_local, Output[bid, hid, bx * block_M : (bx + 1) * block_M, by * block_N : (by + 1) * block_N])

    return main

def ref_program1(qkm):
    r = qkm.detach().abs().sum(dim=-1, keepdim=True).clamp(min=1.0)
    return (qkm/r).to(dtype=torch.float16)

def retnet1(batch, heads, seq_len, block_M):
    dtype = "float16"
    accum_dtype = "float"

    @T.prim_func
    def main(
        QKM: T.Buffer([batch, heads, seq_len, seq_len], dtype), 
        Output: T.Buffer([batch, heads, seq_len, seq_len], dtype)):
        with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch, threads=128) as (bx, by, bz):
            qkm_shared = T.alloc_shared((block_M, seq_len), dtype)
            qkm_local = T.alloc_fragment((block_M, seq_len), dtype)
            r_local = T.alloc_fragment((block_M), accum_dtype)

            T.copy(QKM[bz, by, bx * block_M : (bx + 1) * block_M, :], qkm_shared)
            T.copy(qkm_shared, qkm_local)
            T.reduce_abssum(qkm_local, r_local, dim=1)
            for i, j in T.Parallel(block_M, seq_len):
                qkm_local[i, j] /= r_local[i]
            T.copy(qkm_local, Output[bz, by, bx * block_M : (bx + 1) * block_M, :])

    return main

def ref_program2(qkm, V):
    o = torch.einsum('bhqk,bkhd->bqhd', qkm, V)
    return o.to(dtype=torch.float16)

def retnet2(batch, heads, seq_len, dim_v, block_M, block_N, block_K):
    dtype = "float16"
    accum_dtype = "float"

    @T.prim_func
    def main(
        QKM: T.Buffer([batch, heads, seq_len, seq_len], dtype), 
        V: T.Buffer([batch, seq_len, heads, dim_v], dtype), 
        Output: T.Buffer([batch, seq_len, heads, dim_v], dtype)):
        with T.Kernel(T.ceildiv(seq_len, block_M), T.ceildiv(dim_v, block_N), heads * batch, threads=128 * 2) as (bx, by, bz):
            qkm_shared = T.alloc_shared((block_M, block_K), dtype)
            V_shared = T.alloc_shared((block_K, block_N), dtype)
            O_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            bid = bz // heads
            hid = bz % heads

            T.clear(O_local)
            for k in T.Pipelined(T.ceildiv(seq_len, block_K), num_stages=2):
                T.copy(QKM[bid, hid, bx * block_M : (bx + 1) * block_M, k * block_K : (k + 1) * block_K], qkm_shared)
                T.copy(V[bid, k * block_K : (k + 1) * block_K, hid, by * block_N : (by + 1) * block_N], V_shared)
                T.gemm(qkm_shared, V_shared, O_local)
            T.copy(O_local, Output[bid, bx * block_M : (bx + 1) * block_M, hid, by * block_N : (by + 1) * block_N])

    return main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=1, help='Batch size')
    parser.add_argument('--h', type=int, default=10, help='Number of heads')
    parser.add_argument('--n_ctx', type=int, default=4096, help='Context size')
    parser.add_argument('--dim_qk', type=int, default=256, help='Head dimension')
    parser.add_argument('--dim_v', type=int, default=448, help='Head dimension')
    args = parser.parse_args()
    BATCH, H, N_CTX, dim_qk, dim_v = args.batch, args.h, args.n_ctx, args.dim_qk, args.dim_v
    total_flops = 2.0 * BATCH * H * N_CTX * N_CTX * (dim_qk + dim_v)
    # BLOCK_M = 64
    # BLOCK_N = 128
    # BLOCK_K = 64
    # program = retnet0(BATCH, H, N_CTX, dim_qk, dim_v, BLOCK_M, BLOCK_N, BLOCK_K)
    # mod, params = tl.lower(program)
    # mod = tl.Profiler(mod, params, [3], tl.TensorSupplyType.Integer)
    # mod.assert_allclose(ref_program0, rtol=0.01, atol=0.01)
    # latency = mod.do_bench(mod, n_warmup=10, n_repeat=10, profiler="torch")
    # print("tl: {:.2f} ms".format(latency))
    # print("tl: {:.2f} TFlops".format(total_flops / latency * 1e-9))

    # BLOCK_M = 1
    # program = retnet1(BATCH, H, N_CTX, BLOCK_M)
    # mod, params = tl.lower(program)
    # mod = tl.Profiler(mod, params, [1], tl.TensorSupplyType.Integer)
    # mod.assert_allclose(ref_program1, rtol=0.01, atol=0.01)
    # latency = mod.do_bench(mod, n_warmup=10, n_repeat=10, profiler="torch")
    # print("tl: {:.2f} ms".format(latency))
    # print("tl: {:.2f} TFlops".format(total_flops / latency * 1e-9))

    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_K = 64
    program = retnet2(BATCH, H, N_CTX, dim_v, BLOCK_M, BLOCK_N, BLOCK_K)
    mod, params = tl.lower(program)
    mod = tl.Profiler(mod, params, [2], tl.TensorSupplyType.Integer)
    mod.assert_allclose(ref_program2, rtol=0.01, atol=0.01)
    latency = mod.do_bench(mod, n_warmup=10, n_repeat=10, profiler="torch")
    print("tl: {:.2f} ms".format(latency))
    print("tl: {:.2f} TFlops".format(total_flops / latency * 1e-9))