import torch
import torch.nn as nn
from tvm import tl
import tvm.tl.language as T

def ref_norm(x, w):
    # return w * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-12))
    input_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + 1e-12)
    return w * x.to(input_dtype)

def rms_norm(M, N, blk_m):
    dtype = "float16"
    accum_dtype = "float"

    @T.prim_func
    def main(A: T.Buffer((M, N), dtype), B: T.Buffer((N), dtype), C: T.Buffer((M, N), dtype)):
        with T.Kernel(T.ceildiv(M, blk_m), threads=128) as bx:
            A_shared = T.alloc_shared((blk_m, N), dtype)
            B_shared = T.alloc_shared((N), dtype)
            A_pow_local = T.alloc_fragment((blk_m, N), accum_dtype)
            A_local = T.alloc_fragment((blk_m, N), accum_dtype)
            A_powsum = T.alloc_fragment((blk_m,), accum_dtype)

            T.copy(A[bx * blk_m : (bx + 1) * blk_m, :], A_shared)
            T.copy(B, B_shared)
            T.copy(A_shared, A_local)
            for i, j in T.Parallel(blk_m, N):
                A_pow_local[i, j] = A_local[i, j] * A_local[i, j]
            T.reduce_sum(A_pow_local, A_powsum, dim=1)
            for i in T.Parallel(blk_m):
                A_powsum[i] = T.rsqrt(A_powsum[i] / N) + 1e-12
            for i, j in T.Parallel(blk_m, N):
                A_local[i, j] *= A_powsum[i]
            for i, j in T.Parallel(blk_m, N):
                A_local[i, j] *= B_shared[j]
            T.copy(A_local, C[bx * blk_m : (bx + 1) * blk_m, :])

    return main

def ref_norm_gated(x, w, g):
    input_dtype = x.dtype
    x = x.to(torch.float32)
    x = x * nn.functional.silu(g.to(torch.float32))
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + 1e-12)
    return w * x.to(input_dtype)

def rms_norm_gated(M, N, blk_m):
    dtype = "float16"
    accum_dtype = "float"
    scale = 1.44269504

    @T.prim_func
    def main(A: T.Buffer((M, N), dtype), 
             B: T.Buffer((N), dtype), 
             G: T.Buffer((M, N), dtype), 
             C: T.Buffer((M, N), dtype)):
        with T.Kernel(T.ceildiv(M, blk_m), threads=128) as bx:
            A_shared = T.alloc_shared((blk_m, N), dtype)
            B_shared = T.alloc_shared((N), dtype)
            G_shared = T.alloc_shared((blk_m, N), dtype)
            A_local = T.alloc_fragment((blk_m, N), accum_dtype)
            A_pow_local = T.alloc_fragment((blk_m, N), accum_dtype)
            A_powsum = T.alloc_fragment((blk_m,), accum_dtype)
            G_local = T.alloc_fragment((blk_m, N), accum_dtype)

            T.copy(A[bx * blk_m : (bx + 1) * blk_m, :], A_shared)
            T.copy(G[bx * blk_m : (bx + 1) * blk_m, :], G_shared)
            T.copy(A_shared, A_local)
            T.copy(G_shared, G_local)
            T.copy(B, B_shared)
            
            for i, j in T.Parallel(blk_m, N):
                G_local[i, j] /= T.exp2(G_local[i, j] * (-scale)) + 1
            for i, j in T.Parallel(blk_m, N):
                A_local[i, j] *= G_local[i, j]
            for i, j in T.Parallel(blk_m, N):
                A_pow_local[i, j] = A_local[i, j] * A_local[i, j]
            T.reduce_sum(A_pow_local, A_powsum, dim=1)
            for i in T.Parallel(blk_m):
                A_powsum[i] = T.rsqrt(A_powsum[i] / N) + 1e-12
            for i, j in T.Parallel(blk_m, N):
                A_local[i, j] *= A_powsum[i]
            for i, j in T.Parallel(blk_m, N):
                A_local[i, j] *= B_shared[j]
            T.copy(A_local, C[bx * blk_m : (bx + 1) * blk_m, :])

    return main

# M, hidden_size, block_M = 64 * 1024, 2048, 1
M, hidden_size, block_M = 64 * 1024, 4096, 1


if __name__ == "__main__":
    # program = rms_norm(M, hidden_size, block_M)
    # mod, params = tl.lower(program)

    # mod = tl.Profiler(mod, params, [2], tl.TensorSupplyType.Normal)
    # mod.assert_allclose(ref_norm, rtol=0.01, atol=0.01)
    # print("All checks pass.")

    # latency = mod.do_bench(ref_norm, warmup=500)
    # print("{:.4f} ms".format(latency))
    # latency = mod.do_bench(mod.func, n_warmup=10, n_repeat=10, profiler="torch")
    # print("{:.4f} ms".format(latency))

    program = rms_norm_gated(M, hidden_size, block_M)
    mod, params = tl.lower(program)

    mod = tl.Profiler(mod, params, [3], tl.TensorSupplyType.Normal)
    mod.assert_allclose(ref_norm_gated, rtol=0.01, atol=0.01)
    print("All checks pass.")

    latency = mod.do_bench(ref_norm_gated, n_warmup=10, n_repeat=10)
    print("{:.4f} ms".format(latency))
    latency = mod.do_bench(mod.func, n_warmup=10, n_repeat=10, profiler="torch")
    print("{:.4f} ms".format(latency))