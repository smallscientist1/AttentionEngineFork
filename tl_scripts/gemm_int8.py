import torch
import tvm
from tvm import tl

swizzle_M = 4096
swizzle_N = 4096

def matmul(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    in_dtype,
    out_dtype,
    accum_dtype,
    num_stages,
    threads
):
    A_shape = (M, K)
    B_shape = (N, K)
    A_shared_shape = (block_M, block_K)
    B_shared_shape = (block_N, block_K)

    import tvm.tl.language as T

    @T.prim_func
    def main(
            A: T.Buffer(A_shape, in_dtype),
            B: T.Buffer(B_shape, in_dtype),
            C: T.Buffer((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype)
            A_local = T.alloc_fragment(A_shared_shape, in_dtype)
            A_new_shared = T.alloc_shared(A_shared_shape, in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.annotate_layout({A_new_shared: tl.layout.make_swizzled_layout(A_new_shared)})
            
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K], B_shared)
                T.copy(A_shared, A_local)
                T.copy(A_local, A_new_shared)
                T.gemm(A_new_shared, B_shared, C_local, transpose_B=True)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main

# def matmul(
#     M,
#     N,
#     K,
#     block_M,
#     block_N,
#     block_K,
#     in_dtype,
#     out_dtype,
#     accum_dtype,
#     num_stages,
#     threads
# ):
#     A_shape = (M, K)
#     B_shape = (N, K)
#     A_shared_shape = (block_M, block_K)
#     B_shared_shape = (block_N, block_K)

#     import tvm.tl.language as T

#     @T.prim_func
#     def main(
#             A: T.Buffer(A_shape, in_dtype),
#             B: T.Buffer(B_shape, in_dtype),
#             C: T.Buffer((M, N), out_dtype),
#     ):
#         with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
#             A_shared = T.alloc_shared(A_shared_shape, in_dtype)
#             A_local = T.alloc_fragment(A_shared_shape, in_dtype)
#             B_shared = T.alloc_shared(B_shared_shape, in_dtype)
#             C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

#             T.clear(C_local)
#             for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
#                 T.copy(A[by * block_M, k * block_K], A_shared)
#                 T.copy(A_shared, A_local)
#                 T.copy(B[bx * block_N, k * block_K], B_shared)
#                 T.gemm(A_local, B_shared, C_local, transpose_B=True)
#             T.copy(C_local, C[by * block_M, bx * block_N])

#     return main

# def matmul(
#     M,
#     N,
#     K,
#     block_M,
#     block_N,
#     block_K,
#     in_dtype,
#     out_dtype,
#     accum_dtype,
#     num_stages,
#     threads
# ):
#     A_shape = (M, K)
#     B_shape = (N, K)
#     A_shared_shape = (block_M, block_K)
#     B_shared_shape = (block_N, block_K)

#     import tvm.tl.language as T

#     @T.prim_func
#     def main(
#             A: T.Buffer(A_shape, in_dtype),
#             B: T.Buffer(B_shape, in_dtype),
#             Ct: T.Buffer((N, M), out_dtype),
#     ):
#         with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
#             A_shared = T.alloc_shared(A_shared_shape, in_dtype)
#             B_shared = T.alloc_shared(B_shared_shape, in_dtype)
#             B_local = T.alloc_fragment(B_shared_shape, in_dtype)
#             # B @ At = Ct
#             Ct_local = T.alloc_fragment((block_N, block_M), accum_dtype)

#             T.clear(Ct_local)
#             for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
#                 T.copy(A[by * block_M, k * block_K], A_shared)
#                 T.copy(B[bx * block_N, k * block_K], B_shared)
#                 T.copy(B_shared, B_local)
#                 T.gemm(B_local, A_shared, Ct_local, transpose_B=True)
#             T.copy(Ct_local, Ct[bx * block_N, by * block_M])

#     return main

def ref_program(A, B):
    return (A.to(float) @ B.to(float).transpose(0,1)).to(torch.int)
    # return (A.to(float) @ B.to(float).transpose(0,1)).to(torch.int).transpose(0,1)
    # return (A.to(float) @ B.to(float).transpose(0,1)).to(torch.float16)

if __name__ == "__main__":
    # M, N, K = 8192, 8192, 8192
    # block_M, block_N, block_K = 128, 256, 64
    # num_stages, threads = 4, 256
    M, N, K = 256, 256, 256
    block_M, block_N, block_K = 128, 256, 128
    num_stages, threads = 2, 256
    in_dtype, out_dtype, accum_dtype = "int8", "int32", "int32"
    # in_dtype, out_dtype, accum_dtype = "float16", "float16", "float"

    program = matmul(
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        in_dtype,
        out_dtype,
        accum_dtype,
        num_stages,
        threads
    )

    mod, params = tl.lower(program)

    mod = tl.Profiler(mod, params, [2], tl.TensorSupplyType.Integer)

    out = mod.run_once()
    print(f"output is {out}")
    mod.assert_allclose(ref_program)

    total_flops = 2 * M * N * K
    latency = mod.do_bench(ref_program, n_warmup=10, n_repeat=10)
    print("torch: {:.2f} ms".format(latency))
    print("torch: {:.2f} TFlops".format(total_flops / latency * 1e-9))
    latency = mod.do_bench(mod.func, n_warmup=10, n_repeat=10)
    print("tl: {:.2f} ms".format(latency))
    print("tl: {:.2f} TFlops".format(total_flops / latency * 1e-9))