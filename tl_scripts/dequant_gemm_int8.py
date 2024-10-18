import tvm
from tvm import tl
from tvm.tl.autotuner import *
import itertools
from lop3 import *

def interleave_weight(qweight, nbits=4, target_dtype="float16"):
    import numpy as np
    assert target_dtype in ["float16", "int8"]
    # reinterpret the data type of qweight to int32
    qweight = qweight.view(np.int32)
    new_qweight = np.zeros_like(qweight)
    bits_stride = 8 if target_dtype == "int8" else 16
    mask = (1 << nbits) - 1  # for 4bit the val is 0x0000000f
    num_groups = 32 // bits_stride
    elems_per_group = bits_stride // nbits
    for i in range(num_groups):
        for j in range(elems_per_group):
            offset = i * elems_per_group + j
            shift = (offset % num_groups) * bits_stride + (offset // num_groups) * nbits
            new_qweight |= ((qweight >> (nbits * offset)) & mask) << shift

    if nbits == 1 and target_dtype == "int8":
        # special handling for 1b interleave
        n16_weight = new_qweight & np.int32(np.uint32(0xF0F00F0F))
        n16_weight |= ((new_qweight & np.int32(np.uint32(0x000000F0))) >> 4) << 16
        n16_weight |= ((new_qweight & np.int32(np.uint32(0x0000F000))) >> 12) << 24
        n16_weight |= ((new_qweight & np.int32(np.uint32(0x000F0000))) >> 16) << 4
        n16_weight |= ((new_qweight & np.int32(np.uint32(0x0F000000))) >> 24) << 12
        return n16_weight.view(np.int8)
    elif nbits == 2 and target_dtype == "float16":
        n8_weight = new_qweight & np.int32(np.uint32(0xFF0000FF))
        n8_weight |= ((new_qweight & np.int32(np.uint32(0x0000FF00))) >> 8) << 16
        n8_weight |= ((new_qweight & np.int32(np.uint32(0x00FF0000))) >> 16) << 8
        return n8_weight.view(np.int8)
    elif nbits == 1 and target_dtype == "float16":
        n8_weight = new_qweight & np.int32(np.uint32(0xF000000F))
        n8_weight |= ((new_qweight & np.int32(np.uint32(0x000000F0))) >> 4) << 8
        n8_weight |= ((new_qweight & np.int32(np.uint32(0x00000F00))) >> 8) << 16
        n8_weight |= ((new_qweight & np.int32(np.uint32(0x0000F000))) >> 12) << 24
        n8_weight |= ((new_qweight & np.int32(np.uint32(0x000F0000))) >> 16) << 4
        n8_weight |= ((new_qweight & np.int32(np.uint32(0x00F00000))) >> 20) << 12
        n8_weight |= ((new_qweight & np.int32(np.uint32(0x0F000000))) >> 24) << 20

    return new_qweight.view(np.int8)

def _tir_packed_to_unsigned_convert(storage_type="uint", storage_nbit=8):
    storage_dtype = storage_type + str(storage_nbit)

    def f_convert(nbit: int, val: tvm.tir.PrimExpr, pos: tvm.tir.PrimExpr, dtype: str):
        assert val.dtype == storage_dtype, f"{val.dtype} != {storage_dtype}"
        mask = tvm.tir.const((1 << nbit) - 1, storage_dtype)
        return ((val >> (pos * nbit).astype(storage_dtype)) & mask).astype(dtype)

    return f_convert

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
#     threads,
#     num_bits=4,
# ):
#     num_elems_per_byte = 8 // num_bits
#     storage_dtype = "int8"
#     A_shape = (M, K)
#     B_shape = (N, K // num_elems_per_byte)
#     A_shared_shape = (block_M, block_K)
#     B_shared_shape = (block_N, block_K // num_elems_per_byte)
#     B_dequantize_shared_shape = (block_N, block_K)

#     import tvm.tl.language as T

#     @T.prim_func
#     def main(
#             A: T.Buffer(A_shape, in_dtype),
#             B: T.Buffer(B_shape, storage_dtype),
#             Ct: T.Buffer((N, M), out_dtype),
#     ):
#         with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
#             A_shared = T.alloc_shared(A_shared_shape, in_dtype)
#             B_shared = T.alloc_shared(B_shared_shape, storage_dtype)
#             B_local = T.alloc_fragment(B_shared_shape, storage_dtype)
#             B_dequantize_local = T.alloc_fragment(B_dequantize_shared_shape, in_dtype)
#             B_dequantize_prev_local = T.alloc_fragment(B_dequantize_shared_shape, in_dtype)
#             Ct_local = T.alloc_fragment((block_N, block_M), accum_dtype)

#             T.clear(Ct_local)
#             for k in T.Pipelined(
#                 T.ceildiv(K, block_K), 
#                 num_stages=num_stages,
#                 order=[-1,-1,0,1],
#                 stage=[-1,-1,0,0],
#                 group=[[0],[1],[2,3,4],[5]]
#             ):
#                 T.copy(A[by * block_M, k * block_K], A_shared)
#                 T.copy(B[bx * block_N, k * block_K // num_elems_per_byte], B_shared)
#                 T.copy(B_shared, B_local)
#                 for i, j in T.Parallel(block_N, block_K):
#                     B_dequantize_local[i, j] = _tir_packed_to_unsigned_convert("int", 8)(
#                         num_bits,
#                         B_local[i, j // 2],
#                         j % 2,
#                         dtype=in_dtype,
#                     )
#                 T.copy(B_dequantize_local, B_dequantize_prev_local)
#                 T.gemm(B_dequantize_prev_local, A_shared, Ct_local, transpose_B=True)
#             T.copy(Ct_local, Ct[bx * block_N, by * block_M])

#     return main

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
    threads,
    num_bits=4,
):
    num_elems_per_byte = 8 // num_bits
    storage_dtype = "int8"
    A_shape = (M, K)
    B_shape = (N, K // num_elems_per_byte)
    A_shared_shape = (block_M, block_K)
    B_shared_shape = (block_N, block_K // num_elems_per_byte)
    B_dequantize_shared_shape = (block_N, block_K)

    import tvm.tl.language as T


    lop3_intrin_info = get_lop3_intrin_group(
        out_dtype="int8",
        storage_dtype="int8",
        source_format="uint",
        source_bit=4,
    )
    import_source = lop3_intrin_info["c_source"]
    func_name = lop3_intrin_info["func_name"]

    @T.prim_func
    def main(
            A: T.Buffer(A_shape, in_dtype),
            B: T.Buffer(B_shape, storage_dtype),
            Ct: T.Buffer((N, M), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, storage_dtype)
            B_local = T.alloc_fragment(B_shared_shape, storage_dtype)
            B_dequantize_local = T.alloc_fragment(B_dequantize_shared_shape, in_dtype)
            B_dequantize_prev_local = T.alloc_fragment(B_dequantize_shared_shape, in_dtype)
            Ct_local = T.alloc_fragment((block_N, block_M), accum_dtype)

            T.import_source(import_source)

            T.clear(Ct_local)
            for k in T.Pipelined(
                T.ceildiv(K, block_K), 
                num_stages=num_stages,
                order=[-1,-1,0,1],
                stage=[-1,-1,0,0],
                group=[[0],[1],[2,3,4],[5]]
            ):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K // num_elems_per_byte], B_shared)
                T.copy(B_shared, B_local)
                for i in T.serial(4):
                    T.call_extern(
                        func_name,
                        T.address_of(B_local[0, 8 * i]),
                        T.address_of(B_dequantize_local[0,16 * i]),
                        dtype=in_dtype
                    )
                T.copy(B_dequantize_local, B_dequantize_prev_local)
                T.gemm(B_dequantize_prev_local, A_shared, Ct_local, transpose_B=True)
            T.copy(Ct_local, Ct[bx * block_N, by * block_M])

    return main

def matmul_(
    M,
    N,
    K,
    in_dtype,
    out_dtype,
    accum_dtype,
    num_bits=4,
):
    
    def get_configs():
        # block_M = [64, 128, 256]
        # block_N = [64, 128, 256]
        # block_K = [64, 128]
        # num_stages = [1, 2, 3, 4]
        # thread_num = [128, 256, 512]
        block_M = [256]
        block_N = [128]
        block_K = [128]
        num_stages = [3]
        thread_num = [256]
        _configs = list(itertools.product(block_M, block_N, block_K, num_stages, thread_num))

        configs = [
            {'block_M': c[0], 'block_N': c[1], 'block_K': c[2], 'num_stages': c[3], 'thread_num': c[4]}
            for c in _configs
            if c[4] < c[0] * 2
        ]
        return configs

    @autotune(configs=get_configs(), keys=['block_M', 'block_N', 'block_K', 'num_stages', 'thread_num'], warmup=10, rep=5)
    @jit(out_idx=[2], supply_type=tl.TensorSupplyType.Integer, ref_prog=None)
    def kernel(block_M=None, block_N=None, block_K=None, num_stages=None, thread_num=None):
        num_elems_per_byte = 8 // num_bits
        storage_dtype = "int8"
        A_shape = (M, K)
        B_shape = (N, K // num_elems_per_byte)
        A_shared_shape = (block_M, block_K)
        B_shared_shape = (block_N, block_K // num_elems_per_byte)
        B_dequantize_shared_shape = (block_N, block_K)

        import tvm.tl.language as T

        @T.prim_func
        def main(
                A: T.Buffer(A_shape, in_dtype),
                B: T.Buffer(B_shape, storage_dtype),
                Ct: T.Buffer((N, M), out_dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=thread_num) as (bx, by):
                A_shared = T.alloc_shared(A_shared_shape, in_dtype)
                B_shared = T.alloc_shared(B_shared_shape, storage_dtype)
                B_local = T.alloc_fragment(B_shared_shape, storage_dtype)
                B_dequantize_local = T.alloc_fragment(B_dequantize_shared_shape, in_dtype)
                B_dequantize_prev_local = T.alloc_fragment(B_dequantize_shared_shape, in_dtype)
                Ct_local = T.alloc_fragment((block_N, block_M), accum_dtype)

                T.clear(Ct_local)
                for k in T.Pipelined(
                    T.ceildiv(K, block_K), 
                    num_stages=num_stages,
                    order=[-1,-1,0,1],
                    stage=[-1,-1,0,0],
                    group=[[0],[1],[2,3,4],[5]]
                ):
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    T.copy(B[bx * block_N, k * block_K // num_elems_per_byte], B_shared)
                    T.copy(B_shared, B_local)
                    for i, j in T.Parallel(block_N, block_K):
                        B_dequantize_local[i, j] = _tir_packed_to_unsigned_convert("int", 8)(
                            num_bits,
                            B_local[i, j // 2],
                            j % 2,
                            dtype=in_dtype,
                        )
                    T.copy(B_dequantize_local, B_dequantize_prev_local)
                    T.gemm(B_dequantize_prev_local, A_shared, Ct_local, transpose_B=True)
                T.copy(Ct_local, Ct[bx * block_N, by * block_M])

        return main
    return kernel()


def run_gemm(
    M,
    N,
    K,
    dtypeAB,
    dtypeC,
    dtypeAccum,
    block_M,
    block_N,
    block_K,
    num_stages=3,
    num_threads=128,
):
    program = matmul(
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        dtypeAB,
        dtypeC,
        dtypeAccum,
        num_stages,
        num_threads,
    )
    print(program)

    mod, params = tl.lower(program)
    mod = tl.Profiler(mod, params, [2], tl.TensorSupplyType.Integer)

    # out = mod.run_once()

    # print(f"output is {out}")

    def ref_program(A, qB):
        import torch

        B = (
            torch.zeros(qB.shape[0], qB.shape[1] * 8 // 4,
                        dtype=torch.half).to(torch.half).to(A.device))
        for i in range(B.shape[0]):
            for j in range(B.shape[1]):
                B[i][j] = ((qB[i][j // 2] >> (4 * (j % 2))) & 0xF).to(torch.half)
        torch.set_printoptions(edgeitems=16)
        print("B_ref:", B, B.shape)
        C = torch.matmul(A.to(torch.float), B.T.to(torch.float))
        C = C.to(torch.__getattribute__(dtypeC))
        # Caution: transpose the output
        return C.transpose(0, 1)

    import torch
    # # A = torch.ones((M, K), device="cuda", dtype=torch.int8)
    # # B = torch.ones((N, K // 2), device="cuda", dtype=torch.int8)
    # A = torch.randint(0, 4, (M, K), device="cuda", dtype=torch.int8)
    # B = torch.randint(0, 4, (N, K // 2), device="cuda", dtype=torch.int8)
    # # for x in range(M):
    # #     for y in range(K):
    # #         A[x][y] = (x * K + y) % 100
    # # for x in range(N):
    # #     for y in range(K // 2):
    # #         B[x][y] = (x * 64 + y) % 100
    # B_numpy = B.cpu().numpy()
    # B_interleaved = torch.from_numpy(interleave_weight(B_numpy, 4, target_dtype="int8")).to("cuda")
    # ref_out = ref_program(A, B)
    # mod_out = mod.func(A, B_interleaved)
    # assert torch.allclose(mod_out, ref_out, rtol=0.1, atol=0.1), (mod_out, ref_out)
    # print("Pass")

    # mod.assert_allclose(ref_program)
    total_flops = 2 * M * N * K
    # latency = mod.do_bench(ref_program, n_warmup=10, n_repeat=10)
    # print("torch: {:.2f} ms".format(latency))
    # print("torch: {:.2f} TFlops".format(total_flops / latency * 1e-9))
    latency = mod.do_bench(mod.func, n_warmup=10, n_repeat=10, profiler="tvm")
    print("tl: {:.2f} ms".format(latency))
    print("tl: {:.2f} TFlops".format(total_flops / latency * 1e-9))

def tune_gemm(
    M,
    N,
    K,
    dtypeAB,
    dtypeC,
    dtypeAccum
):
    best_latency, best_config, ref_latency = matmul(
        M,
        N,
        K,
        dtypeAB,
        dtypeC,
        dtypeAccum
    )

    total_flops = 2 * M * N * K
    print(f"Best latency: {best_latency}")
    print(f"Best TFlops: {total_flops / best_latency * 1e-9}")
    print(f"Best config: {best_config}")
    # print(f"Ref TFlops: {total_flops / ref_latency * 1e-9}")


def test_run_dequantize_gemm():
    # run_gemm(256, 256, 256, "int8", "int32", "int32", 256, 128, 128, num_stages=3, num_threads=256)
    # run_gemm(512, 512, 512, "int8", "int32", "int32", 256, 128, 128, num_stages=3, num_threads=256)
    # run_gemm(4096, 4096, 4096, "int8", "int32", "int32", 256, 128, 128, num_stages=3, num_threads=256)
    # run_gemm(8192, 8192, 8192, "int8", "int32", "int32", 256, 128, 128, num_stages=3, num_threads=256)
    # run_gemm(256, 4096, 4096, "int8", "int32", "int32", 256, 128, 128, num_stages=3, num_threads=256)
    run_gemm(256, 8192, 8192, "int8", "int32", "int32", 256, 128, 128, num_stages=3, num_threads=256)
    # run_gemm(256, 16384, 16384, "int8", "int32", "int32", 256, 128, 128, num_stages=3, num_threads=256)
    # run_gemm(16384, 16384, 16384, "int8", "int32", "int32", 256, 128, 128, num_stages=3, num_threads=256)
    # run_gemm(256, 256, 256, "float16", "float16", "float32", 128, 128, 64, num_threads=128)
    # tune_gemm(8192, 8192, 8192, "int8", "int32", "int32")


if __name__ == "__main__":
    # bitblas.testing.main()
    test_run_dequantize_gemm()
