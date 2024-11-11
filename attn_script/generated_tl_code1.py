
import torch
from tvm import tl
import tvm.tl.language as T
import argparse

def ref_program(Q, K, V, scale, casual):
    from flash_attn.flash_attn_interface import flash_attn_func

    return flash_attn_func(Q, K, V, causal=casual, softmax_scale=scale)

def kernel(batch, heads, seq_len, dim, dimv, 
        block_M = None, block_N = None, num_stages = None, thread_num = None):
    scale = (1.0 / dim) ** 0.5 ## * 1.44269504  # log2(e) # 0.69314718  loge(2)
    shape = [batch, seq_len, heads, dim]
    shape_v = [batch, seq_len, heads, dimv]
    dtype = "float16"
    accum_dtype = "float"
    
    # TODO: mask
    is_casual = True


    @T.macro
    def score_mod(
        # scores: T.Buffer([block_M, block_N], accum_dtype),
        scores: T.Buffer([block_M, block_N], accum_dtype), 

        ):
        for i0,i1 in T.Parallel(block_M,block_N):
            scores[i0,i1] = scores[i0,i1] * scale

    
    @T.macro
    def online_func(
        # scores: T.Buffer([block_M, block_N], accum_dtype),
        m: T.Buffer([block_M], accum_dtype), 
        scores: T.Buffer([block_M, block_N], accum_dtype), 
        scores_max_1: T.Buffer([block_M], accum_dtype), 
        m_1: T.Buffer([block_M], accum_dtype), 
        r: T.Buffer([block_M], accum_dtype), 
        scores_2_1_sum_1: T.Buffer([block_M], accum_dtype), 

    ):
        T.reduce_max(scores, scores_max_1,dim=1, clear=True)
        for i0 in T.Parallel(block_M):
            m_1[i0] = T.max(m[i0], scores_max_1[i0])
        for i0 in T.Parallel(block_M):
            m[i0] = m[i0] - m_1[i0]
        for i0 in T.Parallel(block_M):
            m[i0] = T.exp2(m[i0]*1.442695)
        for i0 in T.Parallel(block_M):
            r[i0] = r[i0] * m[i0]
        for i0,i1 in T.Parallel(block_M,block_N):
            scores[i0,i1] = scores[i0,i1] - m_1[i0]
        for i0,i1 in T.Parallel(block_M,block_N):
            scores[i0,i1] = T.exp2(scores[i0,i1]*1.442695)
        T.reduce_sum(scores, scores_2_1_sum_1,dim=1, clear=True)
        for i0 in T.Parallel(block_M):
            r[i0] = r[i0] + scores_2_1_sum_1[i0]


        
    @T.prim_func
    def main(
        Q: T.Buffer(shape, dtype), # type: ignore
        K: T.Buffer(shape, dtype), # type: ignore
        V: T.Buffer(shape_v, dtype), # type: ignore
        

        Output: T.Buffer(shape_v, dtype), # type: ignore
        g_lse: T.Buffer([batch, heads, seq_len], accum_dtype), 

    ):
        with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch, threads=thread_num) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dimv], dtype)
            # acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
            # acc_o = T.alloc_fragment([block_M, dimv], accum_dtype)

            
            m = T.alloc_fragment([block_M], accum_dtype)
            scores = T.alloc_fragment([block_M, block_N], accum_dtype)
            scores_max_1 = T.alloc_fragment([block_M], accum_dtype)
            m_1 = T.alloc_fragment([block_M], accum_dtype)
            r = T.alloc_fragment([block_M], accum_dtype)
            scores_2_1_sum_1 = T.alloc_fragment([block_M], accum_dtype)
            acc_o = T.alloc_fragment([block_M, dimv], accum_dtype)

            

            T.annotate_layout({Q_shared: tl.layout.make_swizzled_layout(Q_shared)})
            T.copy(Q[bz, bx * block_M : (bx + 1) * block_M, by, :], Q_shared)
            T.fill(acc_o, 0)

            T.fill(m, -T.infinity(accum_dtype))
            T.fill(r, 0.0)


            # TODO: mask
            loop_range = (
                T.ceildiv((bx + 1) * block_M, block_N) if is_casual else T.ceildiv(seq_len, block_N)
            )

            for k in T.Pipelined(loop_range, num_stages=num_stages):
                T.copy(K[bz, k * block_N : (k + 1) * block_N, by, :], K_shared)

                # TODO: copy custom_fwd_input_tensor in score_mod&online_func
                # ...

                # TODO: naive solution: if reduce_max, -T.inf; if reduce_sum, 0
                if is_casual and True:
                    for i, j in T.Parallel(block_M, block_N):
                        scores[i, j] = T.if_then_else(
                            bx * block_M + i >= k * block_N + j, 0, -T.infinity(scores.dtype)
                        )
                else:
                    T.clear(scores)
                
                T.gemm(Q_shared, K_shared, scores, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.copy(V[bz, k * block_N : (k + 1) * block_N, by, :], V_shared)
                    
                # call score_mod
                score_mod(scores) # scores
                    
                # call online_func
                online_func(m, scores, scores_max_1, m_1, r, scores_2_1_sum_1) # scores

                for i, j in T.Parallel(block_M, dimv):
                    acc_o[i, j] *= m[i]
                
                # update online_rowscales
                T.copy(m_1, m)


                T.copy(scores, acc_s_cast)
                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
            
            # online_fwd_epilogue
            for i0,i1 in T.Parallel(block_M,dimv):
                acc_o[i0,i1] = acc_o[i0,i1] / r[i0]
            for i0 in T.Parallel(block_M):
                r[i0] = T.log2(r[i0]) * 0.69314718
            for i0 in T.Parallel(block_M):
                r[i0] = r[i0] + m[i0]


            T.copy(acc_o, Output[bz, bx * block_M : (bx + 1) * block_M, by, :])

            # save final_rowscale
            T.copy(r, g_lse[bz, by, bx * block_M : (bx + 1) * block_M])

        
    return main


class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, *custom_fwd_inputs):
        BATCH, N_CTX, H, D_HEAD = q.shape
        D_HEADV = v.shape[-1]
        block_M = 128
        block_N = 128 if D_HEAD <= 128 else 64
        stages = 1
        thread_num = 256
        output_idx_list = [3, 4]
        mod = tl.cached(kernel, output_idx_list, BATCH, H, N_CTX, D_HEAD, D_HEADV, block_M, block_N, stages, thread_num)
        o, *final_scale = mod(q, k, v, *custom_fwd_inputs)
        ctx.save_for_backward(q, k, v, *custom_fwd_inputs, o, *final_scale)
        return o
    
    @staticmethod
    def backward(ctx, grad_o, *bwd_inputs):
        pass

attention = _attention.apply

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=64, help='Batch size')
    parser.add_argument('--h', type=int, default=12, help='Number of heads')
    parser.add_argument('--n_ctx', type=int, default=2048, help='Context size')
    parser.add_argument('--d_head', type=int, default=128, help='Head dimension')
    parser.add_argument('--d_head_v', type=int, default=128, help='Head dimension for V')
    parser.add_argument('--casual', type=bool, default=True, help='Casual flag')
    args = parser.parse_args()
    BATCH, H, N_CTX, D_HEAD, D_HEADV = args.batch, args.h, args.n_ctx, args.d_head, args.d_head_v
    casual = args.casual
    print(f"causal: {casual}")
    flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * D_HEAD
    flops_per_matmulv = 2.0 * BATCH * H * N_CTX * N_CTX * D_HEADV
    total_flops = flops_per_matmul + flops_per_matmulv
    if casual:
        total_flops *= 0.5

    Q = torch.randn(BATCH, N_CTX, H, D_HEAD, dtype=torch.float16, device='cuda')
    K = torch.randn(BATCH, N_CTX, H, D_HEAD, dtype=torch.float16, device='cuda')
    V = torch.randn(BATCH, N_CTX, H, D_HEADV, dtype=torch.float16, device='cuda')
    scale = torch.tensor([1.0 / D_HEAD ** 0.5], dtype=torch.float32, device='cuda')
    o = attention(Q, K, V)

    O_ref = ref_program(Q, K, V, scale, casual)

    close_mask = torch.isclose(o, O_ref, rtol=1e-2, atol=1e-2)
    total_elements = o.numel()
    num_not_close = (~close_mask).sum().item()
    percentage_not_close = (num_not_close / total_elements) * 100
    print(f"{percentage_not_close:.2f}% of the elements are not close.")
    print(f"Total elements: {total_elements}, Not close elements: {num_not_close}")
    # max diff and idx
    max_diff = (o - O_ref).abs().max().item()
    max_diff_idx = (o - O_ref).abs().argmax().item()
    max_diff_idx = torch.unravel_index(torch.tensor(max_diff_idx), o.shape)
    print(f"Max diff: {max_diff} at index {max_diff_idx}")
    print(f"Reference: {O_ref[max_diff_idx]}")
    print(f"Library: {o[max_diff_idx]}")
    print(torch.allclose(o, O_ref, rtol=1e-2, atol=1e-2))

    from tvm.tl.utils import do_bench
    def run():
        o = attention(Q, K, V)
    def run_ref():
        O_ref = ref_program(Q, K, V, scale, casual)
    
    latency = do_bench(run, warmup=200)
    print("tl: {:.2f} ms".format(latency))
    print("tl: {:.2f} TFlops".format(total_flops / latency * 1e-9))
    latency = do_bench(run_ref, warmup=200)
    print("ref: {:.2f} ms".format(latency))
    print("ref: {:.2f} TFlops".format(total_flops / latency * 1e-9))
