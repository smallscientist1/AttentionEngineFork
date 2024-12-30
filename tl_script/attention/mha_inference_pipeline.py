import torch
import torch.nn.functional as F
from tvm import tl
import tvm.tl.language as T
from functools import partial

num_split = 4
def flashattn(batch, heads, seqlen_q, seqlen_kv, dim, dimv, is_casual, 
              block_M, block_N, num_stages = 2, thread_num = 256, shared_fuse = False):
    scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    shape_q = [batch, seqlen_q, heads, dim]
    shape_kv = [batch, seqlen_kv, heads, dim]
    part_shape = [batch, seqlen_q, heads, num_split, dim]
    dtype = "float16"
    accum_dtype = "float"
    
    shape = shape_q
    shape_k = [batch, seqlen_kv, heads, dim]
    shape_v = [batch, seqlen_kv, heads, dimv]
    part_shape_v = [batch, seqlen_kv, heads, num_split, dimv]
    assert(casual == False)

    @T.macro
    def MMA0(
        K: T.Buffer(shape_kv, dtype),
        Q_shared: T.Buffer([block_M, dim], dtype),
        K_shared: T.Buffer([block_N, dim], dtype),
        acc_s: T.Buffer([block_M, block_N], accum_dtype),
        k: T.int32,
        mid: T.int32,
        hid: T.int32,
        bid: T.int32,
        sid: T.int32,
    ):
        T.copy(K[bid, (seqlen_kv // num_split) * sid + k * block_N : (seqlen_kv // num_split) * sid + (k + 1) * block_N, hid, :], K_shared)
        # TODO: Handle casual split case
        if is_casual:
            for i, j in T.Parallel(block_M, block_N):
                acc_s[i, j] = T.if_then_else(
                    mid * block_M + i >= k * block_N + j, 0, -T.infinity(acc_s.dtype)
                )
        else:
            T.clear(acc_s)
        T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

    @T.macro
    def MMA1(
        V: T.Buffer(shape_kv, dtype),
        V_shared: T.Buffer([block_M, dim], dtype),
        acc_s_cast: T.Buffer([block_M, block_N], dtype),
        acc_o: T.Buffer([block_M, dim], accum_dtype),
        k: T.int32,
        hid: T.int32,
        bid: T.int32,
        sid: T.int32,
    ):
        T.copy(V[bid, (seqlen_kv // num_split) * sid + k * block_N : (seqlen_kv // num_split) * sid + (k + 1) * block_N, hid, :], V_shared)
        T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

    @T.macro
    def Softmax(
        acc_s: T.Buffer([block_M, block_N], accum_dtype),
        acc_s_cast: T.Buffer([block_M, block_N], dtype),
        scores_max: T.Buffer([block_M], accum_dtype),
        scores_max_prev: T.Buffer([block_M], accum_dtype),
        scores_scale: T.Buffer([block_M], accum_dtype),
        scores_sum: T.Buffer([block_M], accum_dtype),
        logsum: T.Buffer([block_M], accum_dtype),
    ):
        T.copy(scores_max, scores_max_prev)
        T.fill(scores_max, -T.infinity(accum_dtype))
        T.reduce_max(acc_s, scores_max, dim=1, clear=False)
        # To do causal softmax, we need to set the scores_max to 0 if it is -inf
        # This process is called Check_inf in FlashAttention3 code, and it only need to be done
        # in the first ceil_div(kBlockM, kBlockN) steps.
        # for i in T.Parallel(block_M):
        #     scores_max[i] = T.if_then_else(scores_max[i] == -T.infinity(accum_dtype), 0, scores_max[i])
        for i in T.Parallel(block_M):
            scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
        for i, j in T.Parallel(block_M, block_N):
            # Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
            # max * log_2(e)) This allows the compiler to use the ffma
            # instruction instead of fadd and fmul separately.
            acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
        T.reduce_sum(acc_s, scores_sum, dim=1)
        for i in T.Parallel(block_M):
            logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
        T.copy(acc_s, acc_s_cast)

    @T.macro
    def Rescale(
        acc_o: T.Buffer([block_M, dim], accum_dtype),
        scores_scale: T.Buffer([block_M], accum_dtype),
    ):
        for i, j in T.Parallel(block_M, dim):
            acc_o[i, j] *= scores_scale[i]

    @T.macro
    def flash_attn_split(
        Q: T.Buffer(shape_q, dtype),
        K: T.Buffer(shape_kv, dtype),
        V: T.Buffer(shape_kv, dtype),
        glse: T.Buffer([batch, heads, num_split, seqlen_q], dtype),
        Output_partial: T.Buffer(part_shape, dtype),
    ):
        with T.Kernel(T.ceildiv(seqlen_q, block_M), heads * batch, num_split, threads=128 * 2) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dim], dtype)
            O_shared = T.alloc_shared([block_M, dim], dtype)
            acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
            acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
            scores_max = T.alloc_fragment([block_M], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
            scores_scale = T.alloc_fragment([block_M], accum_dtype)
            scores_sum = T.alloc_fragment([block_M], accum_dtype)
            logsum = T.alloc_fragment([block_M], accum_dtype)

            mid = bx
            hid = by % heads
            bid = by // heads
            sid = bz

            T.annotate_layout({Q_shared: tl.layout.make_swizzled_layout(Q_shared)})
            T.copy(Q[bid, mid * block_M : (mid + 1) * block_M, hid, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            # TODO: Handle casual split case
            loop_range = (
                T.min(T.ceildiv(seqlen_kv, block_N), T.ceildiv((mid + 1) * block_M, block_N)) 
                if is_casual else T.ceildiv((seqlen_kv // num_split), block_N)
            )

            for k in T.Pipelined(loop_range, num_stages=2, 
                                 # order=[-1,0,3,1,-1,2], stage=[-1,0,0,1,-1,1], sync=[[0,13],[1,9]], group=[[0], [1,2], [3,4,5,6,7,8,9,10], [11], [12], [13]]
                                 ):
            # for k in T.Pipelined(loop_range, num_stages=2):
            # for k in T.serial(loop_range): # for ablation
                MMA0(K, Q_shared, K_shared, acc_s, k, mid, hid, bid, sid)
                Softmax(acc_s, acc_s_cast, scores_max, scores_max_prev, scores_scale, scores_sum, logsum)
                Rescale(acc_o, scores_scale)
                MMA1(V, V_shared, acc_s_cast, acc_o, k, hid, bid, sid)
            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] /= logsum[i]
            for i in T.Parallel(block_M):
                logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
            T.copy(logsum, glse[bid, hid, sid, mid * block_M : (mid + 1) * block_M])
            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output_partial[bid, mid * block_M : (mid + 1) * block_M, hid, sid, :])

    @T.macro
    def score_mod(
        # scores: T.Buffer([block_M, block_N], accum_dtype),
        scores: T.Buffer([block_M, block_N], accum_dtype), 

        ):
        for i0,i1 in T.Parallel(block_M,block_N):
            scores[i0,i1] = scores[i0,i1] * float(0.07216878364870323)

        pass
    
    @T.macro
    def online_func(
        # scores: T.Buffer([block_M, block_N], accum_dtype),
        m: T.Buffer([block_M], accum_dtype), 
        scores: T.Buffer([block_M, block_N], accum_dtype), 
        scores_max_0: T.Buffer([block_M], accum_dtype), 
        r: T.Buffer([block_M], accum_dtype), 
        scores_1_0_sum_0: T.Buffer([block_M], accum_dtype), 

    ):
        T.reduce_max(scores, scores_max_0,dim=1, clear=True)
        for i0 in T.Parallel(block_M):
            scores_max_0[i0] = T.max(m[i0], scores_max_0[i0])
        for i0 in T.Parallel(block_M):
            m[i0] = m[i0] - scores_max_0[i0]
        for i0 in T.Parallel(block_M):
            m[i0] = T.exp2(m[i0]*1.442695)
        for i0 in T.Parallel(block_M):
            r[i0] = r[i0] * m[i0]
        for i0,i1 in T.Parallel(block_M,block_N):
            scores[i0,i1] = scores[i0,i1] - scores_max_0[i0]
        for i0,i1 in T.Parallel(block_M,block_N):
            scores[i0,i1] = T.exp2(scores[i0,i1]*1.442695)
        T.reduce_sum(scores, scores_1_0_sum_0,dim=1)
        for i0 in T.Parallel(block_M):
            r[i0] = r[i0] + scores_1_0_sum_0[i0]

        pass

     
    
    @T.macro
    def main_split(
        Q: T.Buffer(shape, dtype), # type: ignore
        K: T.Buffer(shape_k, dtype), # type: ignore
        V: T.Buffer(shape_v, dtype), # type: ignore
        

        Output_partial: T.Buffer(part_shape_v, dtype), # type: ignore
        g_lse: T.Buffer([batch, heads, num_split, seqlen_q], accum_dtype), 

    ):
        with T.Kernel(T.ceildiv(seqlen_q, block_M), heads*batch, num_split, threads=thread_num) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dimv], dtype)
            O_shared = T.alloc_shared([block_M, dimv], dtype)
            # V_local = T.alloc_fragment([block_N, dimv], dtype)
            scores = T.alloc_fragment([block_M, block_N], accum_dtype)
            scores_shared = T.alloc_shared([block_M, block_N], accum_dtype)
            scores_1 = T.alloc_fragment([block_M, block_N], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
            acc_s_cast_1 = T.alloc_fragment([block_M, block_N], dtype)
            # acc_o = T.alloc_fragment([block_M, dimv], accum_dtype)

            
            m = T.alloc_fragment([block_M], accum_dtype)
            scores_max_0 = T.alloc_fragment([block_M], accum_dtype)
            r = T.alloc_fragment([block_M], accum_dtype)
            scores_1_0_sum_0 = T.alloc_fragment([block_M], accum_dtype)
            acc_o = T.alloc_fragment([block_M, dimv], accum_dtype)
            
            mid = bx
            hid = by % heads
            bid = by // heads
            sid = bz

            

            T.annotate_layout({
                Q_shared: tl.layout.make_swizzled_layout(Q_shared),
                scores_shared: tl.layout.make_swizzled_layout(scores_shared),
                
            })
            T.copy(Q[bid, mid * block_M : (mid + 1) * block_M, hid, :], Q_shared)
            
            T.fill(acc_o, 0)
            T.fill(m, 1.0)

            T.fill(m, -T.infinity(accum_dtype))
            T.fill(r, 0.0)


            # TODO: mask
            loop_range = (
                T.ceildiv(seqlen_kv // num_split, block_N)
            )

            for k in T.Pipelined(loop_range, num_stages=num_stages):
                T.copy(K[bid, (seqlen_kv // num_split) * sid + k * block_N : (seqlen_kv // num_split) * sid + (k + 1) * block_N, hid, :], K_shared)

                # TODO: copy custom_fwd_input_tensor in score_mod&online_func
                

                # TODO: naive solution: if reduce_max, -T.inf; if reduce_sum, 0
                if is_casual and True:
                    for i, j in T.Parallel(block_M, block_N):
                        scores[i, j] = T.if_then_else(
                            bx * block_M + i >= k * block_N + j, 0, -T.infinity(scores.dtype)
                        )
                else:
                    T.clear(scores)
                
                T.gemm(Q_shared, K_shared, scores, transpose_B=True, policy= (T.GemmWarpPolicy.FullRow if (not shared_fuse) else T.GemmWarpPolicy.FullCol))
                T.copy(V[bid, (seqlen_kv // num_split) * sid + k * block_N : (seqlen_kv // num_split) * sid + (k + 1) * block_N, hid, :], V_shared)
                    
                
                # call score_mod
                score_mod(scores) # scores
                    
                # call online_func
                if shared_fuse:
                    T.copy(scores, scores_shared)
                    T.copy(scores_shared, scores_1)
                    online_func(m, scores, scores_max_0, r, scores_1_0_sum_0)
                    T.copy(scores_1, acc_s_cast_1)

                else:
                    online_func(m, scores, scores_max_0, r, scores_1_0_sum_0) # scores
                    T.copy(scores, acc_s_cast)

                # for i, j in T.Parallel(block_M, dimv):
                #     acc_o[i, j] *= o_scale[i]
                for i, j in T.Parallel(block_M, dimv):
                    acc_o[i, j] *= m[i]

                
                # update online_rowscales
                T.copy(scores_max_0, m)


                if shared_fuse:
                    T.gemm(acc_s_cast_1, V_shared, acc_o, policy=(T.GemmWarpPolicy.FullCol))
                else:
                    T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
            
            # online_fwd_epilogue
            for i0,i1 in T.Parallel(block_M,dimv):
                acc_o[i0,i1] = acc_o[i0,i1] / r[i0]
            for i0 in T.Parallel(block_M):
                r[i0] = T.log2(r[i0]) * 0.69314718
            for i0 in T.Parallel(block_M):
                r[i0] = r[i0] + m[i0]


            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output_partial[bid, mid * block_M : (mid + 1) * block_M, hid, sid, :])

            # save final_rowscale
            T.copy(r, g_lse[bid, hid, sid, mid * block_M : (mid + 1) * block_M])


    @T.macro
    def combine(
        glse: T.Buffer([batch, heads, num_split, seqlen_q], dtype),
        Output_partial: T.Buffer(part_shape, dtype),
        Output: T.Buffer(shape_q, dtype),
    ):
        with T.Kernel(T.ceildiv(seqlen_q, block_M), heads, batch, threads=128) as (bx, by, bz):
            po_local = T.alloc_fragment([block_M, dim], dtype)
            po_shared = T.alloc_shared([block_M, dim], dtype)
            o_accum_local = T.alloc_fragment([block_M, dim], accum_dtype)
            o_shared = T.alloc_shared([block_M, dim], dtype)
            lse_local = T.alloc_fragment([num_split, block_M], dtype)
            # lse_local2 = T.alloc_fragment([num_split, block_M], dtype)
            # lse_shared = T.alloc_shared([num_split, block_M], dtype)
            # lse_split_shared = T.alloc_shared([block_M], dtype)
            lse_local_split = T.alloc_fragment([block_M], accum_dtype)
            lse_logsum_local = T.alloc_fragment([block_M], accum_dtype)
            lse_max_local = T.alloc_fragment([block_M], accum_dtype)
            scale_local = T.alloc_fragment([block_M], accum_dtype)
            
            T.annotate_layout(
                {
                    o_accum_local: T.Fragment(o_accum_local.shape, lambda i, j: i),
                    lse_local_split: T.Fragment(lse_local_split.shape, lambda i: i),
                    # logsum_accum_local: T.Fragment(logsum_accum_local.shape, lambda i: i),
                    o_shared: tl.layout.make_swizzled_layout(o_shared),
                    po_shared: tl.layout.make_swizzled_layout(po_shared),
                }
            )

            T.clear(lse_logsum_local)
            T.clear(o_accum_local)
            T.copy(glse[bz, by, :, bx * block_M : (bx + 1) * block_M,], lse_local)
            # T.copy(glse[bz, by, :, bx * block_M : (bx + 1) * block_M,], lse_shared)
            T.reduce_max(lse_local, lse_max_local, dim=0, clear=False)
            for k in T.Pipelined(num_split):
                # CAUTION: This is a hack implementation to avoid the compilation bug, need to be fixed
                # The correct implementation should be: T.copy(lse_local[k, :], lse_local_split)
                T.copy(lse_local[0, :], lse_local_split)
                for i in T.Parallel(block_M):
                    lse_logsum_local[i] += T.exp2(lse_local_split[i] - lse_max_local[i])
            for i in T.Parallel(block_M):
                lse_logsum_local[i] = T.log2(lse_logsum_local[i]) + lse_max_local[i]
            for k in T.Pipelined(num_split, num_stages=2):
            # for k in T.serial(num_split): # for ablation
                T.copy(Output_partial[bz, bx * block_M : (bx + 1) * block_M, by, k, :], po_shared)
                T.copy(po_shared, po_local)
                T.copy(lse_local[k, :], lse_local_split)
                for i in T.Parallel(block_M):
                    scale_local[i] = T.exp2(lse_local_split[i] - lse_logsum_local[i])
                for i, j in T.Parallel(block_M, dim):
                    o_accum_local[i, j] += po_local[i, j] * scale_local[i]
            T.copy(o_accum_local, o_shared)
            T.copy(o_shared, Output[bz, bx * block_M : (bx + 1) * block_M, by, :])

    @T.prim_func
    def main(
        Q: T.Buffer(shape_q, dtype),
        K: T.Buffer(shape_kv, dtype),
        V: T.Buffer(shape_kv, dtype),
        glse: T.Buffer([batch, heads, num_split, seqlen_q], dtype),
        Output_partial: T.Buffer(part_shape, dtype), # [batch, seqlen_q, heads, num_split, dim]
        Output: T.Buffer(shape_q, dtype),
    ):
        # flash_attn_split(Q, K, V, glse, Output_partial)
        main_split(Q, K, V, Output_partial, glse)
        combine(glse, Output_partial, Output)

    return main


def ref_program(Q, K, V, glse, Output_partial, casual):
    assert casual is False
    dim = Q.size(-1)
    scores = torch.einsum('bqhd,bkhd->bhqk', Q, K)
    scores = scores / torch.sqrt(torch.tensor(dim, dtype=scores.dtype))
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.einsum('bhqk,bkhd->bqhd', attention_weights, V)
    return output

def reduce_ref(Q, K, V, glse, Output_partial, casual):
    o = torch.empty_like(Output_partial[:, :, :, 0, :]).fill_(0)
    lse_logsum = torch.empty_like(glse[:, :, 0, :]).fill_(0) # [batch, seqlen_q, heads]
    lse_max = glse.max(dim=2, keepdim=False).values
    for ks in range(num_split):
        lse = glse[:, :, ks, :]
        lse_logsum += torch.exp2(lse - lse_max)
    lse_logsum = torch.log2(lse_logsum) + lse_max
    for ks in range(num_split):
        lse = glse[:, :, ks, :]
        scale = torch.exp2(lse - lse_logsum) # [batch, heads, seqlen_q]
        o += Output_partial[:, :, :, ks, :] * scale[:, :, :, None].transpose(1, 2)
    return o.to(torch.float16)

def flash_split_ref(Q, K, V, casual):
    # [batch, seqlen_q, heads, dim]
    batch = Q.size(0)
    block_M = Q.size(1)
    nheads = Q.size(2)
    dim = Q.size(3)
    block_N = 128
    seqlen_kv = K.size(1)

    scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    acc_s = torch.empty((batch, nheads, block_M, block_N), device="cuda", dtype=torch.float)
    acc_s_cast = torch.empty((batch, nheads, block_M, block_N), device="cuda", dtype=torch.float16)
    acc_o = torch.empty((batch, block_M, nheads, dim), device="cuda", dtype=torch.float)
    scores_max = torch.empty((batch, nheads, block_M), device="cuda", dtype=torch.float)
    scores_max_prev = torch.empty((batch, nheads, block_M), device="cuda", dtype=torch.float)
    scores_scale = torch.empty((batch, nheads, block_M), device="cuda", dtype=torch.float)
    scores_sum = torch.empty((batch, nheads, block_M), device="cuda", dtype=torch.float)
    logsum = torch.empty((batch, nheads, block_M), device="cuda", dtype=torch.float)
    gacc_o = torch.empty((num_split, batch, block_M, nheads, dim), device="cuda", dtype=torch.float)
    glogsum = torch.empty((num_split, batch, nheads, block_M), device="cuda", dtype=torch.float)
    gscore_max = torch.empty((num_split, batch, nheads, block_M), device="cuda", dtype=torch.float)
    
    Q_ = Q * scale

    for ks in range(num_split):
        acc_o.fill_(0)
        logsum.fill_(0)
        scores_max.fill_(float('-inf'))
        scores_max_prev.fill_(float('-inf'))
        for i in range(int((seqlen_kv // num_split) / block_N)):
            acc_s.fill_(0)
            acc_s = torch.einsum('bqhd,bkhd->bhqk', Q_, K[:, (seqlen_kv // num_split) * ks + i * block_N : (seqlen_kv // num_split) * ks + (i + 1) * block_N, :, :]) # [batch, seqlen, nheads, block_N]
            scores_max_prev = scores_max
            scores_max = acc_s.max(dim=-1, keepdim=False).values # [blockM]
            scores_scale = torch.exp2(scores_max_prev - scores_max)
            acc_o *= scores_scale[:, :, :, None].transpose(1, 2)
            acc_s = torch.exp2(acc_s - scores_max[:, :, :, None])
            acc_s_cast = acc_s.to(torch.float16)
            acc_o += torch.einsum('bhqk,bkhd->bqhd', acc_s_cast, V[:, (seqlen_kv // num_split) * ks + i * block_N : (seqlen_kv // num_split) * ks + (i + 1) * block_N, :, :])
            scores_sum = acc_s.sum(dim=-1, keepdim=False)
            logsum = logsum * scores_scale + scores_sum
        acc_o /= logsum[:, :, :, None].transpose(1, 2)
        logsum = torch.log2(logsum) + scores_max
        gacc_o[ks, :, :, :, :] = acc_o
        glogsum[ks, :, :, :] = logsum

    return glogsum.to(torch.float16).permute(1, 2, 0, 3), gacc_o.to(torch.float16).permute(1, 2, 3, 0, 4)

if __name__ == "__main__":
    BATCH, H, Q_CTX, KV_CTX, D_HEAD, D_HEADV = 1, 32, 128, 8192, 128, 128
    casual = False
    flops_per_matmul = 2.0 * BATCH * H * Q_CTX * KV_CTX * (D_HEAD +D_HEADV)
    total_flops = flops_per_matmul
    if casual:
        total_flops *= 0.5
    BLOCK_M = 128
    BLOCK_N = 64 # if D_HEAD <= 128 else 32
    program = flashattn(BATCH, H, Q_CTX, KV_CTX, D_HEAD, D_HEADV, casual, BLOCK_M, BLOCK_N)
    ref_program = partial(ref_program, casual=casual)
    mod, params = tl.lower(program)
    mod = tl.Profiler(mod, params, [5], tl.TensorSupplyType.Normal)
    mod.assert_allclose(ref_program, rtol=0.01, atol=0.01)
    print("All checks passed!")

    latency = mod.do_bench(ref_program, warmup=500)
    print("{:.2f} ms".format(latency))
    print("{:.2f} TFlops".format(total_flops / latency * 1e-9))
    latency = mod.do_bench(mod, n_warmup=10, n_repeat=10, profiler="tvm")
    print("{:.4f} ms".format(latency))
    print("{:.2f} TFlops".format(total_flops / latency * 1e-9))