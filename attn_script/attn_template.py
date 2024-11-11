import jinja2

TL_IMPORT = """
import torch
from tvm import tl
import tvm.tl.language as T
"""

TL_KERNEL = """
def kernel(batch, heads, seq_len, dim, dimv, 
        block_M = None, block_N = None, num_stages = None, thread_num = None):
    # scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e) # 0.69314718  loge(2)
    shape = [batch, seq_len, heads, dim]
    shape_v = [batch, seq_len, heads, dimv]
    dtype = "float16"
    accum_dtype = "float"
    
    # TODO: mask
    is_casual = True
"""
TL_MAIN = """

    @T.macro
    def score_mod(
        # scores: T.Buffer([block_M, block_N], accum_dtype),
        {{score_mod_inputs | indent(8)}}
        ):
        {{score_mod_body | indent(8)}}
    
    @T.macro
    def online_func(
        # scores: T.Buffer([block_M, block_N], accum_dtype),
        {{online_func_inputs | indent(8)}}
    ):
        {{online_func_body | indent(8)}}

        
    @T.prim_func
    def main(
        Q: T.Buffer(shape, dtype), # type: ignore
        K: T.Buffer(shape, dtype), # type: ignore
        V: T.Buffer(shape_v, dtype), # type: ignore
        {{custom_fwd_inputs | indent(8)}}

        Output: T.Buffer(shape_v, dtype), # type: ignore
        {{final_rowscales_output | indent(8)}}
    ):
        with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch, threads=thread_num) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dimv], dtype)
            # acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
            # acc_o = T.alloc_fragment([block_M, dimv], accum_dtype)

            {{custom_fwd_inputs_init | indent(12)}}
            {{online_func_init | indent(12)}}
            {{final_rowscales_init | indent(12)}}

            T.annotate_layout({Q_shared: tl.layout.make_swizzled_layout(Q_shared)})
            T.copy(Q[bz, bx * block_M : (bx + 1) * block_M, by, :], Q_shared)
            T.fill(acc_o, 0)

            {{online_rowscales_initvalue | indent(12)}}

            # TODO: mask
            loop_range = (
                T.ceildiv((bx + 1) * block_M, block_N) if is_casual else T.ceildiv(seq_len, block_N)
            )

            for k in T.Pipelined(loop_range, num_stages=num_stages):
                T.copy(K[bz, k * block_N : (k + 1) * block_N, by, :], K_shared)

                # TODO: copy custom_fwd_input_tensor in score_mod&online_func
                # ...

                # TODO: naive solution: if reduce_max, -T.inf; if reduce_sum, 0
                if is_casual and {{is_inf_mask}}:
                    for i, j in T.Parallel(block_M, block_N):
                        scores[i, j] = T.if_then_else(
                            bx * block_M + i >= k * block_N + j, 0, -T.infinity(scores.dtype)
                        )
                else:
                    T.clear(scores)
                
                T.gemm(Q_shared, K_shared, scores, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.copy(V[bz, k * block_N : (k + 1) * block_N, by, :], V_shared)
                    
                # call score_mod
                score_mod({{score_mod_inputs_list}}) # scores
                    
                # call online_func
                online_func({{online_func_inputs_list}}) # scores

                for i, j in T.Parallel(block_M, dimv):
                    acc_o[i, j] *= {{o_scale}}[i]
                
                # update online_rowscales
                {{online_rowscales_update | indent(16)}}

                T.copy(scores, acc_s_cast)
                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
            
            # online_fwd_epilogue
            {{online_func_epilogue | indent(12)}}

            T.copy(acc_o, Output[bz, bx * block_M : (bx + 1) * block_M, by, :])

            # save final_rowscale
            {{final_rowscales_save | indent(12)}}
        
    return main

"""

TL_INFERFACE = """
class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, *custom_fwd_inputs):
        BATCH, N_CTX, H, D_HEAD = q.shape
        D_HEADV = v.shape[-1]
        block_M = 128
        block_N = 128 if D_HEAD <= 128 else 64
        stages = 1
        thread_num = 256
        output_idx_list = {{output_idx_list}}
        mod = tl.cached(kernel, output_idx_list, BATCH, H, N_CTX, D_HEAD, D_HEADV, block_M, block_N, stages, thread_num)
        o, *final_scale = mod(q, k, v, *custom_fwd_inputs)
        ctx.save_for_backward(q, k, v, *custom_fwd_inputs, o, *final_scale)
        return o
    
    @staticmethod
    def backward(ctx, grad_o, *bwd_inputs):
        pass

attention = _attention.apply
"""
class TlAttnTemplate:
    TL_IMPORT = TL_IMPORT
    TL_KERNEL = TL_KERNEL
    TL_MAIN = TL_MAIN
    TL_INFERFACE = TL_INFERFACE
    def __init__(self, custom_fwd_inputs=None, final_rowscales_output=None, 
                 custom_fwd_inputs_init=None, online_func_init=None, final_rowscales_init=None,
                 online_rowscales_initvalue=None,
                 online_rowscales_update=None,
                 is_inf_mask=False, 
                 score_mod_inputs=None, online_func_inputs=None,
                score_mod_body="return", online_func_body="return",
                 score_mod_inputs_list=None, online_func_inputs_list=None, o_scale="o_scale",
                 online_func_epilogue=None, final_rowscales_save=None,
                 output_idx_list="[3]",):

        template = jinja2.Template(TlAttnTemplate.TL_IMPORT+
                                   TlAttnTemplate.TL_KERNEL+
                                   TlAttnTemplate.TL_MAIN+
                                   TlAttnTemplate.TL_INFERFACE)
        kargs = {
            "custom_fwd_inputs": custom_fwd_inputs,
            "final_rowscales_output": final_rowscales_output,
            "custom_fwd_inputs_init": custom_fwd_inputs_init,
            "online_func_init": online_func_init,
            "online_rowscales_update": online_rowscales_update,
            "online_rowscales_initvalue": online_rowscales_initvalue,
            "final_rowscales_init": final_rowscales_init,
            "is_inf_mask": is_inf_mask,
            "score_mod_inputs": score_mod_inputs,
            "online_func_inputs": online_func_inputs,
            "score_mod_body": score_mod_body,
            "online_func_body": online_func_body,
            "score_mod_inputs_list": score_mod_inputs_list,
            "online_func_inputs_list": online_func_inputs_list,
            "o_scale": o_scale,
            "online_func_epilogue": online_func_epilogue,
            "final_rowscales_save": final_rowscales_save,
            "output_idx_list": output_idx_list
        }
        # remove None
        kargs = {k:(v if v is not None else "") for k,v in kargs.items() }
        self.tlcode = template.render(**kargs)
    def __call__(self):
        return self.tlcode

if __name__ == "__main__":
    tl_code = TlAttnTemplate()()
    print(tl_code)
    exec(tl_code)
    print(attention)
