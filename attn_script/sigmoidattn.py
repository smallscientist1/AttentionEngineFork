from attn_engine import AttentionEngine
import torch
import math
from attn_engine import OnlineFunc
from core.core import CustomIO
from core.core import create_block_mask
from core.core import SymbolicArray, SymbolScalar, SymbolicTensor
from core.core import Var

def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

def score_mod(score, custom_fwd_inputs, b, h, q_idx, kv_idx):
    softmax_bias = custom_fwd_inputs.input_tensors["softmax_bias"]
    score = ((score*0.5).tanh() + 1) * 0.5 + softmax_bias
    return score

# def score_mod(score, custom_fwd_inputs, b, h, q_idx, kv_idx):
#     return 1 / (1 + (-score).exp())


class OnlineIdentity(OnlineFunc):
    def __init__(self):
        online_rowscales = {
        }
        final_rowscales = {
        }
        external_fwd_inputs = CustomIO()
        super().__init__(online_rowscales, final_rowscales,
                    external_fwd_inputs)

    @staticmethod
    def online_fwd(scores, online_rowscales, b, h, q_idx):
        o_scale = SymbolScalar("o_scale", Var("1"))
        return scores, online_rowscales, o_scale
    
    @staticmethod
    def online_fwd_epilogue(o, online_rowscales, b, h, q_idx):
        return o, {}
    
    @staticmethod
    def forward(scores, final_rowscales, b, h, q_idx, kv_idx):
        return scores

    @staticmethod
    def backward(dp, scores, final_rowscales, doosum_rowscales, b, h, q_idx, kv_idx):
        return dp
   

if __name__ == "__main__":
    B, H ,S, D = 16,16,8192,64
    custom_fwd_inputs = CustomIO({
        "softmax_bias": (1,),
    })

    block_mask = create_block_mask(causal_mask, 1, 1, S, S, device="cuda")

    mod = AttentionEngine(
        custom_fwd_inputs, score_mod=score_mod, block_mask=block_mask,
        online_func=OnlineIdentity(),
    )
    with open("sigmoid_tl_code.py", "w") as f:
        f.write(mod.tl_code)
    from benchmark.bench_utils import do_bench_sigmoidattn
    do_bench_sigmoidattn(mod, B, H, S, D, D)

