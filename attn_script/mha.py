from core import AttentionEngine
import torch
import math
from core import OnlineFunc
from core import CustomIO
from core import create_block_mask
from core import SymbolicArray, SymbolScalar, SymbolicTensor
from core import Var

"""
Example of causal attention with online softmax
"""

B, H ,S, D = 16,16,8192,64
query = torch.randn(
        B, H, S, D, device="cuda", dtype=torch.float16, requires_grad=True
    )
key = torch.randn(
        B, H, S, D, device="cuda", dtype=torch.float16, requires_grad=True
    )
value = torch.randn(
        B, H, S, D, device="cuda", dtype=torch.float16, requires_grad=True
    )

do = torch.randn(
        B, H, S, D, device="cuda", dtype=torch.float16, requires_grad=True
    )


# mask on attention score
def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

block_mask = create_block_mask(causal_mask, 1, 1, S, S, device="cuda")

# elementwise on attention scores
def score_mod(score, custom_fwd_inputs, b, h, q_idx, kv_idx):
    softmax_scale = custom_fwd_inputs["softmax_scale"]
    return score / softmax_scale

class OnlineSoftmax(OnlineFunc):
    def __init__(self):
        """
        define online_rowscales and final_rowscales
        """
        online_rowscales = {
            "m": SymbolScalar("m", Var("-inf")), # -inf # TODO: init value
            "r": SymbolScalar("r", Var("0.0")),
        }
        final_rowscales = {
            "lse": SymbolScalar("lse", Var("0.0")), # 0
        }
        external_fwd_inputs = CustomIO()
        external_bwd_inputs = CustomIO({
            "dppsum": (B,H,S),
        })
        super().__init__(online_rowscales, final_rowscales,
                    external_fwd_inputs, external_bwd_inputs)
    

    @staticmethod
    def online_fwd(scores, online_rowscales, b, h, q_idx):

        m , r = online_rowscales["m"], online_rowscales["r"]
        m_new = m.max(scores.get_reduce("max"))
        r = r * (m - m_new).exp()
        
        scores = (scores - m_new).exp()
        r = r + scores.get_reduce("sum")

        new_online_rowscales = {
            "m": m_new,
            "r": r,
        }
        o_scale = (m - m_new).exp()
        return scores, new_online_rowscales, o_scale

    @staticmethod
    def online_fwd_epilogue(o, online_rowscales, b, h, q_idx):
        o_new = o / online_rowscales["r"]
        lse = (online_rowscales["r"]).log() + online_rowscales["m"]
        final_rowscales = {
            "lse": lse,
        }
        return o_new, final_rowscales

    @staticmethod
    def forward(scores, final_rowscales, b, h, q_idx, kv_idx):
        lse = final_rowscales["lse"]
        scores_new = (scores-lse).exp()
        return scores_new
    
    @staticmethod
    def backward(dp, scores, final_rowscales, external_bwd_tensor, b, h, q_idx, kv_idx):
        dppsum = external_bwd_tensor.input_tensors["dppsum"]
        dscores = scores*dp*dppsum
        return dscores


custom_fwd_inputs = CustomIO({
    "softmax_scale": (1,),
})
custom_bwd_inputs = CustomIO({
    "dppsum": (B,H,S),
})

online = OnlineSoftmax()
mod = AttentionEngine(
    query, key, value, custom_fwd_inputs, custom_bwd_inputs, score_mod=score_mod, block_mask=block_mask,
    online_func=online,
)
# check

scores,online_rowscales,o_scale = online.online_fwd(SymbolicArray(), online.online_rowscales, 1, 1, 1)
o, final_scales = online.online_fwd_epilogue(SymbolScalar("o",Var("o")), online.online_rowscales, 1, 1, 1)
scores2 = online.forward(SymbolicArray(), online.final_rowscales, 1, 1, 1, 1)
dscores = online.backward(SymbolScalar("dp",Var("dp")), SymbolScalar("scores",Var("scores")), online.final_rowscales, online.external_bwd_tensors, 1, 1, 1, 1)

print(custom_fwd_inputs.input_tensors)
softmax_scale = math.sqrt(D)
o = mod(query, key, value, softmax_scale=softmax_scale)
print(custom_bwd_inputs.input_tensors)
dppsum = torch.sum(do * o, dim=-1)
mod.backward(do, dppsum=dppsum)
