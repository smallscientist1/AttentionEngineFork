from attn_engine import AttentionEngine
import torch
import math
from core import OnlineFunc
from core import CustomIO
from core import create_block_mask
from core import SymbolicArray, SymbolScalar, SymbolicTensor
from core import Var

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

def score_mod(score, custom_fwd_inputs, b, h, q_idx, kv_idx):
    _1 = SymbolScalar("_1", Var("1"))
    return _1 / (_1 + (-score).exp())

class OnlineIdentity(OnlineFunc):
    def __init__(self):
        online_rowscales = {
        }
        final_rowscales = {
        }
        external_fwd_inputs = CustomIO()
        external_bwd_inputs = CustomIO({
        })
        super().__init__(online_rowscales, final_rowscales,
                    external_fwd_inputs, external_bwd_inputs)

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
    def backward(dp, scores, final_rowscales, external_bwd_tensors, b, h, q_idx, kv_idx):
        return dp
    
custom_fwd_inputs = CustomIO({
})
custom_bwd_inputs = CustomIO({
})
online = OnlineIdentity()
mod = AttentionEngine(
    query, key, value, custom_fwd_inputs, custom_bwd_inputs, score_mod=score_mod, block_mask=block_mask,
    online_func=online,
)

# check
score_mod(SymbolScalar("score",Var("score")), custom_fwd_inputs, 1, 1, 1, 1) # TODO: check bwd
scores,online_rowscales,o_scale = online.online_fwd(SymbolicArray(), online.online_rowscales, 1, 1, 1)
o, final_scales = online.online_fwd_epilogue(SymbolScalar("o",Var("o")), online.online_rowscales, 1, 1, 1)
scores2 = online.forward(SymbolicArray(), online.final_rowscales, 1, 1, 1, 1)
dscores = online.backward(SymbolScalar("dp",Var("dp")), SymbolScalar("scores",Var("scores")), online.final_rowscales, online.external_bwd_tensors, 1, 1, 1, 1)

print(custom_fwd_inputs.input_tensors)
o = mod(query, key, value)
print(custom_bwd_inputs.input_tensors)
mod.backward(do)

