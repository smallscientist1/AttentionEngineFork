import torch
from core import OnlineFunc
import create_block_mask

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
mask = torch.randn(
        H, S, S, device="cuda", dtype=torch.float16, requires_grad=True
    )


# mask on attention score
def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

block_mask = create_block_mask(causal_mask, 1, 1, S, S, device="cuda")

# elementwise on attention score
def score_mod(score: scalar, b, h, q_idx, kv_idx):
    return score * mask[h,q_idx,kv_idx] / sqrt(D)



class online_softmax(OnlineFunc):
    def __init__(self):
        online_rowscales = {
            "r_wo_clamp": 0.0,
            "r": 0.0,
        }
        final_rowscales = {
            "r": 0.0,
        }
        super().__init__(online_rowscales, self.final_rowscales)
    
    @staticmethod
    def online_fwd(scores:torch.Tensor,online_rowscales, b, h, q_idx):
        r_wo_clamp = online_rowscales["r_wo_clamp"]
        r = online_rowscales["r"]
        r_wo_clamp = r_wo_clamp + abs(scores).getreduce("sum")
        r_new = max(r_wo_clamp, 1)
        o_scale = r / r_new

        scores = scores / r_new

        online_rowscales["r_wo_clamp"] = r_wo_clamp
        online_rowscales["r"] = r_new

        return o_scale
    
    def set_final_rowscales(final_rowscales, online_rowscales, b, h, q_idx):
        """
        compute final_rowscales at the end of online attention forward
        """
        final_rowscales["r"] = online_rowscales["r"]

    def scale_final_o(o, online_rowscales):
        """
        scale final o with final_rowscales
        """
        pass

    def forward(self, scores, final_rowscales, b, h, q_idx):
        """
        compute scores : scores = g(scores, scale)
        """
        r = final_rowscales["r"]
        scores = scores / r
    
    def backward(dp:scalar, scores:scalar, final_rowscales, b, h, q_idx, kv_idx):
        """
        compute bwd scores: dscores = g_bwd(dp, scores)
        """
        dscores = dp / final_rowscales["r"]

        return dscores




attention_engine(
    query, key, value, score_mod=score_mod, block_mask=block_mask,
    online_func=online_softmax
)
