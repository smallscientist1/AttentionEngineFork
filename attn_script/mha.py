import create_block_mask
import attention_engine
import torch
from core import OnlineFunc

"""
AttentionEngine: o = attention_engine(query: Tensor, key: Tensor, value: Tensor,
                            query_mod, key_mod, value_mod,
                            score_mod, block_mask, 
                            online_func)

# It basically does:

scores = query @ key
p = online_func(scores)
o = p @ value

# A more detailed version is:

scores = query_mod(query) @ key_mod(key)
scores = block_mask(scores)
scores = score_mod(scores)
p = online_func(scores)
o = p @ value_mod(value)


# Parameters:

*_mod is user-defined element-wise function, default is identity function.
block_mask is a mask that applied to scores. the block_mask can be implemented with score_mod, but with block_mask, better performance can be achieved.
online_func is a class that has **row-online algorithm** for scores. We have predefined online_softmax, online_retention, Identity. User can also define their own online algorithm.


# How to define your own online algorithm:

## the online algorithm template:

### online forward: 

online_rowscales = {
    "r0": INIT_VALUE0,
    "r1": INIT_VALUE1,
} 
final_rowscales = {
    "m0": INIT_VALUE,
    "m1": INIT_VALUE,
}
o = torch.zeros(B, H, q_len, D, device="cuda", dtype=torch.float16)
for i in range(kv_len//BLOCK_N):
    scores = q @ k
    scores = block_mask(scores)
    scores = score_mod(scores)
    o_scale = online_func.online_fwd(scores, online_rowscales)
    o = o*o_scale
    o += scores @ v
    k.advance(BLOCK_N)
    v.advance(BLOCK_N)
online_func.set_final_rowscales(final_rowscales, online_rowscales)
online_func.scale_final_o(o, online_rowscales)

### backward:

scores = q @ k
scores = block_mask(scores)
scores = score_mod(scores)
online_func.forward(scores, final_rowscales)
dv = scores^T @ do
dp = do @ v^T
dscores = online_func.backward(dp, scores)
dk = dscores^T @ q
dq = dscores @ k



## User need to do:
1. Define a class that inherits from OnlineFunc.
2. Implement __init__, online_fwd, set_final_rowscales, forward, backward, scale_final_o methods.




 
"""

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


# mask on attention score
def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

block_mask = create_block_mask(causal_mask, 1, 1, S, S, device="cuda")

# elementwise on attention score
def score_mod(score: scalar, b, h, q_idx, kv_idx):
    return score / sqrt(D)


dppsum = torch.sum(do * o, dim=-1)

class online_softmax(OnlineFunc):
    """
    __init__: define online_rowscales and final_rowscales
        online_rowscales: intermediate scale results for online algorithm
        final_rowscales: final scale results for online algorithm

    online_fwd: online algorithm for generate attention forward

    set_final_rowscales: set final rowscales at the end of attention forward, save it for backward

    forward: forward algorithm g(scores, scale) for backward recompute
    backward: backward algorithm
    """
    def __init__(self):
        """
        define online_rowscales and final_rowscales
        """
        online_rowscales = {
            "m": -torch.inf,
            "r": 0.0,
            "m_new": 0,
        }
        final_rowscales = {
            "lse": -torch.inf,
        }
        super().__init__(online_rowscales, final_rowscales)
    

    @staticmethod
    def online_fwd(scores:Array,online_rowscales, b, h, q_idx):
        """
        input: 
        scores: 一维向量, 仅包含reduce(),elementwise()操作
        online_rowscales: 保存在线算法的中间结果

        return: 
            o_scale:  for online rescale o
        """
        m , r = online_rowscales["m"], online_rowscales["r"]
        m_new = max(m, scores.getreduce("max"))
        r = r * exp(m - m_new)
        
        scores.data() = exp(scores.data() - m_new)
        r = r + scores.getreduce("sum")

        online_rowscales["m"] = m_new
        online_rowscales["r"] = r
        # TODO: check m ,r is updated.
        # decorator

        o_scale = exp(m - m_new)
        return o_scale
    
    @staticmethod
    def set_final_rowscales(final_rowscales, online_rowscales, b, h, q_idx):
        """
        compute final_rowscales at the end of online attention forward
        """
        lse = log(online_rowscales["r"]) + online_rowscales["m"]
        final_rowscales["lse"] = lse

    @staticmethod
    def scale_final_o(o, online_rowscales):
        """
        scale final o with final_rowscales
        """
        o = o / online_rowscales["r"]

    @staticmethod
    def forward(self, scores, final_rowscales, b, h, q_idx):
        """
        compute scores : scores = g(scores, scale)
        """
        lse = final_rowscales["lse"]
        scores = exp(scores-lse)
    
    @staticmethod
    def backward(dp:scalar, scores:scalar, final_rowscales, b, h, q_idx, kv_idx):
        """
        compute bwd scores: dscores = g_bwd(dp, scores)
        only support elementwise: 只支持这样写；报错；readme
        """
        dscores = scores*dp*dppsum[b,h,q_idx,kv_idx]

        return dscores




attention_engine(
    query, key, value, score_mod=score_mod, block_mask=block_mask,
    online_func=online_softmax
)
