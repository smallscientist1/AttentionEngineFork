from attn_engine import AttentionEngine
import torch
import math
from attn_engine import OnlineFunc
from core import CustomIO
from core import SymbolicArray, SymbolScalar, SymbolicTensor
from core import Var
from core import meta_tensor

"""
Example of causal attention with online softmax
"""



window_size = 256
# mask on attention score
def block_sparse_mask(b, h, q_idx, kv_idx):
    return torch.logical_and(q_idx >= kv_idx, q_idx < kv_idx + window_size)

D = 128
softmax_scale = 1/D ** 0.5
# elementwise on attention scores
def score_mod(score, custom_fwd_inputs, b, h, q_idx, kv_idx):
    # softmax_scale = custom_fwd_inputs.input_tensors["softmax_scale"]
    return score * softmax_scale

class OnlineSoftmax(OnlineFunc):
    def __init__(self):
        """
        define online_rowscales and final_rowscales
        """
        online_rowscales = {
            "m": SymbolScalar("m", Var("-inf")), # -inf 
            "r": SymbolScalar("r", Var("0.0")),
        }
        final_rowscales = {
            "lse": SymbolScalar("lse", Var("0.0")), # not used in codegen
        }
        external_fwd_inputs = CustomIO()
        # external_bwd_inputs = CustomIO({
        #     "dppsum": (B,H,S),
        # })
        super().__init__(online_rowscales, final_rowscales,
                    external_fwd_inputs) # , external_bwd_inputs)
    

    @staticmethod
    def online_fwd(scores, online_rowscales, b, h, q_idx):

        m , r = online_rowscales["m"], online_rowscales["r"]
        m_new = m.max(scores.get_reduce("max"))
        scale_tmp = (m - m_new).exp()
        r = r * scale_tmp
        
        scores = (scores - m_new).exp()
        r = r + scores.get_reduce("sum")

        new_online_rowscales = {
            "m": m_new,
            "r": r,
        }
        o_scale = scale_tmp
        return scores, new_online_rowscales, o_scale

    @staticmethod
    def online_fwd_epilogue(o, online_rowscales, b, h, q_idx):
        o_new = o / (online_rowscales["r"] + 1e-30) # TO avoid division by zero
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
    def backward(dp, scores, final_rowscales, doosum_rowscales, b, h, q_idx, kv_idx):
        dppsum = doosum_rowscales # external_bwd_tensor.input_tensors["dppsum"]
        dscores = (dp - dppsum)*scores # TODO: bug if swap
        return dscores

if __name__ == "__main__":
    B, H ,S, D, DV = 4,32,2048,D, 128
    dtype = torch.bfloat16 # performance regression for bfloat16
    qkv_meta = (
        meta_tensor(B, H, S, D, dtype=dtype),
        meta_tensor(B, H, S, D, dtype=dtype),
        meta_tensor(B, H, S, DV, dtype=dtype),
    )

    custom_fwd_inputs = CustomIO({
        # "softmax_scale": (1,),
    })

    online = OnlineSoftmax()

    mod = AttentionEngine(
        qkv_meta,
        custom_fwd_inputs, score_mod=score_mod, mask_mod=block_sparse_mask,
        online_func=online,
        tune=False, tune_file="mha_tune.json",
        infer_mask=True # False
    )

    # test sliding window attention
    torch.cuda.manual_seed(0)
    q = torch.randn(B, S, H, D, dtype=dtype, device="cuda")
    k = torch.randn(B, S, H, D, dtype=dtype, device="cuda")
    v = torch.randn(B, S, H, DV, dtype=dtype, device="cuda")
    o = mod(q,k,v)

    def sliding_window_attn(q, k, v, window_size):
        # Get dimensions
        batch_size, seq_len, num_heads, head_dim = q.size()
        
        # Initialize attention output
        output = torch.zeros_like(q)
        
        # Calculate scaling factor
        scale = (head_dim ** -0.5)
        
        # Iterate over each position in the sequence
        for i in range(seq_len):
        # for i in range(0, 384):
            # Determine window range
            start = max(0, i - window_size + 1)
            # start = max(0, i - window_size)
            end = min(seq_len, i+1)
            
            # Select local window of keys and values
            k_window = k[:, start:end, :, :]
            v_window = v[:, start:end, :, :]
            
            # Compute attention scores
            attn_scores = torch.einsum("bhd,bkhd->bhk", q[:, i, :, :], k_window).float()
            attn_scores = attn_scores * scale
            
            # Apply softmax to get attention weights
            attn_weights = torch.softmax(attn_scores, dim=-1)
            
            # Compute the weighted sum of values
            output[:, i, :, :] = torch.einsum("bhk,bkhd->bhd", attn_weights.to(v.dtype), v_window)
        
        return output
    
    
    ref_o = sliding_window_attn(q, k, v, window_size)
    print(torch.allclose(o, ref_o, atol=1e-2, rtol=1e-2))
    from benchmark.bench_utils import print_debug
    print_debug(o[:,:,:,:], ref_o[:,:,:,:], atol=1e-2, rtol=1e-2)
    
