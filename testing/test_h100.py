from examples.mha import causal_softmax_attention
from examples.mha_decode import softmax_attention_decode

import torch
import torch.nn.functional as F
from einops import rearrange, einsum

def test_attention():

    test_softmaxattention(1, 16, 2048, 128, 128)
    # test_softmaxattention_decode(8, 16, 1, 4096, 128, 128) # TODO: compile TileLang with llvm

    print("All tests pass.")
    
def test_softmaxattention(B, H, S, D, DV, device="cuda", dtype=torch.float16, require_grad=True):
    attention_module = causal_softmax_attention(B, H, S, D, DV)
    
    def ref(query, key, value, causal=True, softmax_scale=None):
        dim = query.shape[-1]
        num_head_groups = query.shape[2] // key.shape[2]
        if softmax_scale is None:
            softmax_scale = 1 / dim** 0.5

        query = rearrange(
            query, 'b s (h g) d -> b s g h d',
            g=num_head_groups)  # [batch_size, num_head_groups, groups, dim]
        scores = einsum(query, key,
        'b s g h d, b t h d -> b g h s t')
        if causal:
            seqlenq = query.shape[1]
            seqlenk = key.shape[1]
            mask = torch.tril(
                torch.ones(
                    seqlenq, seqlenk, device=scores.device))
            mask = mask.unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention = F.softmax(
            scores * softmax_scale, dim=-1)

        out = einsum(attention, value,
                'b g h s t, b t h d -> b g h s d')
        out = rearrange(out, 'b g h s d -> b s (h g) d') 
        return out
    
    # init input
    query = torch.randn(B, S, H, D, device=device, dtype=dtype, requires_grad=require_grad)
    key = torch.randn(B, S, H, D, device=device, dtype=dtype, requires_grad=require_grad)
    value = torch.randn(B, S, H, DV, device=device, dtype=dtype, requires_grad=require_grad)
    ref_o = ref(query, key, value)
    o = attention_module(query, key, value)
    torch.testing.assert_close(o, ref_o, rtol=1e-2, atol=1e-2)
    
    # TODO: bwd

def test_softmaxattention_decode(B, H, S, KV, D, DV, device="cuda", dtype=torch.float16, require_grad=True):
    attention_module = softmax_attention_decode(B, H, S, KV, D, DV)
    
    def ref(query, key, value, softmax_scale=None):
        dim = query.shape[-1]
        num_head_groups = query.shape[2] // key.shape[2]
        if softmax_scale is None:
            softmax_scale = 1 / dim** 0.5

        query = rearrange(
            query, 'b s (h g) d -> b s g h d',
            g=num_head_groups)  # [batch_size, num_head_groups, groups, dim]
        scores = einsum(query, key,
        'b s g h d, b t h d -> b g h s t')
        attention = F.softmax(
            scores * softmax_scale, dim=-1)

        out = einsum(attention, value,
                'b g h s t, b t h d -> b g h s d')
        out = rearrange(out, 'b g h s d -> b s (h g) d') 
        return out
    
    # init input
    query = torch.randn(B, S, H, D, device=device, dtype=dtype, requires_grad=require_grad)
    key = torch.randn(B, KV, H, D, device=device, dtype=dtype, requires_grad=require_grad)
    value = torch.randn(B, KV, H, DV, device=device, dtype=dtype, requires_grad=require_grad)
    ref_o = ref(query, key, value)
    o = attention_module(query, key, value)
    torch.testing.assert_close(o, ref_o, rtol=1e-2, atol=1e-2)
     
if __name__ == "__main__":
    test_attention()
