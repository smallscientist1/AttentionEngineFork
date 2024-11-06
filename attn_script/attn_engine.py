import torch
# from core import lower_online_func
class AttentionEngine:
    def __init__(self, query, key, value, custom_fwd_inputs, custom_bwd_inputs, score_mod, block_mask,
    online_func,):
        # self.online_func_lowered = lower_online_func(online_func)
        pass

    def __call__(self, *args, **kargs):
        
        o = torch.tensor(1)
        return o

    def backward(self, *args, **kargs):
        pass
      