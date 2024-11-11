import torch
from attn_template import TlAttnTemplate
from lower import lower_tl

import importlib.util
import tempfile
import os
import hashlib

class AttentionEngine:
    def __init__(self, query, key, value, custom_fwd_inputs, custom_bwd_inputs, score_mod, block_mask,
    online_func,):
        tl_code = lower_tl(score_mod, block_mask, online_func, custom_fwd_inputs)
        # local_vars = {}
        # exec(tl_code, globals(), local_vars)
        # # 将 local_vars 转化为全局变量
        # globals().update(local_vars)
        # self.attention = local_vars["attention"]
        code_hash = hashlib.md5(tl_code.encode()).hexdigest()
        cache_dir = os.path.join(os.path.dirname(__file__),"cache")
        file_path = os.path.join(cache_dir, f"{code_hash}.py")
        os.makedirs(cache_dir, exist_ok=True)
        if not os.path.exists(file_path):
            with open(file_path, "w") as f:
                f.write(tl_code)
                f.flush()
        spec = importlib.util.spec_from_file_location("tl_attn", file_path)
        tl_attn = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tl_attn)
        self.attention = tl_attn.attention

    def __call__(self, *args, **kargs):
        
        o = self.attention(*args, **kargs)
        return o

      