import torch
from core.attn_template import TlAttnTemplate
from core.lower import lower_tl
from core.core import CustomIO, SymbolicArray, SymbolScalar, Var

import importlib.util
import tempfile
import os
import hashlib

class OnlineFunc:
    """
    __init__: define online_rowscales and final_rowscales
        online_rowscales: intermediate scale results for online algorithm
        final_rowscales: final scale results for online algorithm

    online_fwd: online algorithm for generate attention forward

    set_final_rowscales: set final rowscales at the end of attention forward, save it for backward

    forward: forward algorithm g(scores, scale) for backward recompute
    backward: backward algorithm
    """
    def __init__(self, online_rowscales:dict[str, SymbolScalar], final_rowscales:dict[str, SymbolScalar], 
                 external_fwd_tensors:CustomIO): # , external_bwd_tensors:CustomIO):
        # TODO: external_tensors
        """
        define&init online_rowscales and final_rowscales
        """
        self.online_rowscales = online_rowscales
        self.final_rowscales = final_rowscales
        self.vars = {
            "scores": SymbolicArray(),
            "o_scale": None,
        }
        self.external_fwd_tensors = external_fwd_tensors
        # self.external_bwd_tensors = external_bwd_tensors
        self.doosum_rowscales = SymbolicArray("doosum", Var("doosum"), shape_idx=["block_M"])
        
    
    @staticmethod
    def online_fwd(scores:SymbolicArray, online_rowscales, b, h, q_idx):
        """
        compute scores, online_rowscale, o_scale
        input: 
            scores: 一维向量, 仅包含getreduce()
            online_rowscales: 在线算法的上一轮中间结果
        return:
            scores: 一维向量
            online_rowscales: 保存在线算法的更新后中间结果
            o_scale:  for online rescale o

        """
        o_scale = SymbolScalar("o_scale", Var("o_scale"))
        return scores, online_rowscales, o_scale

    @staticmethod
    def online_fwd_epilogue(o, online_rowscales, b, h, q_idx):
        """
        compute o, final_rowscales at the end of online attention forward
        return:
            o: online_fwd 最后对o进行的缩放
            final_rowscales: online_fwd执行完成后保存的scale，用于backward
        """
        final_rowscales = online_rowscales
        return o, final_rowscales

    @staticmethod
    def forward(scores, final_rowscales:dict[str, SymbolScalar], b, h, q_idx, kv_idx):
        """
        compute scores : scores = g(scores, scale), 
            final_rowscales is saved during online forward
        return 
        """
        return scores
    
    @staticmethod
    def backward(dp, scores, final_rowscales:dict[str, SymbolScalar], b, h, q_idx, kv_idx):
        """
        compute bwd scores: dscores = g_bwd(dp, scores)
        only support elementwise
        """
        dscores = dp
        return dscores

class AttentionEngine:
    def __init__(self, query, key, value, custom_fwd_inputs, custom_bwd_inputs, score_mod, block_mask,
    online_func,):
        tl_code = lower_tl(score_mod, block_mask, online_func, custom_fwd_inputs)
        self.tl_code = tl_code # for debug
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

if __name__ == "__main__":
    online = OnlineFunc({},{}, CustomIO(), CustomIO())
    scores,online_rowscales,o_scale = online.online_fwd(SymbolicArray(), online.online_rowscales, 1, 1, 1)
    o, final_scales = online.online_fwd_epilogue(SymbolScalar("o",Var("o")), online.online_rowscales, 1, 1, 1)
    scores2 = online.forward(SymbolicArray(), online.final_rowscales, 1, 1, 1, 1)
    dscores = online.backward(SymbolScalar("dp",Var("dp")), SymbolScalar("scores",Var("scores")), online.final_rowscales, online.external_bwd_tensors, 1, 1, 1, 1)
      