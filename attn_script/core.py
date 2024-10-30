import torch
from typing import Literal
import functools

class CustomIO:
    def __init__(self, ):
        self.input_tensors: dict[str, tuple] = {
        }
    def __call__(self, tensor_name:str, tensor_shape:tuple):
        self.input_tensors[tensor_name] = tensor_shape
        def decorator(func):
            def wrapper(*args, **kwargs):
            # 调用原始函数并返回结果
                result = func(*args, **kwargs)
            
                return result
            return wrapper
        return decorator


class SymbolScalar:
    def __init__(self, varname:str, value:str=None):
        self.varname = varname
        self.code = value
    def __add__(self, other):
        assert isinstance(other, SymbolScalar)
        return SymbolScalar(self.varname,f'{self.varname} = ({self.varname}+{other.varname})')
    def __sub__(self, other):
        assert isinstance(other, SymbolScalar)
        return SymbolScalar(self.varname,f'{self.varname} = ({self.varname}-{other.varname})')
    def __mul__(self, other):
        assert isinstance(other, SymbolScalar)
        return SymbolScalar(self.varname,f'{self.varname} = ({self.varname}*{other.varname})')
    def __truediv__(self, other):
        assert isinstance(other, SymbolScalar)
        return SymbolScalar(self.varname,f'{self.varname} = ({self.varname}/{other.varname})')
    def abs(self):
        return SymbolScalar(self.varname,f'{self.varname} = abs({self.varname})')
    def exp(self):
        return SymbolScalar(self.varname,f'{self.varname} = exp({self.varname})') # TODO
    def log(self):
        return SymbolScalar(self.varname,f'{self.varname} = log({self.varname})') # TODO
    def max(self, other):
        assert isinstance(other, SymbolScalar)
        return SymbolScalar(self.varname,f'{self.varname} = max({self.varname},{other.varname})') # TODO


class SymbolicArray(SymbolScalar):
    """
    Array for OnlineFunc.online_fwd
    """
    def __init__(self):
        self.code = ""
    def __add__(self, other):
        return SymbolicArray()
    def __sub__(self, other):
        return SymbolicArray()
    def __mul__(self, other):
        return SymbolicArray()
    def __truediv__(self, other):
        return SymbolicArray()
    def abs(self):
        return SymbolicArray()
    def exp(self):
        return SymbolicArray()
    def log(self):
        return SymbolicArray()
    def get_reduce(self,op:Literal["sum", "max"]):
        """
        get reduce result of array
        """
        return SymbolScalar(f' ') # TODO
    
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
    def __init__(self, online_rowscales:dict[str, SymbolScalar], final_rowscales:dict[str, SymbolScalar]):
        """
        define&init online_rowscales and final_rowscales
        """
        self.online_rowscales = online_rowscales
        self.final_rowscales = final_rowscales
    
    @staticmethod
    def online_fwd(scores:SymbolicArray, o_scale ,online_rowscales, b, h, q_idx):
        """
        compute scores and o_scale
        input: 
            scores: 一维向量, 仅包含getreduce()
            online_rowscales: 保存在线算法的中间结果
        """
        pass
    
    @staticmethod
    def set_final_rowscales(final_rowscales, online_rowscales, b, h, q_idx):
        """
        compute final_rowscales at the end of online attention forward
        """
        pass

    @staticmethod
    def scale_final_o(o, online_rowscales):
        """
        scale final o with final_rowscales
        """
        pass

    @staticmethod
    def forward(self, scores, final_rowscales:dict[str, SymbolScalar], b, h, q_idx):
        """
        compute scores : scores = g(scores, scale)
        """
        pass
    
    @staticmethod
    def backward(dscores, dp, scores, final_rowscales:dict[str, SymbolScalar], b, h, q_idx, kv_idx):
        """
        compute bwd scores: dscores = g_bwd(dp, scores)
        """
        dscores = dp
        pass

def create_block_mask(causal_mask, B, H, QLen, KVLen, device):
    pass

class AttentionEngine:
    def __init__(self, query, key, value, custom_fwd_inputs, custom_bwd_inputs, score_mod, block_mask,
    online_func,):
        pass

    def __call__(self, *args, **kargs):
        
        o = torch.tensor(1)
        return o

    def backward(self, *args, **kargs):
        pass
        
if __name__ == "__main__":
    online = OnlineFunc({},{})
