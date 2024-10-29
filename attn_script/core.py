import torch

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
    def __init__(self, online_rowscales:dict, final_rowscales:dict):
        """
        define&init online_rowscales and final_rowscales
        """
        self.online_rowscales = online_rowscales
        self.final_rowscales = final_rowscales
    
    @staticmethod
    def online_fwd(scores:Array,online_rowscales, b, h, q_idx):
        """
        input: 
            scores: 一维向量, 仅包含getreduce()
            online_rowscales: 保存在线算法的中间结果

        return: 
            o_scale: for online rescale o
        """
        pass
        return o_scale
    
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
    def forward(self, scores, final_rowscales, b, h, q_idx):
        """
        compute scores : scores = g(scores, scale)
        """
        pass
    
    @staticmethod
    def backward(dp:scalar, scores:scalar, final_rowscales, b, h, q_idx, kv_idx):
        """
        compute bwd scores: dscores = g_bwd(dp, scores)
        """
        pass

        return dscores

