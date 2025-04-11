from core.lower.lower_linear import lower_tl


import importlib.util
import tempfile
import os
import hashlib


class LinearAttentionEngine:
    def __init__(self, qkv_meta, q_mod=None, k_mod=None, v_mod=None, decay_mod=None, custom_io=None,
                 tune=False, tune_filename="tune_result", tune_bwd=False):
        self._compile_tl(qkv_meta, q_mod, k_mod, v_mod, decay_mod, custom_io, tune=tune, tune_filename=tune_filename)
        if tune:
            B, HQ, S, DK = qkv_meta[0].shape
            DV = qkv_meta[2].shape[3]
            HK = qkv_meta[1].shape[1]
            H = qkv_meta[2].shape[1]

            # best_config, best_latency = self.autotune(
            #     B, HQ, HK, H, S, DK, DV, file_path=tune_filename)
            # if tune_bwd:
            #     best_config_bwd, best_latency_bwd = self.autotune_bwd(
            #         B, HQ, HK, H, S, DK, DV, file_path=tune_filename)
            #     best_config.update(best_config_bwd)
            # self._compile_tl(
            #     qkv_meta,
            #     q_mod,
            #     k_mod,
            #     v_mod,
            #     decay_mod,
            #     custom_io,
            #     best_config)

    def __call__(self, *args, **kargs):

        o = self.attention(*args, **kargs)
        return o

    def _compile_tl(self, qkv_meta, q_mod, k_mod, v_mod, decay_mod,
                    custom_io, tuned_config=None,
                    tune=False, tune_filename=""):
        tl_code = lower_tl(
            qkv_meta,
            q_mod,
            k_mod,
            v_mod,
            decay_mod,
            custom_io,
            tuned_config,
            tune=tune,
            tune_filename=tune_filename)
        self.tl_code = tl_code  # for debug
        # local_vars = {}
        # exec(tl_code, globals(), local_vars)
        # globals().update(local_vars)
        # self.attention = local_vars["attention"]
        code_hash = hashlib.md5(tl_code.encode()).hexdigest()
        cache_dir = os.path.join(os.path.dirname(__file__), "cache")
        file_path = os.path.join(cache_dir, f"{code_hash}.py")
        os.makedirs(cache_dir, exist_ok=True)
        if not os.path.exists(file_path):
            with open(file_path, "w") as f:
                f.write(tl_code)
                f.flush()
        # file_path = "/home/aiscuser/cfy/AttentionEngine/attn_script/retention_linear_tlcode1.py"
        spec = importlib.util.spec_from_file_location("tl_attn", file_path)
        tl_attn = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tl_attn)
        self.attention = tl_attn.linear_attention
        self.autotune = tl_attn.autotune
        if hasattr(tl_attn, "autotune_bwd"):
            self.autotune_bwd = tl_attn.autotune_bwd
