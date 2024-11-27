from .core import SymbolScalar, SymbolicArray, CustomIO
from .graph import Var, Const
from .utils import IndentedCode
from .tl_gen import generate_tl_from_dag
from .linear_attn_template import TlLinearAttnTemplate

from dataclasses import dataclass

@dataclass
class lowerKmodOutput:
    k_mod_expr: str
    # custom_inputs_list: str = ""

@dataclass
class lowerDecaymodOutput:
    decay_mod_expr: str

def lowerKmod(k_mod) -> lowerKmodOutput:
    k = SymbolicArray("k", Var("k"), shape_idx=["B", "H", "T", "D"])
    new_k = k_mod(k)
    pytorch_code, input_vars = generate_tl_from_dag([new_k], to_tl=False)
    k_mod_expr = str(pytorch_code)

    # custom_inputs_list = ", ".join([f"{varname}={varname}" for varname in input_vars.keys()])
    # custom_inputs_list += ","
    return lowerKmodOutput(k_mod_expr=k_mod_expr)# , custom_inputs_list=custom_inputs_list)

def lowerDecaymod(decay_mod) -> lowerDecaymodOutput:
    decay = SymbolicArray("decay", Var("decay"), shape_idx=["B", "H", "T"])
    new_decay = decay_mod(decay)
    pytorch_code, input_vars = generate_tl_from_dag([new_decay], to_tl=False)
    decay_mod_expr = str(pytorch_code)
    return lowerDecaymodOutput(decay_mod_expr=decay_mod_expr)

def lower_tl(q_mod, k_mod, v_mod, decay_mod, custom_io):
    if k_mod:
        lower_kmod_output = lowerKmod(k_mod)
    if decay_mod:
        lower_decaymod_output = lowerDecaymod(decay_mod)
    custom_inputs_list = ", ".join([f"{varname}" for varname in custom_io.input_tensors.keys()])
    custom_inputs_list += ","
    return TlLinearAttnTemplate(
        **lower_kmod_output.__dict__,
        **lower_decaymod_output.__dict__,
        custom_inputs_list=custom_inputs_list
    )()


