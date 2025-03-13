from typing import List, Dict, Any, Tuple, Union
from sympy import Symbol
import sympy as sp
from ..core import IndentedCode

def dtype_map(dtype:str)->str:
    if "dtype" in dtype:
        return dtype
    return f"\'{dtype}\'"

def arg_def(arg)->str:
    return f"{arg.name}: T.Buffer({arg.shape}, {dtype_map(arg.dtype)}),"


def alloc_op(mem_type:str, arg)->str:
    return f"{arg.name} = T.alloc_{mem_type}({arg.shape}, {dtype_map(arg.dtype)})"

def alloc_fragment_op(arg)->str:
    return alloc_op("fragment", arg)

def alloc_shared_op(arg)->str:
    return alloc_op("shared", arg)

def fill_op(arg, value:str)->str:
    if "inf" in value:
        value = value.replace("inf", f"T.infinity({dtype_map(arg.dtype)})")
    return f"T.fill({arg.name}, {value})"

def call_op(func_name:str, args:List)->str:
    return f"{func_name}({', '.join([arg.name for arg in args])})"

def load_op(src, dst, src_dim_list:List[int], src_idx_list:List[Symbol], src_step_list:List[Symbol])->str:
    assert(len(src_dim_list) == len(src_idx_list) and len(src_dim_list) == len(src_step_list))
    src_copy_list = [":" for _ in range(len(src_dim_list))]
    for dim in src_dim_list:
        start_idx = src_idx_list[dim]
        end_idx = src_idx_list[dim] + src_step_list[dim]
        if src_step_list[dim] == 1:
            src_copy_list[dim] = f"{start_idx}"
        else:
            src_copy_list[dim] = f"{start_idx}:{end_idx}"
            
    return f"T.copy({src.name}[{', '.join(src_copy_list)}], {dst.name})"

def store_op(src, dst, dst_dim_list:List[int], dst_idx_list:List[Symbol], dst_step_list:List[Symbol])->str:
    assert(len(dst_dim_list) == len(dst_idx_list) and len(dst_dim_list) == len(dst_step_list))
    dst_copy_list = [":" for _ in range(len(dst_dim_list))]
    for dim in dst_dim_list:
        start_idx = dst_idx_list[dim]
        end_idx = dst_idx_list[dim] + dst_step_list[dim]
        print(dst_idx_list[dim], dst_step_list[dim])
        if dst_step_list[dim] == 1:
            dst_copy_list[dim] = f"{start_idx}"
        else:
            dst_copy_list[dim] = f"{start_idx}:{end_idx}"
            
    return f"T.copy({src.name}, {dst.name}[{', '.join(dst_copy_list)}])"

def copy_op(src, dst)->str:
    return f"T.copy({src.name}, {dst.name})"

def parallel_for_block(iter_size:List[str], iter_vars:List[str],body:Union[IndentedCode,str]) -> IndentedCode:
    for_code = IndentedCode()
    
    for_code.add_line(f"for {', '.join([var for var in iter_vars])} in T.Parallel({', '.join([var for var in iter_size])}):")
    for_code.more_indent()
    for_code += body
    for_code.less_indent()
    return for_code
    
    
def func_def_block(func_name:str, args:List) -> IndentedCode:
    def_code = IndentedCode()
    def_code.add_line(f"def {func_name}(")
    for arg in args:
        def_code.add_line(f"{arg_def(arg)},")
    def_code.less_indent()
    def_code.add_line("):")
    return def_code

def func_block(func_name:str, func_args:List, func_body:Union[IndentedCode,str]) -> IndentedCode:
    func_code = IndentedCode()
    func_code += func_def_block(func_name, func_args)
    func_code.more_indent()
    func_code += func_body
    if str(func_body) == "":
        func_code.add_line("pass")
    func_code.less_indent()
    return func_code


    