from .core import SymbolScalar, SymbolicArray, CustomIO
from .graph import Var, Const
from .utils import IndentedCode
from typing import Tuple

def to_tl_op(type:str, *args:SymbolScalar):
    code = IndentedCode()
    if type == "ReduceSum":
        code.add_line(
            f"T.reduce_sum({args[1].varname}, {args[0].varname},dim=1, clear=True)"
        )
    elif type == "ReduceMax":
        code.add_line(
            f"T.reduce_max({args[1].varname}, {args[0].varname},dim=1, clear=True)"
        )
    elif type == "Sub" or type == "Add" or type == "Mul" or type == "Div" or type == "Neg" or type == "Exp" or type == "Log" or type == "Abs" or type == "Max" or type == "Tanh":
        # args idx
        # note: assume input shape is validate: ["1",...] or [arg0[0], ...]
        idx_strs = []
        for _, arg in enumerate(args):
            input_idx = arg.shape_idx
            idx_str = [f"i{i}" if idx!="1" else f"0" for i, idx in enumerate(input_idx)]
            idx_str = ",".join(idx_str)
            idx_strs.append(idx_str)
        # [block_M,block_N]
        loop_str = ",".join(args[0].shape_idx)
        idx_str = idx_strs[0]

        # for loop
        code.add_line(
            f"for {idx_str} in T.Parallel({loop_str}):"
        )
        code.more_indent()

        # remove [] for scalar
        for i, tmp_idx_str in enumerate(idx_strs):
            idx_strs[i] = f"[{tmp_idx_str}]" if len(tmp_idx_str) > 0 else tmp_idx_str

        if type == "Sub":
            code.add_line(
                f"{args[0].varname}[{idx_str}] = {args[1].varname}{idx_strs[1]} - {args[2].varname}{idx_strs[2]}"
            )
        elif type == "Add":
            code.add_line(
                f"{args[0].varname}[{idx_str}] = {args[1].varname}{idx_strs[1]} + {args[2].varname}{idx_strs[2]}"
            )
        elif type == "Max":
            code.add_line(
                f"{args[0].varname}[{idx_str}] = T.max({args[1].varname}{idx_strs[1]}, {args[2].varname}{idx_strs[2]})"
            )
        elif type == "Exp":
            code.add_line(
                f"{args[0].varname}[{idx_str}] = T.exp2({args[1].varname}{idx_strs[1]}*1.442695)"
            )
        elif type == "Mul":
            code.add_line(
                f"{args[0].varname}[{idx_str}] = {args[1].varname}{idx_strs[1]} * {args[2].varname}{idx_strs[2]}"
            )
        elif type == "Div":
            code.add_line(
                f"{args[0].varname}[{idx_str}] = {args[1].varname}{idx_strs[1]} / {args[2].varname}{idx_strs[2]}"
            )
        elif type == "Log":
            code.add_line(
                f"{args[0].varname}[{idx_str}] = T.log2({args[1].varname}{idx_strs[1]}) * 0.69314718"
            )
        elif type == "Tanh":
            code.add_line(
                f"{args[0].varname}[{idx_str}] = T.tanh({args[1].varname}{idx_strs[1]})"
            )
        else: # TODO
            raise NotImplementedError(str(type))
        
        code.less_indent()
    else:
        raise NotImplementedError
    return code

def to_pytorch_op(type:str, *args:SymbolScalar):
    code = IndentedCode()
    if type == "ReduceSum":
        code.add_line(
            f"{args[0].varname} = torch.sum({args[1].varname}, dim=-1)"
        )
    elif type == "ReduceMax":
        code.add_line(
            f"{args[0].varname} = torch.max({args[1].varname}, dim=-1)"
        )
    elif type == "Sub" or type == "Add" or type == "Mul" or type == "Div" or type == "Neg" or type == "Exp" or type == "Log" or type == "Abs" or type == "Max":
        # args idx
        # assume outputn dim max
        output_idx = args[0].shape_idx
        argnames = [arg.varname for arg in args]

        for i, arg in enumerate(args):
            assert len(arg.shape_idx) <= len(output_idx)
            if len(arg.shape_idx) < len(output_idx):
                # a -> a[...,None]
                argnames[i] = f"{arg.varname}[..." + ",None"*(len(output_idx)-len(arg.shape_idx)) + "]"

        if type == "Sub":
            code.add_line(
                f"{argnames[0]} = {argnames[1]} - {argnames[2]}"
            )
        elif type == "Add":
            code.add_line(
                f"{argnames[0]} = {argnames[1]} + {argnames[2]}"
            )
        elif type == "Max":
            code.add_line(
                f"{argnames[0]} = torch.maximum({argnames[1]}, {argnames[2]})"
            )
        elif type == "Exp":
            code.add_line(
                f"{argnames[0]} = torch.exp({argnames[1]})"
            )
        elif type == "Mul":
            code.add_line(
                f"{argnames[0]} = {argnames[1]} * {argnames[2]}"
            )
        elif type == "Div":
            code.add_line(
                f"{argnames[0]} = {argnames[1]} / {argnames[2]}"
            )
        elif type == "Log":
            code.add_line(
                f"{argnames[0]} = torch.log({argnames[1]})"
            )
        else: # TODO
            raise NotImplementedError(str(type))
    else:
        raise NotImplementedError
    return code

def generate_tl_from_dag(x_list:list[SymbolScalar], to_tl:bool=True) -> Tuple[IndentedCode, dict]:
    # global var
    input_vars = {}
    def generate_tl(x:SymbolScalar, varname:str=None):
        tl_code = IndentedCode()
        if x.lowered:
            return tl_code
        if isinstance(x.code, Var):
            # tl_code.add_line(f"{x.varname} = {x.code.name}")
            # add input_var
            input_vars[x.varname] = x
            return tl_code
        if isinstance(x.code, Const):
            return tl_code
        # for i, input_item in enumerate(x.code.inputs):
        #     tl_code += generate_tl(SymbolScalar(f"{x.varname}_i{i}", input_item))
        for i, input_item in enumerate(x.prev):
            tl_code += generate_tl(input_item)
            # if input_item.varname == "dsT_0":
            #     print("dsT_0::",input_item.varname)
            #     print("x:", x.varname)
            #     print(input_item.visit_count)
        # all previous node be generated before visit_count+1
        for i, input_item in enumerate(x.prev):
            input_item.visit_count += 1

        if varname is not None: # overwrite varname
            x.varname = varname
        # optimize tl performance by inplace operation
        # print(x.varname)
        # print([x.varname for x in x.prev])
        # print("count:",[x.count for x in x.prev])
        # print("visit:",[x.visit_count for x in x.prev])
        # print("use_list:",[[usea.varname for usea in x.use_list] for x in x.prev])
        for i, input_item in enumerate(x.prev):
            if input_item.shape_idx == x.shape_idx:
                if input_item.count == 1 or input_item.visit_count == input_item.count:
                    x.varname = input_item.varname
                    break
        # add input_var
        if x.varname not in input_vars.keys():
            input_vars[x.varname] = x
        # tl_code += to_tl_op(x.code.type, x, *[SymbolScalar(f"{x.varname}_i{i}", input_item) for i, input_item in enumerate(x.code.inputs)])
        if to_tl:
            tl_code += to_tl_op(x.code.type, x, *x.prev)
        else:
            tl_code += to_pytorch_op(x.code.type, x, *x.prev)
        x.lowered = True
        return tl_code
    
    tl_code = IndentedCode()
    for x in x_list:
        tl_code += generate_tl(x)
    return tl_code, input_vars

