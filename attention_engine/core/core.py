import torch
from typing import Literal, Type
import functools
from .graph import *
from .utils import IndentedCode
import functools
from copy import copy, deepcopy

# TODO: support online_func extern_input_tensor


def plus_count(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        self.count += 1
        for arg in args:
            if isinstance(arg, SymbolScalar):
                arg.count += 1
            # else:
            #     print(arg)
        for key, arg in kwargs.items():
            if isinstance(arg, SymbolScalar):
                arg.count += 1
            # else:
            #     print(arg)
        return func(self, *args, **kwargs)
    return wrapper


class SymbolScalar:
    def __init__(self, varname: str, value: Node, prev=[], shape_idx: list = ["block_M"],
                 require_grad: bool = True):
        self.varname = varname
        self.code = value
        self.prev = prev
        self.shape_idx = shape_idx
        self.require_grad = require_grad

        self.count = 0
        self.use_list = []
        # for lower
        self.lowered = False
        self.visit_count = 0

        self.grad = None  # SymbolicScalar
    
    def __repr__(self):
        return f"SymbolScalar({self.varname}, {self.code}, {self.prev}, {self.shape_idx}, {self.require_grad})"

    # @plus_count # TODO: plus count bug
    def op(self, code: Type[Node], others: list = [],
           shape_idx: list = None, varname_suffix: str = None):
        for i, other in enumerate(others):
            # if other is python scalar
            if isinstance(other, (int, float)):
                others[i] = SymbolicConst(other)
            else:
                assert isinstance(other, SymbolScalar)
        if shape_idx is None:
            shape_idx = self.shape_idx
        output_varname = self.varname
        if varname_suffix is not None:
            output_varname = f"{output_varname}_{varname_suffix}"

        code = code(*[x.code for x in [self] + others])
        # TODO: now must be var+1 not 1+var
        output = self.__class__(
            f"{output_varname}_{self.count}",
            code,
            [self] + others,
            shape_idx)
        self.use_list.append(output)
        self.count += 1
        # print(self.count)
        # print(len(self.use_list))
        # if self.count != len(self.use_list):
        #     print(self,self.varname)
        for other in others:
            other.use_list.append(output)
            other.count += 1
            # print(other.count)
            # print(len(other.use_list))
            # if other.count != len(other.use_list):
            #     print(other,other.varname)
        return output

    def backward(self, grad=None):  # SymbolicScalar
        if grad:
            self.grad = grad
        self._backward(self.grad)
        for node in self.prev:
            node.backward()

    def _backward(self, grad):  # symblocscalar
        if self.code.type == "Var" or self.code.type == "Const":
            return
        if self.code.type == "Add":
            grad_0 = copy(grad)
            grad_1 = copy(grad)
            if self.prev[0].require_grad:
                if self.prev[0].grad:
                    grad_0 = grad_0 + self.prev[0].grad
                self.prev[0].grad = grad_0
            if self.prev[1].require_grad:
                if self.prev[1].grad:
                    grad_1 = grad_1 + self.prev[1].grad
                self.prev[1].grad = grad_1
        elif self.code.type == "Mul":
            if self.prev[0].require_grad:
                grad0 = grad * self.prev[1]
                if self.prev[0].grad:
                    grad0 = grad0 + self.prev[0].grad
                self.prev[0].grad = grad0
            if self.prev[1].require_grad:
                grad1 = grad * self.prev[0]
                if self.prev[1].grad:
                    grad1 = grad1 + self.prev[1].grad
                self.prev[1].grad = grad1

        elif self.code.type == "Div":
            if self.prev[1].require_grad or self.prev[0].require_grad:
                grad0 = grad / self.prev[1]
                if self.prev[0].require_grad:
                    if self.prev[0].grad:
                        grad0 = grad0 + self.prev[0].grad
                    self.prev[0].grad = grad0
                if self.prev[1].require_grad:
                    grad1 = - grad0 * self.prev[0] / self.prev[1]
                    if self.prev[1].grad:
                        grad1 = grad1 + self.prev[1].grad
                    self.prev[1].grad = grad1

        elif self.code.type == "Tanh":
            if self.prev[0].require_grad:
                # grad_t = self * grad
                # grad_t = self * grad_t
                # grad0 = grad - grad_t
                grad0 = grad - grad * self * self
                if self.prev[0].grad:
                    grad0 = grad0 + self.prev[0].grad
                self.prev[0].grad = grad0
        elif self.code.type == "Max":
            # TODO: max backward,implement is then else lower
            if self.prev[0].require_grad:
                grad0 = grad * (self.prev[0] == self)
                if self.prev[0].grad:
                    grad0 = grad0 + self.prev[0].grad
                self.prev[0].grad = grad0
            if self.prev[1].require_grad:
                grad1 = grad * (self.prev[1] == self)
                if self.prev[1].grad:
                    grad1 = grad1 + self.prev[1].grad
                self.prev[1].grad = grad1

        elif self.code.type == "Log":
            if self.prev[0].require_grad:
                grad0 = grad / self.prev[0]
                if self.prev[0].grad:
                    grad0 = grad0 + self.prev[0].grad
                self.prev[0].grad = grad0
        else:
            raise NotImplementedError(
                f"backward for {self.code.type} is not implemented")
        # change shape_idx
        for idx, node in enumerate(self.prev):
            if node.require_grad:
                self.prev[idx].grad.shape_idx = self.prev[idx].shape_idx

    def clear_usecount(self):
        self.count = 0
        self.use_list.clear()

    def clear_visit(self):
        self.visit_count = 0
        self.lowered = False

    def clear_codegen(self):
        self.clear_usecount()
        self.clear_visit()

    def __add__(self, other):
        return self.op(Add, [other])

    def __neg__(self):
        return self.op(Neg)

    def __sub__(self, other):
        return self.op(Sub, [other])

    def __mul__(self, other):
        return self.op(Mul, [other])

    def __truediv__(self, other):
        return self.op(Div, [other])

    def abs(self):
        return self.op(Abs)

    def tanh(self):
        return self.op(Tanh)

    def exp(self):
        return self.op(Exp)

    def exp2(self):
        return self.op(Exp2)

    def log(self):
        return self.op(Log)

    def max(self, other):
        return self.op(Max, [other])


class SymbolicArray(SymbolScalar):
    """
    Array for OnlineFunc.online_fwd
    """

    def __init__(self, varname: str = "", code: Node = Var(" "),
                 prev=[], shape_idx: list = ["block_M", "block_N"]):
        super().__init__(varname, code, prev, shape_idx)

    def get_reduce(self, op: Literal["sum", "max", "abssum"]):
        """
        get reduce result of array
        """
        if op == "sum":
            return self.op(
                ReduceSum, shape_idx=self.shape_idx[:-1], varname_suffix="sum")
        elif op == "max":
            return self.op(
                ReduceMax, shape_idx=self.shape_idx[:-1], varname_suffix="max")
        elif op == "abssum":
            return self.op(
                ReduceAbsSum, shape_idx=self.shape_idx[:-1], varname_suffix="abssum")
        else:
            raise NotImplementedError


class SymbolicTensor(SymbolScalar):
    """
    Tensor for CustomIO
    """

    def __init__(self, varname: str, shape: tuple):
        # convert shape to shape_idx
        super().__init__(
            varname, Var(varname), shape_idx=[
                str(i) for i in shape])
        self.shape = shape


class SymbolicConst(SymbolScalar):
    """
    Const for constant value
    """

    def __init__(self, value):
        # TODO: not float const # str(value)+(".f" if isinstance(value, int)
        # else "f")
        super().__init__(f"float({str(value)})", Const(value), prev=[], shape_idx=[],
                         require_grad=False)


class CustomIO:
    def __init__(self, input_tensors: dict[str, tuple] = {}):
        self.input_tensors: dict[str, SymbolicTensor] = {}
        for k, v in input_tensors.items():
            self.input_tensors[k] = SymbolicTensor(k, v)

    def __call__(self, tensor_name: str, tensor_shape: tuple):
        if tensor_name in self.input_tensors:
            raise ValueError(f"Tensor {tensor_name} already exists")
        self.input_tensors[tensor_name].shape = tensor_shape

        def decorator(func):
            def wrapper(*args, **kwargs):
                result = func(*args, **kwargs)

                return result
            return wrapper
        return decorator


def create_block_mask(causal_mask, B, H, QLen, KVLen, device):
    return True
