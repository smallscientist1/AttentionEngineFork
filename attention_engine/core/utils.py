import torch
class IndentedCode:
    def __init__(self, indent=0):
        
        self.indent = indent
        self.code = ""
    def __str__(self):
        return self.code
    def add_line(self, line):
        self.code += "    "*self.indent + line + "\n"
    def __iadd__(self, other):
        assert(isinstance(other, IndentedCode))
        for line in other.code.split("\n"):
            if len(line) > 0:
                self.add_line(line)
        self.indent += other.indent
        return self
    def more_indent(self, indent=1):
        self.indent += indent
    def less_indent(self, indent=1):
        self.indent -= indent

# tensor subclass for meta analysis
def meta_tensor(*args, **kargs):
    return torch.empty(*args, **kargs, device="meta")
        