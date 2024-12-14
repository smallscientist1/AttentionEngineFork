import jinja2
import os
import os.path as osp

TEMPLATE_DIR = osp.join(osp.dirname(osp.abspath(__file__)) , 'tl_template/attn/attn_tl.py')
with open(TEMPLATE_DIR, 'r') as f:
    TL_KERNEL = f.read()

class TlAttnTemplate:
    def __init__(self, 
                **kargs
                    ):

        template = jinja2.Template(TL_KERNEL)
        
        # remove None
        kargs = {k:(v if v is not None else "") for k,v in kargs.items() }
        self.tlcode = template.render(**kargs)
    def __call__(self):
        return self.tlcode

if __name__ == "__main__":
    tl_code = TlAttnTemplate()()
    print(tl_code)
    exec(tl_code)
    print(attention)
