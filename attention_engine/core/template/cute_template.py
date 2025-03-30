import jinja2
import os
import os.path as osp


TEMPLATE_DIR = osp.join(osp.dirname(osp.abspath(__file__)), 'cute_template')
OUTPUT_DIR = osp.join(
    osp.dirname(
        osp.abspath(__file__)),
    'cute_template_output')
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# with open('/home/aiscuser/cfy/flash-attention-hopper-template/template/online_func.h', 'r') as f:
#     cute_template = f.read()
# CUTE_ONLINE_FUNC = cute_template


class CuteAttnTemplate:
    def __init__(self,
                 **kwargs):
        for root, dirs, files in os.walk(TEMPLATE_DIR):
            for file in files:
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    template1 = f.read()
                self.render_code(template1, **kwargs)
                with open(os.path.join(OUTPUT_DIR, file), 'w') as f:
                    f.write(self.tlcode)

    def render_code(self, temp_code, **kwargs):
        template = jinja2.Template(
            temp_code
        )
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        self.tlcode = template.render(**kwargs)

    def __call__(self):
        return self.tlcode
