from .arch_base import Arch
from .A100 import *
from .RTX4090 import *
from .H100 import *
from .MI250 import *

AttnDevice = {
    (8,0): A100,
    (8,9): RTX4090,
    (9,0): H100,
}

AttnDeviceAMD = {
    (9,0): MI250,
}
