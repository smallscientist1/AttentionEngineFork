# Installation

## 1

docker run -it --gpus all --name cfy-tl --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -e http_proxy=http://172.17.0.1:11237 -e https_proxy=http://172.17.0.1:11237 -v /mnt/disk1/cfy:/cfy nvcr.io/nvidia/pytorch:25.02-py3

install tilelang==0.1.5 from source with llvm backend # pip install tilelang==0.1.5 # pip install tilelang==0.1.6.post2(error: picking error)

<!-- apt update
apt install libtinfo5 -->
USE_LLVM=True pip install -v -e .

export PYTHONPATH="$(pwd)/attention_engine:$PYTHONPATH"
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so



## 2

# docker run -it --gpus all --name cfy-tl --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -e http_proxy=http://172.17.0.1:11237 -e https_proxy=http://172.17.0.1:11237 -v /mnt/disk1/cfy:/cfy nvcr.io/nvidia/pytorch:25.02-py3

pip install . # need wait for 30 minutes for git clone # pip install tilelang==0.1.4 # pip install tilelang==0.1.6.post2

export PYTHONPATH="$(pwd)/attention_engine:$PYTHONPATH"
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so



# benchmark dependencies

pip install triton==3.2.0
pip install mamba-ssm==2.2.6.post3 (may take 20 minutes for compilation) # /usr/local/lib/python3.12/dist-packages/selective_scan_cuda.cpython-312-x86_64-linux-gnu.so: 

undefined symbol: _ZN3c104cuda9SetDeviceEab

在 mamba-ssm/__init__.py 中 注释掉
```py
# from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
# from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.mamba2 import Mamba2
# from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
```

mamba2 第一次编译 dh 报错，第二次编译就好了；

pip install transformers==4.45.0
pip install flash-linear-attention==0.2.0 # pip install flash-linear-attention==0.4.0 (no head first mode)

git clone https://github.com/apple/ml-sigmoid-attention.git
cd ml-sigmoid-attention/flash_sigmoid
MAX_JOBS=8 python3 setup.py install


tilelang tune 似乎有bug（对于illegal init para）
/cfy/tilelang_new/tilelang/cache/kernel_cache.py先 disable cache解决了

/usr/local/lib/python3.12/dist-packages/tilelang/src/tl_templates/cuda/common.h 添加fasttanh

sigmoid attn bwd : Error: Failed to initialize the TMA descriptor K_desc
（当GPU内存占用大时，以及刚刚autotune完时）

fix bug: Error: b_k ['BT', 'BK'] dt ['batch', 'heads', 'seq_len']

tune retention_linear/mamba时有时会 error，需要重跑（前一次tune的kernel有问题？）

git clone https://github.com/deepseek-ai/FlashMLA.git 
git submodule update --init --recursive
pip install -v . (several minutes)

## install fa3
git clone https://github.com/Dao-AILab/flash-attention.git 
checkout to 2.8.3
cd hopper
FLASH_ATTENTION_DISABLE_PAGEDKV=TRUE FLASH_ATTENTION_DISABLE_APPENDKV=TRUE FLASH_ATTENTION_DISABLE_LOCAL=TRUE FLASH_ATTENTION_DISABLE_SOFTCAP=TRUE FLASH_ATTENTION_DISABLE_PACKGQA=TRUE FLASH_ATTENTION_DISABLE_FP8=TRUE FLASH_ATTENTION_DISABLE_VARLEN=TRUE FLASH_ATTENTION_DISABLE_SM80=TRUE python setup.py install


# 硬件要求

约 150g 内存用于bench infer mask

# TODO
more baseline;
性能fix on H100
    - retnet bwd, sigmoid bwd(是因为bwd 1时batch很小，容易gg，多跑几次就好了)
    - rfa big(速度出现变化), yoco(太快了), 
精度fix mamba on H100
sparse gqa
fix lower error
cutlass path for cute v2

test mla_decode_v1

output more detailed log for tune


# AMD mi250

flash attention 2.8 much faster, identify why and try boost tilelang performance
mamba head 80, slower than mamba2
mamba head 8, also slower than mamba2(baseline become fast)

reluattn bwd, torch faster on small batch (maybe do_bench 500,1000?)

amd tune bug: attn 128,256 fwd 偶尔一次出现autotune成功，但compile不成功的情况
amd test.py mamba2出现一次精度问题
