# Installation

## 1

docker run -it --gpus all --name cfy-tl --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -e http_proxy=http://172.17.0.1:11237 -e https_proxy=http://172.17.0.1:11237 -v /mnt/disk1/cfy:/cfy nvcr.io/nvidia/pytorch:25.02-py3

pip install tilelang==0.1.5 # pip install tilelang==0.1.6.post2(error: picking error)

export PYTHONPATH="$(pwd)/attention_engine:$PYTHONPATH"
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so



## 2

# docker run -it --gpus all --name cfy-tl --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -e http_proxy=http://172.17.0.1:11237 -e https_proxy=http://172.17.0.1:11237 -v /mnt/disk1/cfy:/cfy nvcr.io/nvidia/pytorch:25.02-py3

pip install . # need wait for 30 minutes for git clone # pip install tilelang==0.1.4 # pip install tilelang==0.1.6.post2

export PYTHONPATH="$(pwd)/attention_engine:$PYTHONPATH"
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so



# benchmark dependencies

pip install triton==3.2.0
pip install mamba-ssm==2.2.6.post3 (may take 20 minutes for compilation)

mamba2 第一次编译 dh 报错，第二次编译就好了；

pip install transformers==4.45.0
pip install flash-linear-attention==0.2.0 # pip install flash-linear-attention==0.4.0 (no head first mode)

git clone https://github.com/apple/ml-sigmoid-attention.git
cd ml-sigmoid-attention/flash_sigmoid
MAX_JOBS=8 python3 setup.py install


tilelang tune 似乎有bug（对于illegal init para）
/cfy/tilelang_new/tilelang/cache/kernel_cache.py先 disable cache解决了