# Installation

## 1

docker run -it --gpus all --name cfy-tl --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -e http_proxy=http://172.17.0.1:11237 -e https_proxy=http://172.17.0.1:11237 -v /mnt/disk1/cfy:/cfy nvcr.io/nvidia/pytorch:25.02-py3

pip install tilelang==0.1.5 # pip install tilelang==0.1.6.post2

export PYTHONPATH="$(pwd)/attention_engine:$PYTHONPATH"
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so



## 2

# docker run -it --gpus all --name cfy-tl --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -e http_proxy=http://172.17.0.1:11237 -e https_proxy=http://172.17.0.1:11237 -v /mnt/disk1/cfy:/cfy nvcr.io/nvidia/pytorch:25.02-py3

pip install . # need wait for 30 minutes for git clone # pip install tilelang==0.1.4 # pip install tilelang==0.1.6.post2

export PYTHONPATH="$(pwd)/attention_engine:$PYTHONPATH"
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so
