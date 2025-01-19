# AttentionEngine

AttentionEngine is a unified framework to customized attention, including parallel attention and linear attention (For example, replacing softmax to sigmoid in standard attention, implementing retnet attention, and implementing mamba2).

# Tested Device
- NVIDIA H100
- AMD MI250 (TODO)

# Customized attention Examples

Customized attention examples are under folder `attn_script`, including:
+ parallel attention
    - `attn_script/mha.py`: softmax attention
    - `attn_script/sigmoidattn.py`: sigmoid attention
    - `attn_script/reluattn.py`: relu attention
    - `attn_script/retention.py`: retnet attention
+ linear attention
    - `attn_script/mamba2_ngroup1.py`: mamba2
    - `attn_script/simple_gla.py`: gated retention
    - `attn_script/retnetion_linear.py`: retnet linear

# Installation
- install cuda==12.4 & pytorch
- clone the repo and its submodule
```
git clone --recursive https://github.com/smallscientist1/AttentionEngine.git
```
- install tvm
```
cd 3rdparties/tvm
mkdir -p build && cd build && cp ../cmake/config.cmake . && cmake .. && make -j && cd -
cd ../../
```
- export some environment variables
```
export PYTHONPATH="$(pwd)/python:$(pwd)/3rdparties/tvm/python:$PYTHONPATH"
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so
```
