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
export PYTHONPATH="$(pwd)/attention_engine:$(pwd)/3rdparties/tvm/python:$PYTHONPATH"
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so
```

# Roadmap
- [ ] Support backward on CuTe backend 
- [ ] Support decoding shape
- [ ] Support AMD MI250


## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
