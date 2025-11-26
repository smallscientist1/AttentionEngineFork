I have no name!@hacc-gpu1:~$ rocm-smi


========================================= ROCm System Management Interface =========================================
=================================================== Concise Info ===================================================
Device  Node  IDs              Temp    Power  Partitions          SCLK    MCLK     Fan  Perf  PwrCap  VRAM%  GPU%  
              (DID,     GUID)  (Edge)  (Avg)  (Mem, Compute, ID)                                                   
====================================================================================================================
0       3     0x740f,   21887  32.0°C  41.0W  N/A, N/A, 0         800Mhz  1600Mhz  0%   auto  300.0W  0%     0%    
1       2     0x740f,   1997   38.0°C  41.0W  N/A, N/A, 0         800Mhz  1600Mhz  0%   auto  300.0W  0%     0%    
2       5     0x740f,   22429  43.0°C  42.0W  N/A, N/A, 0         800Mhz  1600Mhz  0%   auto  300.0W  0%     0%    
3       4     0x740f,   32693  37.0°C  43.0W  N/A, N/A, 0         800Mhz  1600Mhz  0%   auto  300.0W  0%     0%    
====================================================================================================================
=============================================== End of ROCm SMI Log ================================================

I have no name!@hacc-gpu1:~$ rocm-smi --version
ROCM-SMI version: 3.0.0+94441cb
ROCM-SMI-LIB version: 7.4.0


singularity pull my_pytorch.sif docker://rocm/pytorch:rocm6.3.2_ubuntu22.04_py3.10_pytorch_release_2.4.0

srun -p mi210_u280_u55c --cpus-per-task=32 --pty bash -i

<!-- 
singularity shell --rocm my_torch.sif

singularity instance start --rocm my_torch.sif my_worker
singularity instance list
singularity shell instance://my_worker
singularity instance stop my_worker -->


singularity exec --rocm --bind /data/chenfeiyang:/working_dir --pwd /working_dir /data/chenfeiyang/my_torch.sif bash # 
singularity exec --rocm --no-home --bind /data/chenfeiyang:/working_dir --pwd /working_dir /data/chenfeiyang/my_torch.sif bash
git clone https://github.com/tile-ai/tilelang.git

cd tilelang

# 3. 检出指定 Tag
git checkout v0.1.5

git config --global --add safe.directory '*'
USE_LLVM=True USE_ROCM=True pip install -v -e . # USE_ROCM=True pip install -v -e .  # may take 20 minutes
USE_ROCM=True pip wheel . --no-deps -w ./dist
pip install "numpy<2.0"

export PYTHONPATH="$(pwd)/attention_engine:$PYTHONPATH"
# export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so

git checkout ppopp_AE
pip install einops matplotlib pandas

pip install flash-attn==2.7.4.post1 --no-build-isolation # pip install flash-attn==2.8.3 --no-build-isolation # may take 60 minutes
# pip install mamba-ssm==2.2.6.post3  # may take 20 minutes for compilation

git clone https://github.com/fla-org/flash-linear-attention.git 
git checkout v0.2.0
mv fla/__init__.py fla/__init__.py.bak
mv fla/ops/__init__.py fla/ops/__init__.py.bak
export PYTHONPATH="$(pwd):$PYTHONPATH"

git clone https://github.com/state-spaces/mamba.git
git checkout v2.2.6.post3
mv mamba_ssm/__init__.py mamba_ssm/__init__.py.bak
export PYTHONPATH="$(pwd):$PYTHONPATH"


srun --jobid=65556 --overlap --pty bash

sbatch run_sleep.sh # srun -p mi210_u280_u55c --cpus-per-task=32 python sleep.py 5

sbatch run_singularity.sh # singularity exec --rocm --bind /data/chenfeiyang:/working_dir --pwd /working_dir /data/chenfeiyang/my_torch.sif python /home/chenfeiyang/sleep.py 5

"""
#!/bin/bash
#SBATCH -p mi210_u280_u55c        # 分区名称
#SBATCH --cpus-per-task=32        # 申请 32 个 CPU
#SBATCH --job-name=singularity_job # 任务名称
#SBATCH --output=train_%j.log     # 输出日志 (%j 代表任务ID)
#SBATCH --error=train_%j.err      # 错误日志

# 执行 Singularity 命令
# 注意：这里直接运行 singularity，无需再加 srun，因为 sbatch 已经分配了资源
singularity exec --rocm \
    --bind /data/chenfeiyang:/working_dir \
    --pwd /working_dir \
    /data/chenfeiyang/my_torch.sif \
    python /home/chenfeiyang/sleep.py 5
"""

scancel -u chenfeiyang

srun --jobid=65557 --pty --overlap bash
singularity shell --rocm --bind /data/chenfeiyang:/working_dir /data/chenfeiyang/my_torch.sif

singularity shell --rocm --fakeroot --no-home --bind /data/chenfeiyang/AttentionEngineFork:/AttentionEngine /data/chenfeiyang/my-image.sif

