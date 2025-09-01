#!/bin/bash
echo "开始安装 zsh..."
apt-get update
apt-get install -y zsh

zsh
sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"


git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions
git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting
# 然后追加新的 plugins 定义
plugins=(
    git                    # 内置 git 插件
    zsh-autosuggestions    # 命令自动建议（灰色提示）
    zsh-syntax-highlighting # 命令语法高亮（正确绿色，错误红色）
)
source ~/.zshrc

ray start --head \
  --node-ip-address=10.249.32.139 \
  --port=6380 \
  --object-manager-port=8076 \
  --node-manager-port=8077 \
  --num-gpus=8 \
  --include-dashboard=true \
  --dashboard-host=0.0.0.0 \
  --dashboard-port=8266 \
  --disable-usage-stats \
  --dashboard-agent-listen-port=52380

ray start --address=10.249.32.139:6380 --num-gpus=8

sudo docker run --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --cap-add=SYS_PTRACE \
  -v /data1:/data1 \
  --network host \
  --name slime \
  -it zhuzilin/slime:latest

zhuzilin/slime:latest

nvcr.io/nvidia/pytorch:25.02-py3

hebiaobuaa/verl:app-verl0.5-sglang0.4.9.post6-mcore0.12.2-te2.2

nvcr.io/nvidia/cuda:12.4.0-devel-ubuntu22.04

export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.3/compilers/bin:$PATH
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.3/compilers/lib:$LD_LIBRARY_PATH
export MANPATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.3/compilers/man:$MANPATH
#export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/comm_libs/mpi/bin:$PATH
export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.3/comm_libs/openmpi4/bin:$PATH
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.3/comm_libs/openmpi4/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/data1/lilei/sglang/fftw/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.3/compilers/extras/qd/lib:$LD_LIBRARY_PATH
export PATH=/data1/lilei/sglang/vasp.6.4.3/bin:$PATH

