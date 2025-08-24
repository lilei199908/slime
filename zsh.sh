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
  --node-ip-address=0.0.0.0 \
  --num-gpus=8 \
  --include-dashboard=true \
  --dashboard-host=0.0.0.0

ray start --address=10.249.32.139:6379 --num-gpus=8

docker run --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v /data1:/data1 \
  --network host \
  --name sglang \
  -it pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

zhuzilin/slime:latest

nvcr.io/nvidia/pytorch:25.02-py3

hebiaobuaa/verl:app-verl0.5-sglang0.4.9.post6-mcore0.12.2-te2.2

nvcr.io/nvidia/cuda:12.4.0-devel-ubuntu22.04