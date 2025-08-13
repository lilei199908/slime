#!/bin/bash
echo "开始安装 zsh..."
apt-get update
apt-get install -y zsh

zsh
sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

echo "install themes"
git clone --depth=1 https://github.com/romkatv/powerlevel10k.git ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/themes/powerlevel10k
sed -i 's/ZSH_THEME=".*"/ZSH_THEME="powerlevel10k\/powerlevel10k"/' ~/.zshrc
cp /data1/lilei/.p10k.zsh ~/.p10k.zsh
echo 'export POWERLEVEL9K_DISABLE_CONFIGURATION_WIZARD=true' >> ~/.zshrc
source ~/.zshrc

git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions
git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting
sed -i '/^plugins=/,/^)/ d' ~/.zshrc  # 删除 plugins=( ... ) 整个块
# 然后追加新的 plugins 定义
cat >> ~/.zshrc << 'EOF'
plugins=(
    git                    # 内置 git 插件
    zsh-autosuggestions    # 命令自动建议（灰色提示）
    zsh-syntax-highlighting # 命令语法高亮（正确绿色，错误红色）
)
EOF

cat >> ~/.bashrc << 'EOF'

# 自动进入工作目录并启动 zsh
cd /data1/lilei
zsh
EOF

source ~/.zshrc
