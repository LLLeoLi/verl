#!/bin/bash

# ============================================
# SSH Setup Script
# 1. 生成 RSA 密钥
# 2. 创建 SSH config
# 3. 显示公钥（手动添加到远程服务器）
# 4. 建立反向端口转发
# ============================================

SSH_DIR="$HOME/.ssh"
KEY_PATH="$SSH_DIR/id_rsa"
CONFIG_PATH="$SSH_DIR/config"

# 确保 .ssh 目录存在且权限正确
mkdir -p "$SSH_DIR"
chmod 700 "$SSH_DIR"

# --- Step 1: 生成 RSA 密钥 ---
if [ -f "$KEY_PATH" ]; then
    echo "⚠️  RSA 密钥已存在: $KEY_PATH，跳过生成。"
else
    echo "🔑 正在生成 RSA 密钥..."
    ssh-keygen -t rsa -b 4096 -f "$KEY_PATH" -N "" -C "lihao@hkust"
    echo "✅ 密钥生成完成。"
fi

# --- Step 2: 创建 SSH config ---
echo "📝 正在写入 SSH config..."

cat > "$CONFIG_PATH" <<'EOF'
Host Proxy-RAS
  HostName ras.cse.ust.hk
  User lihao
  ForwardAgent yes

Host HKUST-NLP
  HostName jxcpu1.cse.ust.hk
  User lihao
  ProxyJump Proxy-RAS
EOF

chmod 600 "$CONFIG_PATH"
echo "✅ SSH config 已写入: $CONFIG_PATH"

# --- Step 3: 显示公钥 ---
echo ""
echo "=========================================="
echo "📋 请复制以下公钥，添加到 HKUST-NLP 服务器的 ~/.ssh/authorized_keys 中："
echo "=========================================="
echo ""
cat "${KEY_PATH}.pub"
echo ""
echo "=========================================="

# --- Step 4: 等待用户确认并输入端口后建立反向端口转发 ---
echo ""
read -p "✅ 公钥添加完成后，按 Enter 键继续配置端口转发..." _

echo ""
read -p "请输入远程端口号 (默认 17029): " REMOTE_PORT
REMOTE_PORT=${REMOTE_PORT:-17029}

read -p "请输入本地端口号 (默认 7029): " LOCAL_PORT
LOCAL_PORT=${LOCAL_PORT:-7029}

echo ""
echo "🔗 正在建立反向端口转发 (远程 $REMOTE_PORT -> 本地 $LOCAL_PORT)..."
echo "   命令: ssh -v -N -R $REMOTE_PORT:127.0.0.1:$LOCAL_PORT HKUST-NLP"
echo ""

ssh -v -N -R "$REMOTE_PORT":127.0.0.1:"$LOCAL_PORT" HKUST-NLP