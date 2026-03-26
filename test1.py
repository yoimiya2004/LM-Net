# test_den.py
import torch
import sys
# 确保能 import repo 内的 basicsr 包（当前目录）
sys.path.append('.')

from basicsr.archs.detail_enhance_net import DENet

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)

    # 构造 DENet（与你之前脚本相同）
    net = DENet(gps=3, blocks=3).to(device)
    net.train()  # 以训练模式（确保能反向传播）

    # 随机输入：batch=2, 3 channels, 128x128
    x = torch.randn(2, 3, 128, 128, device=device, requires_grad=False)

    # 前向
    y = net(x)
    print("y.shape:", y.shape)

    # 简单损失并反向
    loss = y.sum()
    loss.backward()
    print("backward OK")

if __name__ == '__main__':
    main()