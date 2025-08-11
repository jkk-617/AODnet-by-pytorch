import torch
import torch.nn as nn
import torch.optim as optim
from model import AODnet

# 检查CUDA
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")
print(f"Using device: {device}")

# 实例化模型并移动到GPU
model = AODnet().to(device)

# 随机生成输入和标签（实际使用时请替换为真实数据）
inputs = torch.randn(8, 3, 64, 64).to(device)  # batch_size=8, 3通道, 64x64
targets = torch.randn(8, 3, 64, 64).to(device)

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 训练循环
epochs = 5

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

print("训练结束")