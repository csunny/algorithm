#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn


# 定义模型
class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)


    def forward(self, x):
        return self.linear(x)


# 初始化模型
model = LinearRegression(input_size=1, output_size=1)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 创建输入特征x和标签y
x = torch.Tensor([[1], [2], [3], [4]])
y = torch.Tensor([[2], [4], [6], [8]])

# 训练模型
for epoch in range(100):
    predictions = model(x)
    loss = criterion(predictions, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

x_test = torch.Tensor([[5], [6], [7], [8]])
y_test = torch.Tensor([[10], [12], [14], [16]])

# 测试模型
with torch.no_grad():
    predictions = model(x_test)
    loss = criterion(predictions, y_test)
    print(f'Test loss: {loss:.4f}')