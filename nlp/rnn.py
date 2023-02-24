#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
# import torch.nn as nn

# # 创建一个RNN实例
# rnn = nn.RNN(10, 20, 1, batch_first=True) 

# input = torch.randn(5, 3, 10)
# h0 = torch.randn(1, 5, 20)

# output, hn = rnn(input, h0)
# print(output)
# print(hn)

A = torch.ones(3, 3)
B = 2 * torch.ones(3, 3)

print(A)

print(B)

C = torch.cat((A, B), 1)
print(C)



