#!/usr/bin/env python3

# 有一个函数func1 能返回0和1 两个数值，返回0和1的概率都是1/2 怎么利用这个函数得到另一个函数func2
# 使func2也能返回0和1 且返回0的概率为1/4 返回1 的概率为3/4 

# random.random() 返回0，1 之间的随机数
import random


# 返回0， 1 的概率都为1/2
def func1():
    return int(round(random.random()))


# 返回0 1 的概率为1/4， 3/4

def func2():
    a1 = func1()
    a2 = func1()

    tmp = a1 | a2  # a1 & a2

    if tmp == 0:
        return 0
    else:
        return 1

if __name__ == '__main__':
    i = 0
    while i < 16:
        print(func2())
        i += 1
