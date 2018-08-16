#!/usr/bin/env python3

"""
This document is created by magic at 2018/8/16
"""


class BaseStack:
    """
    doc 利用python实现一个栈
    """

    def __init__(self):
        self.items = []

    def push(self, value):
        self.items.append(value)

    def pop(self):
        item = self.items[-1]
        self.items = self.items[0:self.size - 1]
        return item

    @property
    def size(self):
        return len(self.items)

    @property
    def empty(self):
        return len(self.items) == 0


if __name__ == '__main__':
    s = BaseStack()
    s.push('123')
    s.push("magic")
    s.push(5)

    print(s.items)

    print(s.pop())
    print(s.pop())
    print(s.size)
    print(s.pop())
    print(s.empty)
