#!/usr/bin/env python3

"""
This document is created by magic at 2018/8/16
"""


class BaseQueue:
    """
    利用python实现一个简单的队列
    """

    def __init__(self):
        self.items = []

    def enqueue(self, value):
        self.items.append(value)

    def dequeue(self):
        item = self.items[0]
        self.items = self.items[1:]
        return item

    @property
    def size(self):
        return len(self.items)

    @property
    def empty(self):
        return len(self.items) == 0


if __name__ == '__main__':
    q = BaseQueue()
    q.enqueue('123')
    q.enqueue('bbb')
    q.enqueue(4)

    print(q.items)
    a = q.dequeue()
    print(a)

    print(q.size)
