#!/usr/bin/env python3

"""
This document is created by magic at 2018/8/20
"""


class BinaryHeap:
    """
    二叉堆实现
    """

    def __init__(self):
        self.heap_list = []
        self.current_size = 0

    def perc_up(self, i):
        while i // 2 > 0:
            if self.heap_list[i] < self.heap_list[i // 2]:
                tmp = self.heap_list[i // 2]
                self.heap_list[i // 2] = self.heap_list[i]
                self.heap_list[i] = tmp

            i = i // 2

    def insert(self, k):
        self.heap_list.append(k)
        self.current_size = self.current_size + 1

        self.perc_up(self.current_size)

    def per_down(self, i):
        while (i * 2) <= self.current_size:
            mc = self.min(i)
            if self.heap_list[i] > self.heap_list[mc]:
                tmp = self.heap_list[i]
                self.heap_list[i] = self.heap_list[mc]
                self.heap_list[mc] = tmp
            i = mc

    def min(self, i):
        if i * 2 + 1 > self.current_size:
            return i * 2
        else:
            if self.heap_list[i * 2] < self.heap_list[i * 2 + 1]:
                return i * 2
            else:
                return i * 2 + 1

    def del_min(self):
        retval = self.heap_list[1]
        self.heap_list[1] = self.heap_list[self.current_size]
        self.current_size = self.current_size - 1
        self.heap_list.pop()
        self.per_down(1)
        return retval

    def build_heap(self, values):
        i = len(values) // 2
        self.current_size = len(values)
        self.heap_list = [0] + values
        while i > 0:
            self.per_down(i)
            i -= 1


if __name__ == '__main__':
    bh = BinaryHeap()

    v = [8, 4, 2, 1, 6, 7, 9, 11]
    bh.build_heap(v)

    for i in range(len(v) - 1):
        print(bh.del_min())
