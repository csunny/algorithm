#!/usr/bin/env python3

"""
This document is created by magic at 2018/8/20
"""
# from mqueue.base import BaseQueue
# from stack.base import BaseStack

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


class BaseGraph:
    """
    构造一个无向图，并遍历
    """

    def __init__(self, graph):
        self.graph = graph

    def dfs(self, n):
        """
        从一个顶点出发，对图进行深度优先遍历。
        :return:
        """
        visited = dict()
        s = BaseStack()
        s.push(n)

        while not s.empty:
            node = s.pop()

            if not visited.get(node):
                visited[node] = True

                print("dfs visiting ...", node)
                near = self.graph[node]
                for i in near:
                    if not visited.get(i):
                        s.push(i)

    def bfs(self, n):
        """
        从一个顶点出发对图进行广度优先遍历
        :param n:
        :return:
        """
        visited = dict()
        q = BaseQueue()
        q.enqueue(n)

        while not q.empty:
            node = q.dequeue()

            if not visited.get(node):
                visited[node] = True
                print("bfs visiting ...", node)
                near = self.graph[node]
                for n in near:

                    if not visited.get(n):
                        q.enqueue(n)


if __name__ == '__main__':
    g = {
        "A": ["B", "C", "D"],
        "B": ["A", "E"],
        "C": ["A", "E"],
        "D": ["A"],
        "E": ["B", "C", "F"],
        "F": ["E"]
    }

    graph = BaseGraph(g)
    graph.bfs("A")
