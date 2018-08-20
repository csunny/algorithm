#!/usr/bin/env python3

"""
This document is created by magic at 2018/8/20
"""


class Kruskal:
    """
    克鲁斯克算法——最小生成树
    """

    def __init__(self):
        self.parent = dict()
        self.rank = dict()

    def make_set(self, vertex):
        self.parent[vertex] = vertex
        self.rank[vertex] = 0

    def find(self, vertex):
        if self.parent[vertex] != vertex:
            self.parent[vertex] = self.find(self.parent[vertex])
        return self.parent[vertex]

    def union(self, vertex_one, vertex_two):
        root_one = self.find(vertex_one)
        root_two = self.find(vertex_two)

        if root_one != root_two:
            if self.rank[root_one] > self.rank[root_two]:
                self.parent[root_two] = root_one
            else:
                self.parent[root_one] = root_two
            if self.rank[root_one] == self.rank[root_two]:
                self.rank[root_two] += 1

    def kruskal(self, g):
        for vertex in g["vertexs"]:
            self.make_set(vertex)
            minimux_spinning_tree = set()

            edges = sorted(g["edges"])

        for edge in edges:
            weight, vertex_one, vertex_two = edge
            if self.find(vertex_one) != self.find(vertex_two):
                self.union(vertex_one, vertex_two)
                minimux_spinning_tree.add(edge)

        print(sum([i[0] for i in minimux_spinning_tree]))
        return sorted(minimux_spinning_tree)


if __name__ == '__main__':
    g = {
        "vertexs": ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
        "edges": [
            (7, 'A', 'B'),
            (5, 'A', 'D'),
            (7, 'B', 'A'),
            (8, 'B', 'C'),
            (9, 'B', 'D'),
            (7, 'B', 'E'),
            (8, 'C', 'B'),
            (5, 'C', 'E'),
            (5, 'D', 'A'),
            (9, 'D', 'B'),
            (7, 'D', 'E'),
            (6, 'D', 'F'),
            (7, 'E', 'B'),
            (5, 'E', 'C'),
            (15, 'E', 'D'),
            (8, 'E', 'F'),
            (9, 'E', 'G'),
            (6, 'F', 'D'),
            (8, 'F', 'E'),
            (11, 'F', 'G'),
            (9, 'G', 'E'),
            (11, 'G', 'F')
        ]

    }

    k = Kruskal()
    print(k.kruskal(g))