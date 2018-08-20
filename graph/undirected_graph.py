#!/usr/bin/env python3

"""
This document is created by magic at 2018/8/20
"""


class Dijkstra:
    """
    Dijkstra算法
    """

    def __init__(self, graph, costs, parents):
        self.graph = graph
        self.costs = costs
        self.parents = parents

        self.processed = []

    def find_lowest_cost_node(self):
        lowest_cost = float("inf")
        lowest_cost_node = None

        for node in self.costs:
            cost = self.costs[node]

            if cost < lowest_cost and node not in self.processed:
                lowest_cost = cost
                lowest_cost_node = node

        return lowest_cost_node

    def dag(self):
        node = self.find_lowest_cost_node()
        while node:
            cost = self.costs[node]
            neighbors = self.graph[node]
            for n in neighbors.keys():
                new_cost = cost + neighbors[n]
                if self.costs[n] > new_cost:
                    self.costs[n] = new_cost
                    self.parents[n] = node
            self.processed.append(node)
            node = self.find_lowest_cost_node()
        return self.costs


def main():
    graph = {
        "start": {
            "a": 6,
            "b": 2
        },
        "a": {
            "fin": 1
        },
        "b": {
            "a": 3,
            "fin": 5
        },
        "fin": {

        }
    }

    costs = {
        "a": 6,
        "b": 2,
        "fin": float("inf")
    }

    parents = {
        "a": "start",
        "b": "start",
        "fin": None
    }

    D = Dijkstra(graph, costs, parents)
    costs = D.dag()
    print(costs)


if __name__ == '__main__':
    main()
