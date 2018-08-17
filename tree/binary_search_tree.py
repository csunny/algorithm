#!/usr/bin/env python3

"""
This document is created by magic at 2018/8/16
"""


class Node:
    """
    二叉搜索树的节点
    """

    def __init__(self, val, left=None, right=None):
        self._val = val
        self._left = left
        self._right = right

    @property
    def val(self):
        return self._val

    @val.setter
    def val(self, value):
        self._val = value

    @property
    def left(self):
        return self._left

    @left.setter
    def left(self, node):
        self._left = node

    @property
    def right(self):
        return self._right

    @right.setter
    def right(self, node):
        self._right = node

    def __iter__(self):
        if self._left:
            for element in self._left:
                yield element

        yield self._val

        if self.right:
            for element in self.right:
                yield element


class BinarySearchTree:
    """
    实现一个二叉搜索树
    """

    def __init__(self):
        self.root = None

    def insert(self, val):
        self.root = self._insert(self.root, val)

    def _insert(self, root, val):
        if not root:
            root = Node(val)
            return root

        if val < root.val:
            root.left = self._insert(root.left, val)

        else:
            root.right = self._insert(root.right, val)

        return root

    def search(self, target):
        return self._search(self.root, target)

    def _search(self, node, val):
        if not node:
            return False
        if val > node.val:
            return self._search(node.right, val)
        elif val < node.val:
            return self._search(node.left, val)
        else:
            return True

    def __iter__(self):
        if self.root:
            return self.root.__iter__()
        else:
            [].__iter__()


def main():
    # s = input("Input a list of numbers:")
    # lst = s.split()
    lst = [4, 3, 5, 12, 6, 7, 9]

    print(lst)
    tree = BinarySearchTree()
    for x in lst:
        tree.insert(float(x))

    for x in tree:
        print(x)

    assert tree.search(3) is True
    assert tree.search(8) is False


def test_node():
    node1 = Node(1)
    print(node1.val)
    print(node1.left)

    print(node1.right)

    node2 = Node(2, node1)
    print(node2.val, node2.left, node2.right)

    node2.left = Node(3)

    print(node2.left.val)


if __name__ == '__main__':
    main()
