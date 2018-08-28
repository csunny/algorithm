#!/usr/bin/env python3

"""
This document is created by magic at 2018/8/16

doc 红黑树是一颗二叉搜索树，他在每个节点上增加一个存储位置来表示即节点的颜色。可以是RED或者是BLACK，树中的每个节点包括5个属性： color、key、left、right、parent

红黑树满足下列性质：
- 每个节点或是红色的，或是黑色的
- 根节点是黑色的
- 每个叶子节点
- 如果一个节点是红色的，则它的两个子节点都是黑色的。
- 对每个节点，从该节点到其他所有后代叶节点的简单路径上，均包含相同数目的黑色节点。

https://zh.wikipedia.org/wiki/%E7%BA%A2%E9%BB%91%E6%A0%91
"""


class TreeNode:
    def __init__(self, item):
        self.key = item
        self.left = None
        self.right = None
        self.parent = None
        self.color = 'black'


class RedBlackTree:
    """
    红黑树
    """

    def __init__(self, root):
        self.root = root

    def left_rotate(self, x):
        """左旋转"""
        pass

    def right_rotate(self, x):
        """
        右旋转
        :param x:
        :return:
        """
        pass

    def insert(self, node):
        """
        红黑树插入
        :param node:
        :return:
        """
        pass

    def color(self, node):
        """红黑树上色"""
        pass
