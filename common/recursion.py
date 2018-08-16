#!/usr/bin/env python3

"""
This document is created by magic at 2018/8/16
"""


def fab(n):
    """
    use fab to explain recursion function

    :doc 斐波拉切数列的递归实现。递归函数是调用自身的函数。
    :param n:
    :return:
    """

    if n <= 2:
        return n
    else:
        return fab(n - 1) + fab(n - 2)


if __name__ == '__main__':
    print(fab(5))
