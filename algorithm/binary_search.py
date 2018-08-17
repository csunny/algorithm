#!/usr/bin/env python3

"""
This document is created by magic at 2018/8/17
"""


def binary_search(values, target):
    """

    :param values:
    :param target:
    :return:
    """
    left, right = 0, len(values) - 1

    while left <= right:
        mid = int((left + right) / 2)

        if target < values[mid]:
            right = mid - 1

        elif target > values[mid]:
            left = mid + 1

        else:
            return mid

    return False


if __name__ == '__main__':
    v = [1, 2, 3, 5, 7, 8]

    assert binary_search(v, 2) == 1
    assert binary_search(v, 5) == 3
    assert binary_search(v, 4) is False

