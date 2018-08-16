#!/usr/bin/env python3

"""
This document is created by magic at 2018/8/16
"""


def bubble_sort(values):
    """
    冒泡排序
    :param values:
    :return:
    """
    length = len(values) - 1

    if length <= 1:
        return values

    for i, _ in enumerate(values):
        for j in range(length, i, -1):
            if values[j] < values[j - 1]:
                values[j], values[j - 1] = values[j - 1], values[j]

    return values


if __name__ == '__main__':
    v = [3, 4, 5, 2, 7, 9, 8, 1]
    sort_v = bubble_sort(v)
    print(sort_v)
