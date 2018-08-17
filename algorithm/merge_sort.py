#!/usr/bin/env python3

"""
This document is created by magic at 2018/8/17

This is impletement of merge sort
归并排序
"""


def merge_sort(values):
    n = len(values)
    if n <= 1:
        return values

    key = int(n / 2)
    left = merge_sort(values[:key])
    right = merge_sort(values[key:])

    return merge(left, right)


def merge(left, right):
    """
    合并两个已经排好序的列表
    :param left:
    :param right:
    :return:
    """
    tmp = []
    i, j = 0, 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            tmp.append(left[i])
            i += 1
        else:
            tmp.append(right[j])
            j += 1

    tmp += left[i:]
    tmp += right[j:]

    return tmp


if __name__ == '__main__':
    v = [5, 3, 2, 4, 7, 8, 1, 9]

    print(merge_sort(v))
