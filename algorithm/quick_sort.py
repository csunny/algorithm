#!/usr/bin/env python3

"""
This document is created by magic at 2018/8/16
"""


def quick_sort(values):
    """
    a simple impletement
    :param values:
    :return:
    """
    if len(values) <= 1:
        return values

    priv = values[0]

    less = [i for i in values[1:] if i < priv]
    greater = [i for i in values[1:] if i > priv]

    return quick_sort(less) + [priv] + quick_sort(greater)


# def quick_sort_2(values):
#     """
#     another impletement
#     :param values:
#     :return:
#     """
#
#     if len(values) <= 1:
#         return values
#
#     priv = values[0]
#
#     head, tail = 0, len(values) - 1
#
#     i = 1
#     while i <= tail:
#
#         if values[i] > priv:
#             values[i], values[tail] = values[tail], values[i]
#             tail -= 1
#         else:
#             values[i], values[head] = values[head], values[i]
#             head += 1
#             i += 1
#
#     quick_sort_2(values[:head])
#     quick_sort_2(values[head+1:])
#
#     return values

def quick_sort_v3(values, lo, hi):
    if lo < hi:
        p = partition(values, lo, hi)
        quick_sort_v3(values, lo, p)
        quick_sort_v3(values, p + 1, hi)
    return values


def partition(lst, lo, hi):
    pivot = lst[hi - 1]
    i = lo - 1
    for j in range(lo, hi):
        if lst[j] < pivot:
            i += 1
            lst[i], lst[j] = lst[j], lst[i]
    if lst[hi - 1] < lst[i + 1]:
        lst[i + 1], lst[hi - 1] = lst[hi - 1], lst[i + 1]
    return i + 1


if __name__ == '__main__':
    v1 = [3, 4, 2, 6]
    v2 = quick_sort_v3(v1, 0, 4)
    print(v2)
