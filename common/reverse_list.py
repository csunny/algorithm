#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
This document is created by magic at 2018/8/15
"""


def list_reverse(a):
    """
    This function is order to reverse a list with python.
    :param a: a list object
    :return:

    : doc 这是一个字符串或者列表反转的python实现。
    """
    # judge type of the a, is not str or list raise a value Error.
    if not isinstance(a, list):
        if isinstance(a, str):
            a = list(a)
        else:
            raise ValueError('Value Error, Excepted list or str, but {0} accepted'.format(type(a)))

    for i in range(int(len(a) / 2)):
        # tmp = a[i]
        # a[i] = a[len(a) - i - 1]
        # a[len(a) - i - 1] = tmp
        a[i], a[len(a) -i -1] = a[len(a) -i -1], a[i]
    return a


def test_reverse_list():
    a_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    list_reverse(a_list)

    print(a_list)
    assert a_list == ['g', 'f', 'e', 'd', 'c', 'b', 'a']


def test_reverse_str():
    reversed_b_str = list_reverse("iosadjjs")

    res = ''.join(reversed_b_str)
    print(res)
    assert res == "sjjdasoi"


if __name__ == '__main__':
    test_reverse_list()
    test_reverse_str()
