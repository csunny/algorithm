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
    """
    # judge type of the a, is not str or list raise a value Error.
    if not isinstance(a, list):
        if isinstance(a, str):
            a = list(a)
        else:
            raise ValueError('Value Error, Excepted list or str, but {0} Input'.format(type(a)))

    for i in range(int(len(a) / 2)):
        tmp = a[i]
        a[i] = a[len(a) - i - 1]
        a[len(a) - i - 1] = tmp

    return a


def test_reverse_list():
    a_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    list_reverse(a_list)

    assert a_list == ['g', 'f', 'e', 'd', 'c', 'b', 'a']


def test_reverse_str():
    b_str = 'iosadjjs'
    reversed_b_str = list_reverse(b_str)

    res = ''.join(reversed_b_str)
    assert res == "sjjdasoi"


if __name__ == '__main__':
    test_reverse_list()
    test_reverse_str()
