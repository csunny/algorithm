#!/usr/bin/env python
# -*- coding:utf-8 -*-


def find_dump(values):
    tmp_list = []
    for i in values:
        if i in tmp_list:
            print("{0} is dupmed.".format(i))
            return i

        tmp_list.append(i)
    print("没有重复元素")


if __name__ == '__main__':
    l = [1, 2, 3, 2, 4, 5]
    find_dump(l)