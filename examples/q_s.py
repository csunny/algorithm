#!/usr/bin/env python
# -*- coding:utf-8 -*-

def q_s(v):
    if len(v) <=1:
        return v
    left, right = 0, len(v) - 1 

    item = v[int((left+right))/2]

    left_value = [i for i in v if i < item]
    right_value = [i for i in v if i > item]


    return q_s(left_value) + [item] + q_s(right_value)


if __name__ == "__main__":
    v = [1, 2, 4, 3, 6, 7, 8, 5, 9]
    q_v = q_s(v)
    print(q_v)
