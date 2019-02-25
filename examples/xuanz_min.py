#!/usr/bin/env python
# -*- coding:utf-8 -*-


def rev_l(v):
    if len(v) <=1:
        return v
    
    for i in range(len(v)):
        if v[i+1] <= v[i]:
            return v[i+1]

    return v[0]

if __name__ == '__main__':
    m = rev_l([2, 2, 2, 2, 2, 2])
    print(m)


