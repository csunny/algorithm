#!/usr/bin/env python
# -*- coding:utf-8 -*-

def max_min(v):
    m_min, m_max = min(v[0], v[-1]), max(v[0], v[-1])
    for i in v[1:len(v)-1]:
        if i<m_min:
            m_min = i
        elif i > m_max:
            m_max = i
    
    return m_min, m_max

if __name__ == '__main__':
    m_min, m_max = max_min([9, 3, 4, 7, 2, 0])
    print(m_min, m_max)
