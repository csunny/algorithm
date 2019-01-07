#!/usr/bin/env python3


def permurarion(s, start):
    if not s or start<0:
        return
    
    if start == len(s) -1:
        print("".join(s))
    else:
        i = start
        while i < len(s):
            s[i], s[start] = s[start], s[i]
            permurarion(s, start+1)
            s[i], s[start] = s[start], s[i] 

            i += 1

def permulation_trans(s):
    strr = list(s)
    permurarion(strr, 0)

if __name__ == '__main__':
    permulation_trans('ab')