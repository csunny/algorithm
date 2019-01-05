#!/usr/bin/env python3

# 给定一个正整数n， 求解出所有和为n的整数组合， 要求组合按照递增的方式展示，而且唯一，例如： 4=1+1+1+1，4=1+1+2

def get_all_combination(sums, result, count):
    if sums < 0:
        return
    
    if sums == 0:
        print("满足条件的组合...")
        i = 0
        while i < count:
            print(result[i])
            i += 1

    i = (1 if count == 0 else result[count-1])

    while i <= sums:
        result[count] = i
        count += 1
        get_all_combination(sums - i, result, count)
        count -= 1
        i += 1

def show_all(n):
    if n <1:
        print("参数不满足要")
        return
    result = [0] * n
    get_all_combination(n, result, 0)


if __name__ == '__main__':
    show_all(6)