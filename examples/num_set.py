#!/usr/bin/env python3

#  用数字1 2 2 3 4 5 这六个数字，写一个main函数，打印出所有不同的排列， 例如512234, 412345.
#  要求4不能在第三位，且3跟5不能相邻

# num_lis = [1, 2, 2, 3, 4, 5]

# res_set = []
# for i1 in num_lis:
#     for i2 in [i for i in num_lis if i != i1]:  
#         for i3 in [i for i in num_lis if i not in [i1, i2]]:
#             for i4 in [i for i in num_lis if i not in [i1, i2, i3]]:
#                 for i5 in [ i for i in num_lis if i not in [i1, i2, i3, i4]]:
                    
#                     res = str(i1) + str(i2) + str(i3) + str(i4) + str(i5) + str(i6)
#                     res_set.append(res)

# no_rpt = (list(set(res_set)))
# print(no_rpt)

class NS:
    def __init__(self, arr):
        self.numbers = arr
        self.visited = [0] * len(self.numbers)
        self.graph = [([0] * len(self.numbers)) for i in range(len(self.numbers))]
        self.n = 6
        self.combination = ""
        self.s = set()

    def dfs(self, start):
        self.visited[start] = True
        self.combination += str(self.numbers[start])
        if len(self.combination) == self.n:
            if self.combination[2] != '4':
                self.s.add(self.combination)

        j = 0
        while j < self.n:
            if self.graph[start][j] == 1 and self.visited[j] == False:
                self.dfs(j)
            j += 1
        self.combination = self.combination[:-1]
        self.visited[start] = False


    def get_all_com(self):
        i = 0
        while i < self.n:
            j = 0
            while j < self.n:
                if i == j:
                    self.graph[i][j] = 0
                else:
                    self.graph[i][j] = 1
                j += 1

            i += 1

        self.graph[3][5] = 0
        self.graph[5][3] = 0

        print(self.graph)
            # 开始遍历
        i = 0
        while i < self.n:
            self.dfs(i)
            i += 1


if __name__ == '__main__':
    arr = [1, 2, 2, 3, 4, 5]
    t = NS(arr)

    t.get_all_com()
    print(t.s)