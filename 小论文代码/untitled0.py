# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 19:14:04 2019

@author: Winry
"""

import sys
import os
import re

#请完成下面这个函数，实现题目要求的功能
#当然，你也可以不按照下面这个模板来作答，完全按照自己的想法来 ^-^ 
#******************************开始写代码******************************


def  main():
    res = []
    n = int(input())
    color = input()
    color = color.split(" ")
    color = [int(x) for x in color]
    numColor = max(color)
    minCc = min(color)
    if minCc == numColor:
        return 1
    else:
        flag = numColor
        while flag>0:
            temp = 0
            for i in range(n):
                if color[i] == flag:
                    temp = temp+1
            res.append(temp)
            flag = flag-1
        minres = min(res)
        if minres == 1:
            return 0
        else:
            rree = 0

            for j in range(minres-1):
                coun = j+2
                trycount = [x%coun for x in res]
                if sum(trycount) == 0:
                    result = [x/coun for x in res]
                    rree = int(sum(result))
                    
            return rree

            
            
                


#******************************结束写代码******************************


  
res = main()

print(str(res) + "\n")