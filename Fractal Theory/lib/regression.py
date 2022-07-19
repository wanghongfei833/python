# -*- coding: utf-8 -*-
# @Time    : 2022/7/19 15:17
# @Author  : HongFei Wang
import numpy as np


def linear_regression(x, y):
    sumx, sumy, N = np.sum(x), np.sum(y), len(x)
    sumx2, sumxy = np.sum(x ** 2), np.sum(x * y)
    A = np.array([[N, sumx], [sumx, sumx2]])
    b = np.array([sumy, sumxy])
    b, k = np.linalg.solve(A, b)
    ess = x * k + b # 预测值
    sse = np.sum((ess - y) ** 2)
    sst = np.sum((y-np.mean(y))**2)
    RR = 1- sse/sst
    return RR, b, k