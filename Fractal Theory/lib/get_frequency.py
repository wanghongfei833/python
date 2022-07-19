# -*- coding: utf-8 -*-
# @Time    : 2022/7/19 14:53
# @Author  : HongFei Wang
import matplotlib.pyplot as plt
import cv2
import numpy as np
import numba
from osgeo import gdal



@numba.njit()  # 计算频数
def count_frequency(datas, lists):
    for i in range(len(datas)):
        for j in range(len(datas[i])):
            lists[datas[i][j]] += 1
    return lists

@numba.njit()
def count_accumulative_frequency(datas):
    nums = 0
    datas_copy = datas.copy()
    for i in range(1, 1 + len(datas_copy)):
        nums += datas_copy[-i]
        datas_copy[-i] = nums
    return datas_copy


def save_log(file_name:str):
    if '.tif'  in file_name or '.' not in file_name:
        image_data = gdal.Open(rf'{file_name}')
        image_data = image_data.ReadAsArray()
    else:
        image_data = np.array(cv2.imread(file_name,0))
        # image_data = plt.imread(file_name).astype(np.int64)

    mins_image = np.min(image_data)
    image_data -= mins_image-1 # 平移至1--..
    image_data_list, frequency = np.unique(image_data, return_counts=True) # 返回 索引值，频次
    accumulative_frequency = count_accumulative_frequency(frequency)  # 累积频率


    plt.figure('image')
    plt.imshow(image_data,cmap='gray')
    plt.figure('frequency')
    plt.plot(np.log(image_data_list),np.log(frequency))
    plt.figure('accumulative_frequency')
    plt.plot(np.log(image_data_list),np.log(accumulative_frequency))
    plt.show()

    if '.' in file_name:
        np.save("../db/%s_frequency.npy" % ''.join(file_name.split('.')[0]), frequency) # 频次
        np.save("../db/%s_accumulative_frequency.npy" % ''.join(file_name.split('.')[0]), accumulative_frequency)  # 累计频次
        np.save("../db/%s_index.npy" % ''.join(file_name.split('.')[0]), image_data_list) # 索引
    else:
        np.save("../db/%s_frequency.npy" % file_name, frequency)  # 频次
        np.save("../db/%s_accumulative_frequency.npy" % file_name, accumulative_frequency)  # 累计频次
        np.save("../db/%s_index.npy" % file_name, image_data_list)  # 索引


if __name__ == '__main__':
    save_log('image.jfif')