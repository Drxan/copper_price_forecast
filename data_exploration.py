#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author: liyinwei
@E-mail: coridc@foxmail.com
@Time: 2017/5/12 9:48
@Description: TODO
"""

from data_loading import read_co_data
import matplotlib.pyplot as plt


def data_visualization(df):
    """
    原始数据可视化
    """
    x_values = df.price_date
    y_values = df.close_price_i

    plt.figure(figsize=(10, 6))
    plt.title('history copper price')
    plt.xlabel('')
    plt.ylabel('history price(rmb/t)')

    plt.plot(x_values, y_values, '-', label='history price')

    plt.legend(loc='upper right')

    plt.show()


if __name__ == '__main__':
    # 读取原始数据
    raw_data = read_co_data()
    # 可视化铜价历史数据及PCB价格历史数据
    data_visualization(raw_data)
