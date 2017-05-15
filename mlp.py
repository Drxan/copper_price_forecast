#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author: liyinwei
@E-mail: coridc@foxmail.com
@Time: 2017/4/25 9:08
@Description: 采用MLP进行铜价预测
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, median_absolute_error, \
    r2_score
from common import read_co_data


def feature_engineering(df):
    """
    特征工程
    """
    # 数据预处理
    df = pre_process(df).copy()
    # 特征构建
    df = feature_construction(df)
    # 特征选择
    df = feature_selection(df)
    # 删除求历史均值带来的多余的样本(n_sample = max(get_history_days()))
    df = delete_redundant_samples(df)
    # 归一化处理
    df = pp_min_max_scale(df)
    return df


def pre_process(df):
    """
    数据预处理
    """
    # 将星期转换为数值
    df = pp_weekday(df)
    # 处理settlement_price_m异常值（空）
    df = pp_settlement_price_m(df)
    return df


def feature_construction(df):
    """
    特征构建
    """
    df = fc_trend(df)
    df = fc_past_avg(df)
    return df


def feature_selection(df):
    """
    特征选择
    """
    label_column = 'close_price_i'
    features = get_selected_features()
    features.append(label_column)
    return df[features]


def delete_redundant_samples(df):
    """
    删除求历史均值带来的多余的样本(n_sample = max(get_history_days()))
    """
    return df.iloc[max(get_history_days()):]


def pp_min_max_scale(df):
    """
    特征值归一化处理
    """
    # 保存index信息及column信息
    index = df.index
    columns = df.columns
    # 对特征进行归一化
    feature_scaled = preprocessing.MinMaxScaler().fit_transform(df.iloc[:, :-1])

    target = np.array(df.iloc[:, -1])
    target.shape = (len(target), 1)

    # 合并归一化后的X和未做归一化的y（归一化后Pandas 的 DataFrame类型会转换成numpy的ndarray类型）
    df_scaled = pd.DataFrame(np.hstack((feature_scaled, target)))
    # 重新设置索引及column信息
    df_scaled.index = index
    df_scaled.columns = columns

    return df_scaled


def pp_weekday(df):
    """
    Pre_Process: 将星期中文替换为数值
    """
    df['o_weekday'] = df.o_weekday.map({'一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '日': 7}).astype(int)
    return df


def pp_settlement_price_m(df):
    """
    Pre_Process: 过滤settlement_price_m字段为空的记录
    """
    return df.dropna()


def fc_trend(df):
    """
    构建历史趋势特征
    """
    history_days = get_history_days()
    df = get_trend(df, history_days)
    return df


def fc_past_avg(df):
    """
    构建过去n天均值作为特征，一方面在特征中加入历史影响，另一方面避免用当天特征预测当天价格的情况
    """
    history_days = get_history_days()
    df = get_history_avg(df, history_days)
    return df


def get_history_days():
    """
    获取历史数据统计的时间跨度（天/交易日）
    """
    # return [1, 3, 5, 15]
    return [1, 3]


def get_history_features():
    """
    获取需要提取历史统计数据的特征
    """
    features = [  # 'id', 'price_date', 'product_id', 'product_sort_no', 'product_name', 'delivery_month',
        'pre_settlement_price_i', 'open_price_i', 'highest_price_i', 'lowest_price_i',
        'close_price_i', 'settlement_price_i', 'zd1_chg', 'zd2_chg', 'volume_i',
        'open_interest', 'open_interest_chg',
        # 'order_no',
        'highest_price_p', 'lowest_price_p',
        'avg_price_p', 'volume_p', 'turn_over', 'year_volume', 'year_turn_over',
        # 'trading_day',
        'last_price', 'open_price_m', 'close_price_m', 'pre_close_price_m',
        'updown', 'updown1', 'updown2', 'highest_price_m', 'lowest_price_m',
        'avg_price_m', 'settlement_price_m']
    # 'o_year', 'o_month', 'o_day', 'o_weekday', 'o_year_num', 'o_total_num', 'o_trade_day'
    # 'o_imchange_data', 'o_code', 'o_msg', 'report_date', 'update_date', 'print_date'
    return features


def get_common_features():
    """
    原始数据中需要参与预测但不参与构建历史统计特征的特征集合
    """
    features = ['delivery_month', 'o_year', 'o_month', 'o_day', 'o_weekday', 'o_year_num', 'o_total_num']
    return features


def get_trend(df, history_days):
    """
    构建历史趋势信息
    """
    updown_features = ['zd1_chg']
    for index, row in df.iterrows():
        for day in history_days:
            if index[0] < day:
                continue
            for feature in updown_features:
                df.set_value(index[0], feature + '_trend_' + str(day), df[index[0] - day: index[0]][feature].sum())
    return df


def get_history_avg(df, history_days):
    """
    构建包含历史统计信息特征
    """
    features = get_history_features()
    for index, row in df.iterrows():
        for day in history_days:
            if index[0] < day:
                continue
            for feature in features:
                df.set_value(index[0], feature + '_' + str(day), df[index[0] - day: index[0]][feature].mean())
    return df


def get_selected_features():
    # 原始特征
    features = get_common_features()
    # 需考虑历史统计信息的特征
    history_features = get_history_features()
    # 历史统计特征时间跨度（天/交易日）
    history_days = get_history_days()
    for day in history_days:
        features.extend(list(map(lambda x: x + '_' + str(day), [feature for feature in history_features])))
    return features


def model_evaluate(actual, predict):
    print("explained_variance_score: " + str(explained_variance_score(actual, predict)))
    print("mean_absolute_error: " + str(mean_absolute_error(actual, predict)))
    print("mean_squared_error: " + str(mean_squared_error(actual, predict)))
    print("median_absolute_error: " + str(median_absolute_error(actual, predict)))
    print("r2_score: " + str(r2_score(actual, predict)))

    # 趋势正确性评估
    calc_trend_accuracy(predict)


def calc_trend_accuracy(predict):
    """
    趋势正确性评估，即判断当前收盘价与前一天收盘价对比上升/下降趋势是否正确
    """
    # 全局索引
    global_index = fed_data.index
    # 预测的样本总数
    predict_sample_no = predict.size
    # 预测结果趋势正确的样本总数
    correct_trend_no = 0
    for index, pre in predict.iterrows():
        # 获取前一天记录的索引值
        pre_index = global_index.get_values()[global_index.get_loc(index) - 1]
        # 获取前一天的收盘价
        v_pre = fed_data.loc[pre_index]['close_price_i']
        # 获取当前的收盘价
        v_target = fed_data.loc[index]['close_price_i']
        # 当天收盘价的预测值
        v_predict = pre[0]
        if (v_target - v_pre) * (v_predict - v_pre) > 0:
            correct_trend_no += 1
    print("trend accuracy rate: " + str(correct_trend_no / predict_sample_no))


def model_visualization(actual, predict):
    """
    预测结果可视化
    """
    x = range(1, len(actual) + 1)

    plt.figure(figsize=(10, 6))
    plt.title('copper price forecast model evaluating')
    plt.xlabel('samples')
    plt.ylabel('actual price vs. predict price')
    plt.grid(x)

    plt.plot(x, actual, 'x-', label='actual price')
    plt.plot(x, predict, '+-', label='predict price')

    plt.legend(loc='upper right')

    plt.show()


if __name__ == '__main__':
    # 读取原始数据
    raw_data = read_co_data()
    # 特征工程
    fed_data = feature_engineering(raw_data)
    # fed_data.to_excel('fed_data.xlsx')
    # feature vector
    X = fed_data.take(list(range(fed_data.shape[1] - 1)), axis=1)
    # target
    y = np.ravel(fed_data.take([fed_data.shape[1] - 1], axis=1))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    # 定义一个BP神经网络
    reg = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
    # 训练
    reg.fit(X_train, y_train)
    # 预测
    y_pred = reg.predict(X_test)
    y_pred = pd.DataFrame(y_pred)
    y_pred.index = X_test.index
    # 将结果写入文件
    # pd.DataFrame(y_pred).to_excel('y_pred.xlsx')
    # 模型评估
    model_evaluate(y_test, y_pred)
    # 可视化
    model_visualization(y_test, y_pred)

    print(type(X), type(y), type(X_train), type(X_test), type(y_train), type(y_test), type(y_pred))
