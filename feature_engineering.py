#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author: liyinwei
@E-mail: coridc@foxmail.com
@Time: 2017/5/15 14:14
@Description: 特征工程
"""

import pandas as pd
import numpy as np

from sklearn import preprocessing


def feature_engineering(df):
    """
    特征工程
    """
    # 数据预处理
    df = _pre_process(df).copy()
    # 特征构建
    df = _feature_construction(df)
    # 特征选择
    df = _feature_selection(df)
    # 删除求历史均值带来的多余的样本(n_sample = max(get_history_days()))
    df = _delete_redundant_samples(df)
    # 归一化处理
    df = _pp_min_max_scale(df)
    return df


def _pre_process(df):
    """
    数据预处理
    """
    # 将星期转换为数值
    df = _pp_weekday(df)
    # 处理settlement_price_m异常值（空）
    df = _pp_settlement_price_m(df)
    return df


def _feature_construction(df):
    """
    特征构建
    """
    df = _fc_trend(df)
    df = _fc_past_avg(df)
    return df


def _feature_selection(df):
    """
    特征选择
    """
    label_column = 'close_price_i'
    features = _get_selected_features()
    features.append(label_column)
    return df[features]


def _delete_redundant_samples(df):
    """
    删除求历史均值带来的多余的样本(n_sample = max(get_history_days()))
    """
    return df.iloc[max(_get_history_days()):]


def _pp_min_max_scale(df):
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


def _pp_weekday(df):
    """
    Pre_Process: 将星期中文替换为数值
    """
    df['o_weekday'] = df.o_weekday.map({'一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '日': 7}).astype(int)
    return df


def _pp_settlement_price_m(df):
    """
    Pre_Process: 过滤settlement_price_m字段为空的记录
    """
    return df.dropna()


def _fc_trend(df):
    """
    构建历史趋势特征
    """
    history_days = _get_history_days()
    df = _get_trend(df, history_days)
    return df


def _fc_past_avg(df):
    """
    构建过去n天均值作为特征，一方面在特征中加入历史影响，另一方面避免用当天特征预测当天价格的情况
    """
    history_days = _get_history_days()
    df = _get_history_avg(df, history_days)
    return df


def _get_history_days():
    """
    获取历史数据统计的时间跨度（天/交易日）
    """
    # return [1, 3, 5, 15]
    return [1, 3]


def _get_history_features():
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


def _get_common_features():
    """
    原始数据中需要参与预测但不参与构建历史统计特征的特征集合
    """
    features = ['delivery_month', 'o_year', 'o_month', 'o_day', 'o_weekday', 'o_year_num', 'o_total_num']
    return features


def _get_trend(df, history_days):
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


def _get_history_avg(df, history_days):
    """
    构建包含历史统计信息特征
    """
    features = _get_history_features()
    for index, row in df.iterrows():
        for day in history_days:
            if index[0] < day:
                continue
            for feature in features:
                df.set_value(index[0], feature + '_' + str(day), df[index[0] - day: index[0]][feature].mean())
    return df


def _get_selected_features():
    # 原始特征
    features = _get_common_features()
    # 需考虑历史统计信息的特征
    history_features = _get_history_features()
    # 历史统计特征时间跨度（天/交易日）
    history_days = _get_history_days()
    for day in history_days:
        features.extend(list(map(lambda x: x + '_' + str(day), [feature for feature in history_features])))
    return features


if __name__ == '__main__':
    pass
