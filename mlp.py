#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author: liyinwei
@E-mail: coridc@foxmail.com
@Time: 2017/4/25 9:08
@Description: TODO
"""

import mysql.connector
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, median_absolute_error, \
    r2_score


def read_data():
    """
    连接MySQL数据库，并读取沪期铜主力的交易记录，以pandas.DataFrame格式返回
    """
    conn = None
    try:
        conn = mysql.connector.connect(host='mysqlhost', user='cvte', password='cvte@cvte', database='dataset',
                                       use_unicode=True)
        sql = "SELECT * FROM (SELECT * FROM shfe_daily WHERE product_id != '总计' AND product_name != '小计' ORDER BY volume_i) AS a WHERE a.product_name = '铜' AND a.price_date > '2015-01-05' GROUP BY a.price_date, a.product_id ORDER BY a.price_date;"
        df = pd.read_sql(sql, conn)
        return df
    except Exception as e:
        print(e)
    finally:
        conn.close()


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


def fc_past_avg(df):
    """
    构建过去n天均值作为特征，一方面在特征中加入历史影响，另一方面避免用当天特征预测当天价格的情况
    """
    history_days = get_history_days()
    for day in history_days:
        df = get_history_avg(df, day)
    # print(df.describe())
    return df


def get_history_days():
    """
    获取历史数据统计的时间跨度（天/交易日）
    """
    return [3, 5, 15, 30]


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
    features = ['delivery_month', 'o_year', 'o_month', 'o_day', 'o_weekday', 'o_year_num', 'o_total_num', 'o_trade_day']
    return features


def get_history_avg(df, day):
    """
    构建包含历史统计信息特征
    """
    features = get_history_features()
    for index, row in df.iterrows():
        if index < day:
            continue
        for feature in features:
            df.set_value(index, feature + '_' + str(day), df[index - day: index][feature].mean())
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


def model_evaluate(actual, predict):
    print("explained_variance_score: " + str(explained_variance_score(actual, predict)))
    print("mean_absolute_error: " + str(mean_absolute_error(actual, predict)))
    print("mean_squared_error: " + str(mean_squared_error(actual, predict)))
    print("median_absolute_error: " + str(median_absolute_error(actual, predict)))
    print("r2_score: " + str(r2_score(actual, predict)))


def model_visualization(actual, predict):
    """
    预测结果可视化 
    """
    x = range(1, len(actual) + 1)

    plt.figure(figsize=(10, 6))
    plt.title('copper price forecast model evaluating')
    plt.xlabel('days')
    plt.ylabel('actual price vs. predict price')
    plt.grid(x)

    plt.plot(x, actual, 'x-', label='actual price')
    plt.plot(x, predict, '+-', label='predict price')

    plt.legend(loc='upper right')

    plt.show()


if __name__ == '__main__':
    # 读取原始数据
    raw_data = read_data()
    # 特征工程
    fed_data = feature_engineering(raw_data)
    # print(fed_data.describe())
    X = fed_data[30:][get_selected_features()]
    y = fed_data[30:]['close_price_i']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    pd.DataFrame(y_test).to_excel('y_test.xlsx')
    # 定义一个BP神经网络
    reg = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
    # 训练
    reg.fit(X_train, y_train)
    # 预测
    y_pred = reg.predict(X_test)
    pd.DataFrame(y_pred).to_excel('y_pred.xlsx')
    # 模型评估
    model_evaluate(y_test, y_pred)
    # 可视化
    model_visualization(y_test, y_pred)
