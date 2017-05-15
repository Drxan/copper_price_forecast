#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author: liyinwei
@E-mail: coridc@foxmail.com
@Time: 2017/5/12 9:54
@Description: TODO
"""

import mysql.connector
import pandas as pd


def read_data_from_mysql_by_sql(sql):
    """
    根据sql语句查询数据，并以pandas.DataFrame格式返回
    """
    conn = None
    try:
        conn = mysql.connector.connect(host='mysqlhost', user='cvte', password='cvte@cvte', database='dataset',
                                       use_unicode=True)
        df = pd.read_sql(sql, conn)
        return df
    except Exception as e:
        print(e)
    finally:
        conn.close()


def read_co_data():
    """
    获取沪期铜主力历史交易数据
    """
    sql = """
        SELECT
            *
        FROM
            (
                SELECT
                    *
                FROM
                    shfe_daily
                WHERE
                    product_id != '总计'
                AND product_name != '小计'
                ORDER BY
                    volume_i
            ) AS a
        WHERE
            a.product_name = '铜'
        AND a.price_date > '2015-01-05'
        GROUP BY
            a.price_date,
            a.product_id
        ORDER BY
            a.price_date;
    """
    df = read_data_from_mysql_by_sql(sql)
    tuples = list(zip(*[range(len(df)), df.price_date]))
    # 添加数据索引
    index = pd.MultiIndex.from_tuples(tuples, names=['id', 'date'])
    df.index = index
    return df


def read_pcb_price():
    """
    获取PCB历史价格数据 
    """
    sql = "SELECT * FROM (SELECT * FROM shfe_daily WHERE product_id != '总计' AND product_name != '小计' ORDER BY volume_i) AS a WHERE a.product_name = '铜' AND a.price_date > '2015-01-05' GROUP BY a.price_date, a.product_id ORDER BY a.price_date;"
    df = read_data_from_mysql_by_sql(sql)
    return df


if __name__ == '__main__':
    pass
