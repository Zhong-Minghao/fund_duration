"""
中债指数数据处理模块
功能：读取中债财富指数价格和久期数据
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys

from util import smooth_series

# 设置UTF-8编码输出（兼容notebook环境）
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass


class BondIndexDataProcessor:
    """中债指数数据处理类"""

    def __init__(self, index_path='中债财富指数.xlsx'):
        """
        初始化

        参数:
        index_path: 中债财富指数文件路径
        """
        self.index_path = index_path
        self.price_df = None
        self.duration_df = None

        # 研报中使用的指数代码映射
        # 中长期利率型债基使用的指数
        self.medium_long_rate_indices = [
            # 'CBA05821.CS',  # 中债国债及政策性银行债指数(1-3年)
            'CBA05831.CS',  # 中债国债及政策性银行债指数(3-5年)
            'CBA05841.CS',  # 中债国债及政策性银行债指数(5-7年)
            'CBA05851.CS',  # 中债国债及政策性银行债指数(7-10年)
            'CBA05861.CS',  # 中债国债及政策性银行债指数(10年以上)
            'CBA07501.CS',  # 中债同业存单总指数(总值)
        ]

        # 中长期信用型债基使用的指数
        self.medium_long_credit_indices = [
            # 'CBA02711.CS',  # 中债信用债总指数(1年以下)
            'CBA02721.CS',  # 中债信用债总指数(1-3年)
            'CBA02731.CS',  # 中债信用债总指数(3-5年)
            'CBA02741.CS',  # 中债信用债总指数(5-7年)
            'CBA02751.CS',  # 中债信用债总指数(7-10年)
            'CBA02761.CS',  # 中债信用债总指数(10年以上)

            'CBA05831.CS',  # 中债国债及政策性银行债指数(3-5年)
            'CBA05841.CS',  # 中债国债及政策性银行债指数(5-7年)
            'CBA05851.CS',  # 中债国债及政策性银行债指数(7-10年)
            'CBA05861.CS',  # 中债国债及政策性银行债指数(10年以上)
            'CBA07501.CS',  # 中债同业存单总指数(总值)
        ]

        # 短期利率型债基使用的指数
        self.short_rate_indices = [
            'CBA07511.CS',  # 中债同业存单总指数(0-3个月)
            'CBA07521.CS',  # 中债同业存单总指数(3-6个月)
            'CBA07531.CS',  # 中债同业存单总指数(6-9个月)
            'CBA07541.CS',  # 中债同业存单总指数(9-12个月)
            'CBA05821.CS',  # 中债国债及政策性银行债财富(1-3年)
        ]

        # 短期信用型债基使用的指数
        self.short_credit_indices = [
            'CBA01831.CS',  # 中债短融总指数(0-3个月)
            'CBA01841.CS',  # 中债短融总指数(3-6个月)
            'CBA01851.CS',  # 中债短融总指数(6-9个月)
            'CBA01861.CS',  # 中债短融总指数(9-12个月)
            'CBA02821.CS',  # 中债新中期票据总指数(1-3年)
        ]

    def load_price_data(self):
        """加载指数价格数据"""
        # 读取原始数据
        df_raw = pd.read_excel(self.index_path, sheet_name='Sheet1', header=None)

        # 只使用前22列（索引0-21），后面的列是重复的
        df_raw = df_raw.iloc[:, :22]

        # 第4行（索引4）是代码行，作为列名
        codes_row = df_raw.iloc[4, :].tolist()
        data_start_row = 5  # 数据从第6行开始（索引5）

        # 创建新的DataFrame
        self.price_df = df_raw.iloc[data_start_row:, :].copy()
        self.price_df.columns = codes_row

        # 第0列是日期列
        date_col = self.price_df.columns[0]

        # 过滤掉日期为空的行
        self.price_df = self.price_df[self.price_df[date_col].notna()]

        # 转换日期
        self.price_df[date_col] = pd.to_datetime(self.price_df[date_col], errors='coerce')
        self.price_df = self.price_df.dropna(subset=[date_col])

        # 设置日期为索引，并命名索引为'date'
        self.price_df = self.price_df.set_index(date_col)
        self.price_df.index.name = 'date'

        # 转换数据类型为数值
        for col in self.price_df.columns:
            self.price_df[col] = pd.to_numeric(self.price_df[col], errors='coerce')

        return self.price_df

    def load_duration_data(self):
        """加载指数久期数据"""
        # 第3行（索引2）是列名行
        self.duration_df = pd.read_excel(self.index_path, sheet_name='Sheet2', header=2)

        # 第0列是时间列
        time_col = self.duration_df.columns[0]
        # 跳过非日期的值
        self.duration_df = self.duration_df[self.duration_df[time_col].notna()]
        self.duration_df[time_col] = pd.to_datetime(self.duration_df[time_col], errors='coerce')
        self.duration_df = self.duration_df.dropna(subset=[time_col])
        self.duration_df = self.duration_df.set_index(time_col)

        # 直接转换列名后缀：.CB -> .CS
        new_columns = []
        for col in self.duration_df.columns:
            if isinstance(col, str) and col.endswith('.CB'):
                new_columns.append(col.replace('.CB', '.CS'))
            else:
                new_columns.append(col)

        self.duration_df.columns = new_columns

        return self.duration_df

    def get_index_prices(self, index_codes, start_date, end_date):
        """
        获取指定指数的价格数据

        参数:
        index_codes: 指数代码列表
        start_date: 开始日期 'YYYY-MM-DD'
        end_date: 结束日期 'YYYY-MM-DD'

        返回:
        DataFrame: 指数价格数据
        """
        if self.price_df is None:
            self.load_price_data()

        # 过滤出实际存在的列
        valid_codes = [code for code in index_codes if code in self.price_df.columns]
        if not valid_codes:
            return pd.DataFrame()

        # 筛选日期范围
        mask = (self.price_df.index >= start_date) & (self.price_df.index <= end_date)
        df = self.price_df.loc[mask, valid_codes]

        return df

    def get_index_prices_smoothed(self, index_codes, start_date, end_date, window=5):
        """
        获取平滑后的指数价格数据（滚动均值），与基金侧平滑口径保持一致

        参数:
        index_codes: 指数代码列表
        start_date: 开始日期 'YYYY-MM-DD'
        end_date: 结束日期 'YYYY-MM-DD'
        window: 平滑窗口，默认5（与WindDataFetcher.get_fund_nav_smoothed的默认一致）

        返回:
        DataFrame: 平滑后的指数价格数据
        """
        df = self.get_index_prices(index_codes, start_date, end_date)
        if df.empty:
            return df
        return smooth_series(df, window=window)

    def get_index_durations(self, index_codes, start_date, end_date):
        """
        获取指定指数的久期数据

        参数:
        index_codes: 指数代码列表
        start_date: 开始日期 'YYYY-MM-DD'
        end_date: 结束日期 'YYYY-MM-DD'

        返回:
        DataFrame: 指数久期数据
        """
        if self.duration_df is None:
            self.load_duration_data()

        # 过滤出实际存在的列
        valid_codes = [code for code in index_codes if code in self.duration_df.columns]
        if not valid_codes:
            return pd.DataFrame()

        # 筛选日期范围
        mask = (self.duration_df.index >= start_date) & (self.duration_df.index <= end_date)
        df = self.duration_df.loc[mask, valid_codes]

        return df

    def get_latest_duration(self, index_code, target_date):
        """
        获取指定指数在目标日期的最新久期值

        参数:
        index_code: 指数代码
        target_date: 目标日期 'YYYY-MM-DD'

        返回:
        float: 久期值
        """
        if self.duration_df is None:
            self.load_duration_data()

        # 检查代码是否存在
        if index_code not in self.duration_df.columns:
            return None

        # 获取目标日期之前的所有久期数据
        mask = self.duration_df.index <= target_date
        df = self.duration_df.loc[mask, [index_code]]

        if df.empty:
            return None

        # 返回最新的久期值
        return df.iloc[-1, 0]

    def calculate_index_returns(self, price_df):
        """
        计算指数收益率

        参数:
        price_df: 指数价格DataFrame

        返回:
        DataFrame: 指数收益率
        """
        return_df = price_df.pct_change()
        return return_df


if __name__ == '__main__':
    # 测试
    processor = BondIndexDataProcessor()

    # 加载数据
    print("加载指数价格数据...")
    processor.load_price_data()
    print(f"价格数据Shape: {processor.price_df.shape}")
    print(processor.price_df.head())

    print("\n加载指数久期数据...")
    processor.load_duration_data()
    print(f"久期数据Shape: {processor.duration_df.shape}")
    print(processor.duration_df.head())

    # 测试获取特定指数数据
    print("\n中长期利率型指数代码:")
    print(processor.medium_long_rate_indices)

    print("\n测试获取指数价格数据...")
    test_codes = processor.medium_long_rate_indices[:3]
    test_prices = processor.get_index_prices(test_codes, '2024-01-01', '2024-12-20')
    print(test_prices.head())
