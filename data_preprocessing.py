"""
数据预处理模块
功能：读取基金样本数据，筛选符合条件的基金
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys

# 设置UTF-8编码输出（兼容notebook环境）
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    # notebook环境中不支持reconfigure
    pass


class FundDataPreprocessor:
    """基金数据预处理类"""

    def __init__(self,
                 short_term_path='短期纯债基金样本数据.xlsx',
                 medium_long_term_path='中长期纯债基金样本数据.xlsx'):
        """
        初始化

        参数:
        short_term_path: 短期纯债基金样本数据路径
        medium_long_term_path: 中长期纯债基金样本数据路径
        """
        self.short_term_path = short_term_path
        self.medium_long_term_path = medium_long_term_path

    def read_fund_data(self, file_path):
        """
        读取基金数据

        返回:
        DataFrame: 基金数据
        """
        df = pd.read_excel(file_path, header=1)
        return df

    def filter_funds(self, df, target_date):
        """
        筛选符合条件的基金

        筛选条件:
        1. 基金已经成立 (fund_setupdate <= target_date)
        2. 基金未到期 (fund_maturitydate_2 > target_date 或为空)
        3. 不是摊余成本法 (fund_valuationmethod != '摊余成本法')
        4. 不是定期开放基金 (fund_regulopenfundornot != '是')
        5. 是初始基金 (fund_initial == '是')

        参数:
        df: 基金数据
        target_date: 目标日期 (datetime 或 str 'YYYY-MM-DD')

        返回:
        DataFrame: 筛选后的基金
        """
        if isinstance(target_date, str):
            target_date = pd.to_datetime(target_date)

        # 处理日期列
        df['fund_setupdate'] = pd.to_datetime(df['fund_setupdate'], errors='coerce')

        # 条件1: 基金已经成立
        mask1 = df['fund_setupdate'] <= target_date

        # 条件2: 基金未到期
        df['fund_maturitydate_2'] = pd.to_datetime(df['fund_maturitydate_2'], errors='coerce')
        mask2 = (df['fund_maturitydate_2'].isna()) | (df['fund_maturitydate_2'] > target_date)

        # 条件3: 不是摊余成本法
        mask3 = df['fund_valuationmethod'] != '摊余成本法'

        # 条件4: 不是定期开放基金
        mask4 = df['fund_regulopenfundornot'] != '是'

        # 条件5: 是初始基金
        mask5 = df['fund_initial'] == '是'

        # 合并条件
        mask = mask1 & mask2 & mask3 & mask4 & mask5

        return df[mask].copy()

    def get_fund_pool(self, target_date, fund_type='all'):
        """
        获取指定日期的基金池

        参数:
        target_date: 目标日期 (datetime 或 str 'YYYY-MM-DD')
        fund_type: 基金类型 ('all', 'short', 'medium_long')

        返回:
        dict: 包含短期和中长期基金的字典
        """
        result = {}

        if fund_type in ['all', 'short']:
            short_df = self.read_fund_data(self.short_term_path)
            short_filtered = self.filter_funds(short_df, target_date)
            result['short'] = short_filtered

        if fund_type in ['all', 'medium_long']:
            medium_long_df = self.read_fund_data(self.medium_long_term_path)
            medium_long_filtered = self.filter_funds(medium_long_df, target_date)
            result['medium_long'] = medium_long_filtered

        return result


if __name__ == '__main__':
    # 测试
    preprocessor = FundDataPreprocessor()

    # 测试获取最新日期的基金池
    test_date = datetime.now()
    print(f"测试日期: {test_date}")

    fund_pool = preprocessor.get_fund_pool(test_date)

    print(f"\n短期纯债基金数量: {len(fund_pool['short'])}")
    print(f"中长期纯债基金数量: {len(fund_pool['medium_long'])}")

    print("\n短期基金样例:")
    print(fund_pool['short'].head())

    print("\n中长期基金样例:")
    print(fund_pool['medium_long'].head())
