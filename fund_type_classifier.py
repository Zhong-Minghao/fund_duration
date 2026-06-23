"""
基金类型判断模块
功能：根据持仓情况表判断基金是利率债基金还是信用债基金
"""

import pandas as pd
import numpy as np
from datetime import datetime
import re
import sys

# 设置UTF-8编码输出（兼容notebook环境）
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass


class FundTypeClassifier:
    """基金类型判断类"""

    def __init__(self, holdings_path='纯债基金持仓情况.xlsx'):
        """
        初始化

        参数:
        holdings_path: 持仓情况表路径
        """
        self.holdings_path = holdings_path
        self.holdings_df = None
        self.quarter_map = self._build_quarter_map()

    def _build_quarter_map(self):
        """
        构建季度映射

        返回:
        dict: 列索引到季度信息的映射
        """
        df_raw = pd.read_excel(self.holdings_path, header=None)

        quarter_map = {}
        for idx in range(2, df_raw.shape[1]):  # 从第3列开始（索引2）
            chinese_name = df_raw.iloc[0, idx]
            english_name = df_raw.iloc[1, idx]

            # 解析季度信息
            if pd.notna(chinese_name):
                # 提取年份和季度
                match = re.search(r'(\d{4})(?:一|二|三|四)季报', str(chinese_name))
                if match:
                    year = int(match.group(1))
                    quarter_str = chinese_name

                    # 判断是政金债还是国债
                    if '政策性金融债' in chinese_name:
                        bond_type = 'pfb'
                    elif '国债' in chinese_name:
                        bond_type = 'gov'
                    else:
                        continue

                    if idx not in quarter_map:
                        quarter_map[idx] = {}
                    quarter_map[idx] = {
                        'year': year,
                        'quarter_str': quarter_str,
                        'bond_type': bond_type
                    }

        return quarter_map

    def load_holdings_data(self):
        """加载持仓数据"""
        self.holdings_df = pd.read_excel(self.holdings_path, header=1)
        return self.holdings_df

    def get_quarter_date(self, year, quarter):
        """
        获取季度对应的日期（季度末）

        参数:
        year: 年份
        quarter: 季度 (1, 2, 3, 4)

        返回:
        datetime: 季度末日期
        """
        quarter_end_dates = {
            1: f'{year}-03-31',
            2: f'{year}-06-30',
            3: f'{year}-09-30',
            4: f'{year}-12-31'
        }
        return pd.to_datetime(quarter_end_dates[quarter])

    def parse_quarter_from_str(self, quarter_str):
        """
        从字符串解析季度信息

        参数:
        quarter_str: 季度字符串，如 '2024一季报'

        返回:
        tuple: (year, quarter)
        """
        quarter_map = {'一': 1, '二': 2, '三': 3, '四': 4}
        match = re.search(r'(\d{4})(一|二|三|四)季报', str(quarter_str))
        if match:
            year = int(match.group(1))
            quarter = quarter_map[match.group(2)]
            return year, quarter
        return None, None

    def get_fund_type(self, fund_code, target_date):
        """
        判断基金在指定日期的类型（利率债或信用债）

        判断标准: 政金债占比 + 国债占比 > 80% 为利率债，否则为信用债

        参数:
        fund_code: 基金代码
        target_date: 目标日期

        返回:
        str: 'rate' (利率债) 或 'credit' (信用债)
        """
        if self.holdings_df is None:
            self.load_holdings_data()

        # 找到该基金
        fund_row = self.holdings_df[self.holdings_df['Code'] == fund_code]
        if fund_row.empty:
            return None

        # 确定使用哪个季度的数据
        # 目标日期对应的最近的已披露季度
        # 例如：2024-05-15 应该使用 2024一季报（3月31日披露）的数据
        target_dt = pd.to_datetime(target_date)

        # 找到所有可用的季度
        available_quarters = []
        for idx, info in self.quarter_map.items():
            year, quarter = self.parse_quarter_from_str(info['quarter_str'])
            quarter_date = self.get_quarter_date(year, quarter)
            if quarter_date <= target_dt:
                available_quarters.append((quarter_date, idx))

        if not available_quarters:
            return None

        # 使用最近的季度
        latest_quarter_date, latest_idx = max(available_quarters, key=lambda x: x[0])

        # 获取政金债占比和国债占比
        pfb_ratio = fund_row.iloc[0, latest_idx]
        # 国债列的索引是政金债列索引 + 32（因为政金债有32列）
        gov_idx = latest_idx + 32
        if gov_idx < len(fund_row.columns):
            gov_ratio = fund_row.iloc[0, gov_idx]
        else:
            gov_ratio = 0

        # 空值设为0
        pfb_ratio = pfb_ratio if pd.notna(pfb_ratio) else 0
        gov_ratio = gov_ratio if pd.notna(gov_ratio) else 0

        # 判断类型
        total_ratio = pfb_ratio + gov_ratio
        return 'rate' if total_ratio > 80 else 'credit'

    def get_fund_types_batch(self, fund_codes, target_date):
        """
        批量判断基金类型

        参数:
        fund_codes: 基金代码列表
        target_date: 目标日期

        返回:
        dict: {fund_code: fund_type}
        """
        if self.holdings_df is None:
            self.load_holdings_data()

        result = {}
        for fund_code in fund_codes:
            result[fund_code] = self.get_fund_type(fund_code, target_date)
        return result


if __name__ == '__main__':
    # 测试
    classifier = FundTypeClassifier()
    classifier.load_holdings_data()

    # 测试单个基金
    test_code = '000033.OF'
    test_date = '2024-12-20'

    fund_type = classifier.get_fund_type(test_code, test_date)
    print(f"基金 {test_code} 在 {test_date} 的类型: {fund_type}")

    # 测试季度映射
    print("\n季度列映射:")
    for idx, info in list(classifier.quarter_map.items())[:5]:
        print(f"列索引 {idx}: {info}")
