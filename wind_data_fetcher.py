"""
Wind数据获取模块
功能：获取基金后复权净值数据
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from WindPy import w
import sys

# 设置UTF-8编码输出（兼容notebook环境）
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass


class WindDataFetcher:
    """Wind数据获取类"""

    def __init__(self):
        """初始化Wind连接"""
        self.connected = False
        self.connect()

    def connect(self):
        """连接Wind"""
        try:
            w.start()
            self.connected = True
            print("Wind连接成功")
        except Exception as e:
            print(f"Wind连接失败: {e}")
            self.connected = False

    def get_fund_nav(self, fund_code, start_date, end_date):
        """
        获取基金后复权净值数据

        参数:
        fund_code: 基金代码
        start_date: 开始日期 'YYYY-MM-DD'
        end_date: 结束日期 'YYYY-MM-DD'

        返回:
        DataFrame: 日期为索引，NAV为列的净值数据
        """
        if not self.connected:
            print("Wind未连接，尝试重新连接...")
            self.connect()
            if not self.connected:
                return None

        try:
            data = w.wsd(fund_code, "nav", start_date, end_date, "PriceAdj=B")

            if data.ErrorCode != 0:
                print(f"获取基金 {fund_code} 数据失败: ErrorCode={data.ErrorCode}")
                return None

            df = pd.DataFrame(data.Data[0], index=data.Times, columns=['NAV'])
            df.index = pd.to_datetime(df.index)
            return df

        except Exception as e:
            print(f"获取基金 {fund_code} 数据时出错: {e}")
            return None

    def get_fund_nav_smoothed(self, fund_code, start_date, end_date, window=5):
        """
        获取平滑后的基金净值数据

        参数:
        fund_code: 基金代码
        start_date: 开始日期 'YYYY-MM-DD'
        end_date: 结束日期 'YYYY-MM-DD'
        window: 平滑窗口

        返回:
        DataFrame: 平滑后的净值数据
        """
        df = self.get_fund_nav(fund_code, start_date, end_date)
        if df is None:
            return None

        # 计算移动平均
        df['NAV_smooth'] = df['NAV'].rolling(window=window, min_periods=1).mean()

        # 计算日度收益率
        df['return'] = df['NAV_smooth'].pct_change()

        return df

    def get_funds_nav_batch(self, fund_codes, start_date, end_date, window=5):
        """
        批量获取基金净值数据

        参数:
        fund_codes: 基金代码列表
        start_date: 开始日期 'YYYY-MM-DD'
        end_date: 结束日期 'YYYY-MM-DD'
        window: 平滑窗口

        返回:
        dict: {fund_code: DataFrame}
        """
        result = {}
        total = len(fund_codes)

        for i, fund_code in enumerate(fund_codes):
            if (i + 1) % 10 == 0:
                print(f"进度: {i + 1}/{total}")

            df = self.get_fund_nav_smoothed(fund_code, start_date, end_date, window)
            if df is not None:
                # 移除收益率中的异常值
                df = self._remove_outliers(df)
                result[fund_code] = df

        return result

    def _get_latest_rpt_date(self, target_date):
        """
        获取不超过target_date的最近季度末日期（用于Wind wss的rptDate参数）

        返回:
        str: 格式 'YYYYMMDD'，如 '20251231'；无法确定时返回 None
        """
        target_dt = pd.to_datetime(target_date)
        year = target_dt.year

        candidates = []
        for y in [year - 1, year]:
            for month, day in [(3, 31), (6, 30), (9, 30), (12, 31)]:
                d = pd.Timestamp(y, month, day)
                if d <= target_dt:
                    candidates.append(d)

        if not candidates:
            return None

        return max(candidates).strftime('%Y%m%d')

    def get_fund_reported_duration(self, fund_code, target_date):
        """
        通过Wind wss获取基金最近一期季报披露的组合久期

        参数:
        fund_code: 基金代码，如 '000015.OF'
        target_date: 目标日期 'YYYY-MM-DD'

        返回:
        float: 披露久期（年）；Wind返回错误或数据缺失时返回 None
        """
        if not self.connected:
            return None

        rpt_date = self._get_latest_rpt_date(target_date)
        if rpt_date is None:
            return None

        try:
            data = w.wss(fund_code, "risk_duration", f"rptDate={rpt_date}")
            if data.ErrorCode != 0:
                return None
            if not data.Data or not data.Data[0]:
                return None
            value = data.Data[0][0]
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return None
            return float(value)
        except Exception:
            return None

    def _remove_outliers(self, df, threshold=3.0):
        """
        移除异常值

        参数:
        df: 净值数据DataFrame
        threshold: 异常值阈值（倍数）

        返回:
        DataFrame: 移除异常值后的数据
        """
        # 计算收益率的标准差
        return_std = df['return'].std()

        # 如果波动超过threshold倍标准差，则设为NaN
        df.loc[df['return'].abs() > threshold * return_std, 'return'] = np.nan

        return df


if __name__ == '__main__':
    # 测试
    fetcher = WindDataFetcher()

    # 测试获取单个基金数据
    test_code = '000033.OF'
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')

    print(f"获取基金 {test_code} 从 {start_date} 到 {end_date} 的净值数据...")
    df = fetcher.get_fund_nav_smoothed(test_code, start_date, end_date)

    if df is not None:
        print(f"成功获取 {len(df)} 条数据")
        print(df.head(10))
        print(df.tail(10))
