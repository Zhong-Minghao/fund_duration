"""
通用工具模块
"""


def smooth_series(data, window=5, min_periods=1):
    """
    对价格/净值数据做滚动均值平滑，Series（如单只基金净值）和 DataFrame
    （如多个指数组成的价格矩阵）均适用。

    min_periods=1 保证平滑不产生额外的前导 NaN、不改变序列长度/索引，
    这是下游按日期对齐（Index.intersection / join(how='inner').dropna()）
    不需要额外处理长度变化的前提。

    参数:
    data: pd.Series 或 pd.DataFrame，按日期升序索引的原始价格/净值数据
    window: 滚动窗口大小（交易日数），默认5
    min_periods: 窗口内最少可用观测数，默认1

    返回:
    与 data 同类型：滚动均值平滑后的结果，索引不变
    """
    return data.rolling(window=window, min_periods=min_periods).mean()
