"""
诊断和修复缓存文件问题
"""

import pandas as pd
import numpy as np
from pathlib import Path

def check_cache_file(fund_code, target_date='2025-09-30'):
    """检查单个缓存文件"""
    cache_path = Path('data/nav') / f'{fund_code}.pkl'
    if not cache_path.exists():
        return None

    try:
        df = pd.read_pickle(cache_path)
        target = pd.to_datetime(target_date)
        start = (target - pd.Timedelta(days=90))

        # 检查覆盖
        has_coverage = df.index.min() <= start and df.index.max() >= target

        # 检查目标日期附近的数据
        nearby = df.loc[(df.index >= start) & (df.index <= target)]

        return {
            'fund_code': fund_code,
            'cache_min': df.index.min(),
            'cache_max': df.index.max(),
            'has_coverage': has_coverage,
            'nearby_count': len(nearby),
            'target_in_cache': target in df.index,
            'last_dates': df.index[-5:].tolist()
        }
    except Exception as e:
        return {'fund_code': fund_code, 'error': str(e)}

# 测试几只失败的基金
test_funds = ['000084.OF', '000089.OF', '000128.OF', '000808.OF']

print("=" * 60)
print("诊断缓存文件问题")
print("=" * 60)

for fund in test_funds:
    result = check_cache_file(fund)
    if result:
        print(f"\n{fund}:")
        if 'error' in result:
            print(f"  错误: {result['error']}")
        else:
            print(f"  缓存范围: {result['cache_min']} 到 {result['cache_max']}")
            print(f"  覆盖2025-09-30前90天: {result['has_coverage']}")
            print(f"  目标日期附近数据量: {result['nearby_count']}")
            print(f"  最近5个日期: {result['last_dates']}")

# 检查 pandas 版本和问题
print("\n" + "=" * 60)
print("检查 pandas pickle 兼容性问题")
print("=" * 60)

import pickle

try:
    with open('data/nav/000089.OF.pkl', 'rb') as f:
        # 使用 pickle 直接加载
        data = pickle.load(f)
        print(f"直接 pickle 加载成功")
        print(f"数据类型: {type(data)}")
        print(f"索引类型: {type(data.index)}")
        print(f"索引 dtype: {data.index.dtype}")

        # 尝试转换索引
        if hasattr(data.index, 'astype'):
            try:
                data.index = data.index.astype('datetime64[ns]')
                print(f"索引转换为 datetime64[ns] 成功")
            except Exception as e:
                print(f"索引转换失败: {e}")

        # 检查数据范围
        print(f"数据范围: {data.index.min()} 到 {data.index.max()}")

        # 检查 2025-09-30
        target = pd.Timestamp('2025-09-30')
        print(f"2025-09-30 在索引中: {target in data.index}")

        # 检查范围覆盖
        start = pd.Timestamp('2025-07-02')  # 90天前
        mask = (data.index >= start) & (data.index <= target)
        print(f"2025-07-02 到 2025-09-30 范围内有 {mask.sum()} 条数据")

except Exception as e:
    print(f"直接 pickle 加载失败: {e}")
