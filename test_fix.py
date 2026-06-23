"""
测试修复后的_constrained_wls方法
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 导入自定义模块
from data_preprocessing import FundDataPreprocessor
from fund_type_classifier import FundTypeClassifier
from wind_data_fetcher import WindDataFetcher
from bond_index_data import BondIndexDataProcessor
from duration_model import DurationModel

print('='*60)
print('测试修复后的_constrained_wls方法')
print('='*60)

# 设置目标日期
target_date = '2026-06-12'

# 初始化各模块
preprocessor = FundDataPreprocessor(
    short_term_path='短期纯债基金样本数据.xlsx',
    medium_long_term_path='中长期纯债基金样本数据.xlsx'
)

classifier = FundTypeClassifier(holdings_path='纯债基金持仓情况.xlsx')
classifier.load_holdings_data()

wind_fetcher = WindDataFetcher()

index_processor = BondIndexDataProcessor(index_path='中债财富指数.xlsx')
index_processor.load_price_data()
index_processor.load_duration_data()

# 获取测试基金
fund_pool = preprocessor.get_fund_pool(target_date)
fund_df_short = fund_pool.get('short', pd.DataFrame())

print(f'\n短期基金池数量: {len(fund_df_short)}')

# 测试前5只基金
results = []

for idx, row in fund_df_short.head(5).iterrows():
    fund_code = row['Code']
    fund_name = row['Name']

    print(f'\n测试基金: {fund_code} - {fund_name}')

    # 获取基金类型和指数
    fund_bond_type = classifier.get_fund_type(fund_code, target_date)
    index_codes = index_processor.short_credit_indices

    # 获取净值数据
    start_date_calc = (pd.to_datetime(target_date) - pd.Timedelta(days=90)).strftime('%Y-%m-%d')
    fund_nav_df = wind_fetcher.get_fund_nav_smoothed(fund_code, start_date_calc, target_date)

    if fund_nav_df is None:
        print('  无法获取净值数据')
        continue

    # 获取回归数据
    end_date = pd.to_datetime(target_date)
    start_date = end_date - pd.Timedelta(days=60)
    fund_returns = fund_nav_df['return'].loc[start_date:end_date].dropna()
    fund_returns = fund_returns.iloc[-30:]

    if len(fund_returns) < 30:
        print(f'  数据不足: {len(fund_returns)}')
        continue

    # 获取指数收益率
    index_prices = index_processor.get_index_prices(
        index_codes,
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d')
    )

    index_returns = index_prices.pct_change().dropna()

    # 对齐日期
    common_dates = fund_returns.index.intersection(index_returns.index)
    fund_returns_aligned = fund_returns.loc[common_dates]
    index_returns_aligned = index_returns.loc[common_dates]

    # Lasso筛选
    model = DurationModel(index_processor)
    selected_factors = model._lasso_select_factors(fund_returns_aligned, index_returns_aligned)

    if len(selected_factors) == 0:
        print('  没有选中任何因子')
        continue

    print(f'  选中的因子: {selected_factors}')

    index_returns_selected = index_returns_aligned[selected_factors]

    # 生成时间权重
    time_weights = model._get_time_weights(len(fund_returns_aligned))

    # 调用修复后的_constrained_wls
    coefficients = model._constrained_wls(
        fund_returns_aligned,
        index_returns_selected,
        time_weights
    )

    if coefficients is None:
        print('  回归失败')
        continue

    # 计算久期
    duration = model.calculate_fund_duration(fund_nav_df, index_codes, target_date)

    print(f'  回归系数:')
    total_weight = 0
    for factor, weight in coefficients.items():
        print(f'    {factor}: {weight:.4f}')
        total_weight += weight
    print(f'  总权重: {total_weight:.4f}')
    print(f'  久期: {duration:.2f}' if duration is not None else '  久期: 计算失败')

    results.append({
        'code': fund_code,
        'name': fund_name,
        'coefficients': coefficients,
        'total_weight': total_weight,
        'duration': duration
    })

# 总结
print('\n' + '='*60)
print('测试总结')
print('='*60)

print(f'\n成功计算: {len(results)}/{min(5, len(fund_df_short))} 只基金')

if results:
    print('\n各基金权重分布:')
    for r in results:
        print(f"  {r['code']} - {r['name']}: 总权重={r['total_weight']:.4f}, 久期={r['duration']:.2f if r['duration'] else 'N/A'}")

print('\n测试完成！')
