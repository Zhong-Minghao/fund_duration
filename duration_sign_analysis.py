"""
符号分类久期分析模块
功能：基于基金收益率与指数综合收益率的符号关系分类分析久期
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass


class SignBasedDurationAnalyzer:
    """基于符号分类的久期分析器"""

    def __init__(self, duration_model, index_processor, wind_fetcher):
        """
        初始化

        参数:
        duration_model: DurationModel实例
        index_processor: BondIndexDataProcessor实例
        wind_fetcher: WindDataFetcher实例
        """
        self.duration_model = duration_model
        self.index_processor = index_processor
        self.wind_fetcher = wind_fetcher

    def calculate_weighted_index_return(self, index_returns, coefficients):
        """
        计算加权平均指数收益率

        参数:
            index_returns: DataFrame, 指数收益率（列为指数代码）
            coefficients: dict, {指数代码: 权重}

        返回:
            Series: 加权平均收益率，索引与index_returns相同
        """
        weighted_returns = pd.Series(0.0, index=index_returns.index)

        for code, weight in coefficients.items():
            if code in index_returns.columns and weight > 0:
                weighted_returns += index_returns[code] * weight

        return weighted_returns

    def classify_observations_by_sign(self, fund_returns, weighted_index_returns):
        """
        按符号关系分类观测点

        参数:
            fund_returns: Series, 基金收益率
            weighted_index_returns: Series, 加权平均指数收益率

        返回:
            dict: {
                'same_positive': Series,  # 同正：都>0
                'same_negative': Series,  # 同负：都<0
                'opposite': Series,       # 反向：一正一负
            }
            每个Series的值为基金收益率
        """
        # 确保索引对齐
        aligned = pd.DataFrame({
            'fund': fund_returns,
            'weighted_idx': weighted_index_returns
        }).dropna()

        if aligned.empty:
            return {
                'same_positive': pd.Series(dtype=float),
                'same_negative': pd.Series(dtype=float),
                'opposite': pd.Series(dtype=float)
            }

        # 分类
        same_positive_mask = (aligned['fund'] > 0) & (aligned['weighted_idx'] > 0)
        same_negative_mask = (aligned['fund'] < 0) & (aligned['weighted_idx'] < 0)
        opposite_mask = ~same_positive_mask & ~same_negative_mask

        result = {
            'same_positive': aligned.loc[same_positive_mask, 'fund'],
            'same_negative': aligned.loc[same_negative_mask, 'fund'],
            'opposite': aligned.loc[opposite_mask, 'fund']
        }

        return result

    def calculate_duration_for_subset(self, fund_nav_df, index_codes, target_date,
                                     date_filter=None, reported_duration=None,
                                     fund_code=None, verbose=False):
        """
        对指定日期子集计算久期

        参数:
            fund_nav_df: DataFrame, 基金净值数据（含return列）
            index_codes: list, 指数代码列表
            target_date: str, 目标日期
            date_filter: Series or list, 需要保留的日期索引
            reported_duration: float, Wind披露久期（用于Lasso单因子退化时锚定）
            fund_code: str, 基金代码
            verbose: bool, 是否输出详细日志

        返回:
            tuple: (duration, coefficients_dict, n_obs)
        """
        # 获取回归窗口
        end_date = pd.to_datetime(target_date)
        start_date = end_date - pd.Timedelta(days=120)  # 多取一些确保有足够交易日

        # 获取基金收益率
        fund_returns = fund_nav_df['return'].loc[start_date:end_date].dropna()

        if len(fund_returns) < self.duration_model.window:
            if verbose:
                print(f"  [警告] {fund_code} 数据点不足: {len(fund_returns)} < {self.duration_model.window}")
            return None, None, 0

        # 应用日期过滤
        if date_filter is not None:
            if isinstance(date_filter, pd.Series):
                valid_dates = date_filter.index.intersection(fund_returns.index)
            elif isinstance(date_filter, (list, pd.Index)):
                valid_dates = pd.Index(date_filter).intersection(fund_returns.index)
            else:
                valid_dates = fund_returns.index

            fund_returns = fund_returns.loc[valid_dates]

        if len(fund_returns) < 5:  # 至少需要5个观测点
            if verbose:
                print(f"  [警告] {fund_code} 过滤后数据点不足: {len(fund_returns)}")
            return None, None, len(fund_returns)

        # 获取指数收益率
        index_prices = self.index_processor.get_index_prices_smoothed(
            index_codes,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d'),
            window=self.duration_model.index_smooth_window
        )

        if index_prices.empty:
            if verbose:
                print(f"  [警告] {fund_code} 指数数据为空")
            return None, None, 0

        index_returns = index_prices.pct_change().dropna()

        # 对齐日期
        common_dates = fund_returns.index.intersection(index_returns.index)
        if len(common_dates) < 5:
            if verbose:
                print(f"  [警告] {fund_code} 对齐后数据点不足: {len(common_dates)}")
            return None, None, 0

        fund_returns = fund_returns.loc[common_dates]
        index_returns = index_returns.loc[common_dates]

        # Lasso筛选因子
        selected_factors = self.duration_model._lasso_select_factors(fund_returns, index_returns)

        # 单因子退化兜底
        if len(selected_factors) <= 1 and reported_duration is not None:
            selected_factors = self.duration_model._anchor_factor_by_duration(
                selected_factors, index_codes, reported_duration, target_date,
                fund_code=fund_code
            )

        # 带约束的WLS（但禁用边界迭代，因为子样本数据少）
        # 这里直接调用底层WLS，不触发迭代
        aligned_data = pd.DataFrame({
            'fund': fund_returns
        }).join(index_returns[selected_factors], how='inner').dropna()

        if aligned_data.shape[0] < len(selected_factors):
            if verbose:
                print(f"  [警告] {fund_code} 观测数 < 因子数")
            return None, None, aligned_data.shape[0]

        # 剔除离群点
        fund_clean, idx_clean = self.duration_model._remove_regression_outliers(
            aligned_data['fund'], aligned_data.iloc[:, 1:]
        )

        X = idx_clean.values
        y = fund_clean.values
        weights = self.duration_model._get_time_weights(len(y))

        intercept, coefficients = self.duration_model._solve_qp_osqp(X, y, weights)

        if coefficients is None:
            if verbose:
                print(f"  [警告] {fund_code} WLS求解失败")
            return None, None, len(y)

        # 计算久期
        coef_dict = dict(zip(selected_factors, coefficients))
        total_duration = 0
        for code, weight in coef_dict.items():
            idx_dur = self.index_processor.get_latest_duration(code, target_date)
            if idx_dur is not None and not np.isnan(idx_dur):
                total_duration += weight * idx_dur

        return total_duration, coef_dict, len(y)

    def calculate_duration_by_sign_category(self, fund_nav_df, index_codes, target_date,
                                            reported_duration=None, fund_code=None,
                                            extended_window=60, verbose=False):
        """
        按符号分类分别计算久期

        参数:
            fund_nav_df: DataFrame, 基金净值数据（含return列）
            index_codes: list, 指数代码列表
            target_date: str, 目标日期 'YYYY-MM-DD'
            reported_duration: float, Wind披露的实际久期
            fund_code: str, 基金代码
            extended_window: int, 扩展窗口的交易日数（默认60天）
            verbose: bool, 是否输出详细日志

        返回:
            dict: {
                'same_positive': {'duration': float, 'count': int, 'coefficients': dict},
                'same_negative': {'duration': float, 'count': int, 'coefficients': dict},
                'opposite': {'duration': float, 'count': int, 'coefficients': dict},
                'full_sample': {'duration': float, 'count': int, 'coefficients': dict},
                'reported_duration': float,
                'bias_to_reported': dict
            }
        """
        # 获取扩展窗口数据
        end_date = pd.to_datetime(target_date)
        start_date = end_date - pd.Timedelta(days=180)  # 足够长以获取extended_window个交易日

        fund_returns_ext = fund_nav_df['return'].loc[start_date:end_date].dropna()

        if len(fund_returns_ext) < extended_window:
            if verbose:
                print(f"  [警告] {fund_code} 扩展窗口数据不足: {len(fund_returns_ext)} < {extended_window}")
            return None

        # 只使用最近extended_window个交易日
        fund_returns_ext = fund_returns_ext.iloc[-extended_window:]

        # 获取指数收益率
        index_prices = self.index_processor.get_index_prices_smoothed(
            index_codes,
            fund_returns_ext.index[0].strftime('%Y-%m-%d'),
            fund_returns_ext.index[-1].strftime('%Y-%m-%d'),
            window=self.duration_model.index_smooth_window
        )

        if index_prices.empty:
            if verbose:
                print(f"  [警告] {fund_code} 指数数据为空")
            return None

        index_returns_ext = index_prices.pct_change().dropna()

        # 对齐
        common_dates = fund_returns_ext.index.intersection(index_returns_ext.index)
        fund_returns_ext = fund_returns_ext.loc[common_dates]
        index_returns_ext = index_returns_ext.loc[common_dates]

        if len(fund_returns_ext) < self.duration_model.window:
            if verbose:
                print(f"  [警告] {fund_code} 对齐后数据不足: {len(fund_returns_ext)}")
            return None

        # 步骤1：用全样本拟合得到权重
        if verbose:
            print(f"  [步骤1] {fund_code} 全样本拟合...")

        full_duration, full_coefs, full_n = self.calculate_duration_for_subset(
            fund_nav_df, index_codes, target_date,
            date_filter=common_dates,
            reported_duration=reported_duration,
            fund_code=fund_code,
            verbose=False
        )

        if full_coefs is None:
            if verbose:
                print(f"  [警告] {fund_code} 全样本拟合失败")
            return None

        # 步骤2：计算加权指数收益率
        weighted_idx_returns = self.calculate_weighted_index_return(
            index_returns_ext, full_coefs
        )

        # 步骤3：分类
        classified = self.classify_observations_by_sign(fund_returns_ext, weighted_idx_returns)

        if verbose:
            print(f"  [分类] {fund_code} 同正:{len(classified['same_positive'])} "
                  f"同负:{len(classified['same_negative'])} 反向:{len(classified['opposite'])}")

        # 步骤4：分别计算久期
        results = {
            'same_positive': {'duration': None, 'count': len(classified['same_positive']), 'coefficients': None},
            'same_negative': {'duration': None, 'count': len(classified['same_negative']), 'coefficients': None},
            'opposite': {'duration': None, 'count': len(classified['opposite']), 'coefficients': None},
            'full_sample': {'duration': full_duration, 'count': full_n, 'coefficients': full_coefs},
            'reported_duration': reported_duration,
            'bias_to_reported': {}
        }

        # 对每个分类计算久期
        for category, returns_series in classified.items():
            if len(returns_series) >= 5:  # 至少5个观测点
                duration, coefs, n = self.calculate_duration_for_subset(
                    fund_nav_df, index_codes, target_date,
                    date_filter=returns_series.index,
                    reported_duration=reported_duration,
                    fund_code=fund_code,
                    verbose=False
                )
                results[category]['duration'] = duration
                results[category]['coefficients'] = coefs
                results[category]['count'] = n
            else:
                if verbose:
                    print(f"  [跳过] {fund_code} {category} 观测点不足: {len(returns_series)}")

        # 计算偏差
        if reported_duration is not None:
            for cat in ['same_positive', 'same_negative', 'opposite', 'full_sample']:
                dur = results[cat]['duration']
                if dur is not None:
                    results['bias_to_reported'][cat] = dur - reported_duration

        return results

    def analyze_single_fund(self, fund_code, fund_nav_df, index_codes, target_date,
                           reported_duration=None, verbose=True):
        """
        分析单只基金（便捷方法）

        参数:
            fund_code: str, 基金代码
            fund_nav_df: DataFrame, 基金净值数据
            index_codes: list, 指数代码列表
            target_date: str, 目标日期
            reported_duration: float, Wind披露久期
            verbose: bool, 是否输出日志

        返回:
            dict: 分析结果
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"分析基金: {fund_code}")
            print(f"目标日期: {target_date}")
            print(f"{'='*60}")

        results = self.calculate_duration_by_sign_category(
            fund_nav_df, index_codes, target_date,
            reported_duration=reported_duration,
            fund_code=fund_code,
            verbose=verbose
        )

        if results and verbose:
            print(f"\n结果汇总:")
            print(f"  Wind披露久期: {results['reported_duration']:.3f} 年" if results['reported_duration'] else "  Wind披露久期: 未知")
            for cat in ['full_sample', 'same_positive', 'same_negative', 'opposite']:
                dur = results[cat]['duration']
                count = results[cat]['count']
                bias = results['bias_to_reported'].get(cat)
                if dur is not None:
                    bias_str = f" (偏差: {bias:+.3f})" if bias is not None else ""
                    print(f"  {cat:20s}: {dur:6.3f} 年 ({count} 个观测点){bias_str}")

        return results

    def batch_analyze(self, fund_pool, target_date, calculator, max_funds=None,
                     output_file=None, verbose=True):
        """
        批量分析基金

        参数:
            fund_pool: dict, 基金池（来自preprocessor.get_fund_pool()）
            target_date: str, 目标日期
            calculator: FundDurationCalculator实例
            max_funds: int, 最大分析基金数（用于测试）
            output_file: str, 输出Excel文件路径
            verbose: bool, 是否输出日志

        返回:
            DataFrame: 批量分析结果
        """
        all_results = []
        processed = 0
        success = 0

        for fund_type in ['short', 'medium_long']:
            fund_df = fund_pool.get(fund_type, pd.DataFrame())

            for idx, row in fund_df.iterrows():
                if max_funds and processed >= max_funds:
                    break

                fund_code = row['Code']
                fund_name = row['Name']
                processed += 1

                if verbose and processed % 10 == 0:
                    print(f"进度: {processed}/{max_funds or '全部'} | 成功: {success}")

                try:
                    # 判断基金类型
                    fund_bond_type = calculator.fund_classifier.get_fund_type(fund_code, target_date)
                    if fund_bond_type == 'rate':
                        index_codes = (calculator.index_processor.short_rate_indices
                                     if fund_type == 'short'
                                     else calculator.index_processor.medium_long_rate_indices)
                    elif fund_bond_type == 'credit':
                        index_codes = (calculator.index_processor.short_credit_indices
                                     if fund_type == 'short'
                                     else calculator.index_processor.medium_long_credit_indices)
                    else:
                        continue

                    # 获取净值数据
                    start_date = (pd.to_datetime(target_date) - pd.Timedelta(days=180)).strftime('%Y-%m-%d')
                    fund_nav_df = calculator.wind_fetcher.get_fund_nav_smoothed(
                        fund_code, start_date, target_date
                    )

                    if fund_nav_df is None:
                        continue

                    # 获取Wind披露久期
                    reported_duration = calculator.wind_fetcher.get_fund_reported_duration(
                        fund_code, target_date
                    )

                    # 分析
                    result = self.calculate_duration_by_sign_category(
                        fund_nav_df, index_codes, target_date,
                        reported_duration=reported_duration,
                        fund_code=fund_code,
                        verbose=False
                    )

                    if result is None:
                        continue

                    # 汇总结果
                    summary = {
                        'fund_code': fund_code,
                        'fund_name': fund_name,
                        'fund_type': fund_type,
                        'bond_type': fund_bond_type,
                        'reported_duration': result['reported_duration'],
                        'full_sample_duration': result['full_sample']['duration'],
                        'same_positive_duration': result['same_positive']['duration'],
                        'same_negative_duration': result['same_negative']['duration'],
                        'opposite_duration': result['opposite']['duration'],
                        'same_positive_count': result['same_positive']['count'],
                        'same_negative_count': result['same_negative']['count'],
                        'opposite_count': result['opposite']['count'],
                    }

                    # 添加偏差
                    for cat in ['full_sample', 'same_positive', 'same_negative', 'opposite']:
                        bias = result['bias_to_reported'].get(cat)
                        summary[f'{cat}_bias'] = bias

                    all_results.append(summary)
                    success += 1

                except Exception as e:
                    if verbose:
                        print(f"  [错误] {fund_code}: {str(e)[:50]}")
                    continue

            if max_funds and processed >= max_funds:
                break

        # 转为DataFrame
        results_df = pd.DataFrame(all_results)

        if not results_df.empty and output_file:
            results_df.to_excel(output_file, index=False)
            if verbose:
                print(f"\n结果已保存至: {output_file}")

        return results_df


if __name__ == '__main__':
    # 测试代码
    print("符号分类久期分析模块加载成功")
