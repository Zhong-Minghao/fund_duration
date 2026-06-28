"""
久期测算模型
功能：使用Lasso回归和带约束的WLS测算基金久期
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from scipy.optimize import minimize
import osqp
import scipy.sparse as sp
import warnings
warnings.filterwarnings('ignore')


class DurationModel:
    """久期测算模型类"""

    def __init__(self,
                 index_processor,
                 window=15,
                 lasso_alpha=0.1,
                 min_lev=0.8,
                 max_lev=1.4):
        """
        初始化

        参数:
        index_processor: BondIndexDataProcessor实例
        window: 回归窗口（交易日数）
        lasso_alpha: Lasso正则化参数
        min_lev: 最小杠杆率
        max_lev: 最大杠杆率
        """
        self.index_processor = index_processor
        self.window = window
        self.lasso_alpha = lasso_alpha
        self.min_lev = min_lev
        self.max_lev = max_lev

    def update_index_processor(self, index_processor):
        """更新index_processor引用"""
        self.index_processor = index_processor

    def _get_time_weights(self, n):
        """
        生成时间权重（较近的数据权重更高）

        参数:
        n: 观测点数量

        返回:
        array: 权重数组，最新数据权重为1
        """
        # 线性递增权重，归一化使最新数据权重为1
        weights = np.arange(1, n + 1)
        return weights / n  # 等价于 weights / weights.max()

    def _lasso_select_factors(self, fund_returns, index_returns):
        """
        使用Lasso回归筛选因子

        参数:
        fund_returns: 基金收益率Series
        index_returns: 指数收益率DataFrame

        返回:
        list: 选中的因子列名
        """
        # 对齐数据
        aligned_data = pd.DataFrame({
            'fund': fund_returns
        }).join(index_returns, how='inner').dropna()

        if aligned_data.empty or aligned_data.shape[0] < 5:
            return index_returns.columns.tolist()

        X = aligned_data.iloc[:, 1:].values
        y = aligned_data['fund'].values

        # 标准化数据（重要！）
        from sklearn.preprocessing import StandardScaler
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_std = scaler_X.fit_transform(X)
        y_std = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

        # 使用Lasso回归
        lasso = Lasso(alpha=self.lasso_alpha, max_iter=10000)
        lasso.fit(X_std, y_std)

        # 选择系数不为0的因子
        selected_indices = np.where(lasso.coef_ != 0)[0]
        selected_factors = index_returns.columns[selected_indices].tolist()

        # 如果没有选中任何因子，返回所有因子
        if not selected_factors:
            selected_factors = index_returns.columns.tolist()

        return selected_factors

    def _solve_qp_osqp(self, X, y, weights, min_lev=None, max_lev=None):
        """
        使用OSQP求解带截距项的约束WLS

        参数:
            X: 设计矩阵 (n_obs, n_factors)
            y: 响应变量 (n_obs,)
            weights: 权重 (n_obs,)
            min_lev: 最小杠杆率
            max_lev: 最大杠杆率

        返回:
            tuple: (截距, 系数数组) 或 (None, None)
        """
        if min_lev is None:
            min_lev = self.min_lev
        if max_lev is None:
            max_lev = self.max_lev

        # 构造设计矩阵 Z = [1, X]
        Z = np.column_stack([np.ones(len(y)), X])
        n_params = Z.shape[1]  # = n_factors + 1
        n_factors = X.shape[1]

        # 构造权重矩阵
        W = np.diag(weights)

        # QP矩阵: P = 2 * Z' * W * Z
        P = 2 * Z.T @ W @ Z
        q_vec = -2 * Z.T @ W @ y

        # 约束矩阵 A
        # 行0: 截距无约束
        # 行1~n_factors: β的非负约束
        # 行 n_factors+1: 上限约束 Σβ_i <= max_lev
        # 行 n_factors+2: 下限约束 -Σβ_i <= -min_lev
        A_rows = 1 + n_factors + 2
        A = np.zeros((A_rows, n_params))

        A[0, 0] = 1  # 截距行
        A[1:n_factors+1, 1:] = -np.eye(n_factors)  # 非负约束
        A[n_factors+1, 1:] = 1  # 上限
        A[n_factors+2, 1:] = -1  # 下限

        # 约束边界
        INF = 1e10
        l = np.array([-INF] + [-INF] * n_factors + [-INF, -INF])
        u = np.array([INF] + [0] * n_factors + [max_lev, -min_lev])

        # 转为稀疏矩阵
        P_sparse = sp.csr_matrix(P)
        A_sparse = sp.csr_matrix(A)

        # 求解
        prob = osqp.OSQP()
        prob.setup(P=P_sparse, q=q_vec, A=A_sparse, l=l, u=u,
                   eps_abs=1e-9, eps_rel=1e-9, verbose=False)
        result = prob.solve()

        if result.info.status != 'solved':
            return None, None

        return result.x[0], result.x[1:]  # (截距, 系数)

    def _constrained_wls(self, fund_returns, index_returns, time_weights):
        """
        带约束的加权最小二乘法（使用OSQP求解器）

        参数:
            fund_returns: 基金收益率Series
            index_returns: 指数收益率DataFrame
            time_weights: 时间权重（已废弃，保留参数以兼容）

        返回:
            dict: 回归系数
        """
        # 对齐数据
        aligned_data = pd.DataFrame({
            'fund': fund_returns
        }).join(index_returns, how='inner').dropna()

        if aligned_data.shape[0] < len(index_returns.columns):
            return None

        X = aligned_data.iloc[:, 1:].values
        y = aligned_data['fund'].values

        n_factors = X.shape[1]
        n_obs = X.shape[0]

        # 根据对齐后的数据长度生成时间权重
        adjusted_weights = self._get_time_weights(n_obs)

        # 使用OSQP求解
        intercept, coefficients = self._solve_qp_osqp(X, y, adjusted_weights)

        if coefficients is None:
            # OSQP求解失败，使用等权作为后备
            equal_weight = (self.min_lev + self.max_lev) / 2 / n_factors
            final_params = np.full(n_factors, equal_weight)
        else:
            final_params = coefficients

        return dict(zip(index_returns.columns, final_params))

    def _anchor_factor_by_duration(self, selected_factors, all_index_codes,
                                   reported_duration, target_date, fund_code=None):
        """
        当Lasso只选出<=1个因子时，在候选指数池中找久期最近的指数作为锚定因子补充进来。

        参数:
        selected_factors: Lasso已选因子列表
        all_index_codes: 全部候选指数代码列表
        reported_duration: Wind披露的基金组合久期
        target_date: 目标日期
        fund_code: 基金代码（仅用于日志）

        返回:
        list: 可能扩充后的因子列表
        """
        best_code = None
        best_diff = float('inf')

        for code in all_index_codes:
            dur = self.index_processor.get_latest_duration(code, target_date)
            if dur is None or np.isnan(dur):
                continue
            diff = abs(dur - reported_duration)
            if diff < best_diff:
                best_diff = diff
                best_code = code

        if best_code is None:
            return selected_factors

        if best_code not in selected_factors:
            return selected_factors + [best_code]
        else:
            print(f"[警告] {fund_code} {target_date} 回归自变量只有1个指数 {best_code}，且该指数已是最近久期匹配，无额外因子可补充")
            return selected_factors

    def calculate_fund_duration(self, fund_nav_df, index_codes, target_date,
                                reported_duration=None, fund_code=None):
        """
        计算单只基金在目标日期的久期

        参数:
        fund_nav_df: 基金净值DataFrame（包含return列）
        index_codes: 待选指数代码列表
        target_date: 目标日期

        返回:
        float: 久期值
        """
        # 获取回归窗口
        end_date = pd.to_datetime(target_date)
        start_date = end_date - pd.Timedelta(days=60)  # 多取一些天确保有足够的交易日

        # 获取基金收益率
        fund_returns = fund_nav_df['return'].loc[start_date:end_date].dropna()

        if len(fund_returns) < self.window:
            return None

        # 只使用最近window个交易日
        fund_returns = fund_returns.iloc[-self.window:]

        # 获取指数收益率
        index_prices = self.index_processor.get_index_prices(
            index_codes,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )

        if index_prices.empty:
            return None

        index_returns = index_prices.pct_change().dropna()

        # 对齐基金和指数的日期
        common_dates = fund_returns.index.intersection(index_returns.index)
        if len(common_dates) < self.window:
            return None

        fund_returns = fund_returns.loc[common_dates]
        index_returns = index_returns.loc[common_dates]

        # Lasso筛选因子
        selected_factors = self._lasso_select_factors(fund_returns, index_returns)

        # 单因子退化兜底：Lasso<=1个因子时，用Wind披露久期锚定额外因子
        if len(selected_factors) <= 1 and reported_duration is not None:
            selected_factors = self._anchor_factor_by_duration(
                selected_factors, index_codes, reported_duration, target_date,
                fund_code=fund_code
            )

        index_returns_selected = index_returns[selected_factors]

        # 生成时间权重
        time_weights = self._get_time_weights(len(fund_returns))

        # 带约束的WLS
        coefficients = self._constrained_wls(fund_returns, index_returns_selected, time_weights)

        if coefficients is None:
            return None

        # 计算久期
        total_duration = 0
        total_weight = 0

        for factor_code, weight in coefficients.items():
            # 获取该指数的久期
            index_duration = self.index_processor.get_latest_duration(factor_code, target_date)

            if index_duration is not None and not np.isnan(index_duration):
                total_duration += weight * index_duration
                total_weight += weight

        if total_weight == 0:
            return None

        # 调整杠杆后的久期
        duration = total_duration / total_weight

        return duration


class FundDurationCalculator:
    """基金久期计算器"""

    def __init__(self,
                 data_preprocessor,
                 fund_classifier,
                 wind_fetcher,
                 index_processor):
        """
        初始化

        参数:
        data_preprocessor: FundDataPreprocessor实例
        fund_classifier: FundTypeClassifier实例
        wind_fetcher: WindDataFetcher实例
        index_processor: BondIndexDataProcessor实例
        """
        self.data_preprocessor = data_preprocessor
        self.fund_classifier = fund_classifier
        self.wind_fetcher = wind_fetcher
        self.index_processor = index_processor

        # 创建久期模型
        self.duration_model = DurationModel(index_processor)

    def calculate_fund_pool_duration(self, target_date):
        """
        计算基金池中所有基金的久期

        参数:
        target_date: 目标日期 'YYYY-MM-DD'

        返回:
        dict: {fund_code: duration}
        """
        # 获取基金池
        fund_pool = self.data_preprocessor.get_fund_pool(target_date)

        results = {}

        # 分别处理短期和中长期基金
        for fund_type, fund_df in fund_pool.items():
            print(f"\n处理{fund_type}基金...")

            # 确定使用的指数
            if fund_type == 'short':
                # 短期基金需要进一步判断是利率型还是信用型
                for idx, row in fund_df.iterrows():
                    fund_code = row['Code']

                    # 判断基金类型
                    fund_bond_type = self.fund_classifier.get_fund_type(fund_code, target_date)

                    if fund_bond_type == 'rate':
                        index_codes = self.index_processor.short_rate_indices
                    elif fund_bond_type == 'credit':
                        index_codes = self.index_processor.short_credit_indices
                    else:
                        continue

                    # 获取基金净值数据
                    start_date = (pd.to_datetime(target_date) - pd.Timedelta(days=90)).strftime('%Y-%m-%d')
                    fund_nav_df = self.wind_fetcher.get_fund_nav_smoothed(
                        fund_code, start_date, target_date
                    )

                    if fund_nav_df is None:
                        continue

                    # 获取Wind披露久期（Lasso单因子退化时用于锚定额外因子）
                    reported_duration = self.wind_fetcher.get_fund_reported_duration(
                        fund_code, target_date
                    )

                    # 计算久期
                    duration = self.duration_model.calculate_fund_duration(
                        fund_nav_df, index_codes, target_date,
                        reported_duration=reported_duration,
                        fund_code=fund_code
                    )

                    if duration is not None:
                        results[fund_code] = {
                            'duration': duration,
                            'fund_type': fund_type,
                            'bond_type': fund_bond_type
                        }

            elif fund_type == 'medium_long':
                # 中长期基金
                for idx, row in fund_df.iterrows():
                    fund_code = row['Code']

                    # 判断基金类型
                    fund_bond_type = self.fund_classifier.get_fund_type(fund_code, target_date)

                    if fund_bond_type == 'rate':
                        index_codes = self.index_processor.medium_long_rate_indices
                    elif fund_bond_type == 'credit':
                        index_codes = self.index_processor.medium_long_credit_indices
                    else:
                        continue

                    # 获取基金净值数据
                    start_date = (pd.to_datetime(target_date) - pd.Timedelta(days=90)).strftime('%Y-%m-%d')
                    fund_nav_df = self.wind_fetcher.get_fund_nav_smoothed(
                        fund_code, start_date, target_date
                    )

                    if fund_nav_df is None:
                        continue

                    # 获取Wind披露久期（Lasso单因子退化时用于锚定额外因子）
                    reported_duration = self.wind_fetcher.get_fund_reported_duration(
                        fund_code, target_date
                    )

                    # 计算久期
                    duration = self.duration_model.calculate_fund_duration(
                        fund_nav_df, index_codes, target_date,
                        reported_duration=reported_duration,
                        fund_code=fund_code
                    )

                    if duration is not None:
                        results[fund_code] = {
                            'duration': duration,
                            'fund_type': fund_type,
                            'bond_type': fund_bond_type
                        }

        return results

    def calculate_duration_statistics(self, target_date):
        """
        计算久期统计数据（中位数和分歧度）

        参数:
        target_date: 目标日期 'YYYY-MM-DD'

        返回:
        dict: 久期统计数据
        """
        results = self.calculate_fund_pool_duration(target_date)

        if not results:
            return None

        # 转换为DataFrame
        df = pd.DataFrame.from_dict(results, orient='index')

        # 分类统计
        stats = {}

        for fund_type in ['short', 'medium_long']:
            for bond_type in ['rate', 'credit']:
                key = f'{fund_type}_{bond_type}'

                mask = (df['fund_type'] == fund_type) & (df['bond_type'] == bond_type)
                subset = df[mask]

                if len(subset) > 0:
                    durations = subset['duration'].values

                    stats[key] = {
                        'count': len(durations),
                        'median': np.median(durations),
                        'mean': np.mean(durations),
                        'std': np.std(durations),
                        'cv': np.std(durations) / np.mean(durations) if np.mean(durations) > 0 else np.nan,
                        'min': np.min(durations),
                        'max': np.max(durations)
                    }

        return stats


if __name__ == '__main__':
    # 这里需要其他模块的支持，不单独测试
    pass
