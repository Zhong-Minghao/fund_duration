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
                 max_lev=1.4,
                 outlier_threshold=2.5):
        """
        初始化

        参数:
        index_processor: BondIndexDataProcessor实例
        window: 回归窗口（交易日数）
        lasso_alpha: Lasso正则化参数
        min_lev: 最小杠杆率
        max_lev: 最大杠杆率
        outlier_threshold: OLS残差标准化阈值，超过此值的数据点整行剔除（默认2.5）
        """
        self.index_processor = index_processor
        self.window = window
        self.lasso_alpha = lasso_alpha
        self.min_lev = min_lev
        self.max_lev = max_lev
        self.outlier_threshold = outlier_threshold
        self.iteration_logs = {}  # {fund_code: log_dict}，记录每只基金的迭代过程

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

    def _remove_regression_outliers(self, fund_returns, index_returns):
        """
        基于 OLS 标准化残差，剔除 (y, x1, x2, ...) 联合关系中偏离正常状态的数据点。
        若剔除后剩余观测点不足则回退到原始数据。
        """
        n_obs = len(fund_returns)
        n_factors = index_returns.shape[1]
        min_obs = max(8, n_factors + 3)

        if n_obs <= min_obs:
            return fund_returns, index_returns

        y = fund_returns.values
        X = np.column_stack([np.ones(n_obs), index_returns.values])

        try:
            beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            residuals = y - X @ beta
            sigma = residuals.std()
            if sigma < 1e-10:
                return fund_returns, index_returns
            std_resid = residuals / sigma

            mask = np.abs(std_resid) <= self.outlier_threshold

            if mask.sum() < min_obs:
                return fund_returns, index_returns

            return fund_returns.iloc[mask], index_returns.iloc[mask]
        except Exception:
            return fund_returns, index_returns

    def _detect_boundary_status(self, coefficients, min_lev=None, max_lev=None, tol=1e-4):
        """
        检测WLS解是否在边界上

        参数:
            coefficients: 回归系数数组
            min_lev: 最小杠杆率
            max_lev: 最大杠杆率
            tol: 容差

        返回:
            str: 'upper' | 'lower' | 'interior' | None
        """
        if min_lev is None:
            min_lev = self.min_lev
        if max_lev is None:
            max_lev = self.max_lev

        total = np.sum(coefficients)
        if abs(total - max_lev) < tol:
            return 'upper'
        elif abs(total - min_lev) < tol:
            return 'lower'
        else:
            return 'interior'

    def _sort_factors_by_duration(self, factor_codes, target_date):
        """
        按久期从小到大排序因子

        参数:
            factor_codes: 因子代码列表
            target_date: 目标日期

        返回:
            list: [(code, duration), ...] sorted by duration ascending
        """
        factor_durations = []
        for code in factor_codes:
            dur = self.index_processor.get_latest_duration(code, target_date)
            if dur is not None and not np.isnan(dur):
                factor_durations.append((code, dur))

        return sorted(factor_durations, key=lambda x: x[1])

    def _calculate_p_values(self, X, y, weights, coefficients):
        """
        计算WLS回归系数的P值

        参数:
            X: 设计矩阵 (n_obs, n_factors)
            y: 响应变量
            weights: 权重
            coefficients: 回归系数

        返回:
            dict: {factor_index: p_value}
        """
        from scipy import stats

        n_obs, n_factors = X.shape

        # 构造权重矩阵
        W = np.diag(weights)

        # 计算残差
        y_pred = X @ coefficients
        residuals = y - y_pred

        # 计算残差标准误（加权）
        dof = n_obs - n_factors - 1  # 自由度
        if dof <= 0:
            return {i: 1.0 for i in range(n_factors)}  # 自由度不足，返回最大P值

        # σ² = (residuals' * W * residuals) / dof
        sigma2 = (residuals.T @ W @ residuals) / dof

        # 计算系数协方差矩阵: Var(β) = σ² * (X' * W * X)^(-1)
        XtW = X.T @ W
        try:
            XtWX_inv = np.linalg.inv(XtW @ X)
        except np.linalg.LinAlgError:
            # 矩阵奇异，返回最大P值
            return {i: 1.0 for i in range(n_factors)}

        cov_matrix = sigma2 * XtWX_inv

        # 提取对角线元素（各系数的方差）
        variances = np.diag(cov_matrix)

        # 计算t统计量和P值
        p_values = {}
        for i in range(n_factors):
            if variances[i] <= 0 or coefficients[i] == 0:
                p_values[i] = 1.0
            else:
                se = np.sqrt(variances[i])  # 标准误
                t_stat = coefficients[i] / se
                # 双边检验的P值
                p_values[i] = 2 * (1 - stats.t.cdf(abs(t_stat), df=dof))

        return p_values

    def _find_best_swap(self, fund_returns, current_factors, all_candidate_codes,
                        boundary_type, target_date, fund_code=None):
        """
        在久期方向约束下枚举所有合规 swap，返回 WLS 目标最小的方案。

        约束逻辑：
          上界(upper)：dur(F_new) > pool_avg，且 dur(F_remove) < dur(F_new)
          下界(lower)：dur(F_new) < pool_avg，且 dur(F_remove) > dur(F_new)

        这两条约束保证每次 swap 后池平均久期严格单调移动，从数学上消除振荡。

        参数:
            fund_returns: 基金收益率Series（已对齐）
            current_factors: 当前因子列表
            all_candidate_codes: 全部候选指数代码列表
            boundary_type: 'upper' 或 'lower'
            target_date: 目标日期
            fund_code: 基金代码（用于日志）

        返回:
            tuple: (new_factors, F_remove, F_new) 或 (None, None, None)
        """
        # 1. 当前池久期（跳过久期数据缺失的因子）
        pool_durations = {}
        for f in current_factors:
            d = self.index_processor.get_latest_duration(f, target_date)
            if d is not None and not np.isnan(d):
                pool_durations[f] = float(d)

        if not pool_durations:
            return None, None, None

        pool_avg_dur = np.mean(list(pool_durations.values()))

        # 2. 候选因子：不在池中，且满足久期方向约束
        candidate_durations = {}
        for code in all_candidate_codes:
            if code in current_factors:
                continue
            d = self.index_processor.get_latest_duration(code, target_date)
            if d is None or np.isnan(d):
                continue
            d = float(d)
            if boundary_type == 'upper' and d > pool_avg_dur:
                candidate_durations[code] = d
            elif boundary_type == 'lower' and d < pool_avg_dur:
                candidate_durations[code] = d

        if not candidate_durations:
            if fund_code:
                print(f"[无候选] {fund_code} {target_date}: {boundary_type}界，"
                      f"池均久期={pool_avg_dur:.2f}Y，无满足方向约束的候选因子，当前解为最优")
            return None, None, None

        # 3. 枚举所有合规 (F_remove, F_new) 对，选 WLS 目标最小者
        start_date = fund_returns.index[0].strftime('%Y-%m-%d')
        end_date   = fund_returns.index[-1].strftime('%Y-%m-%d')

        best_Q          = float('inf')
        best_new_factors = None
        best_remove     = None
        best_add        = None

        if fund_code:
            print(f"[swap搜索] {fund_code} {target_date}: {boundary_type}界，"
                  f"池均久期={pool_avg_dur:.2f}Y，"
                  f"候选因子{len(candidate_durations)}个: {list(candidate_durations.keys())}")

        for F_new, dur_new in candidate_durations.items():
            # 移除方向约束：替换一个"错误方向"的因子
            valid_removals = [
                f for f in current_factors
                if f in pool_durations and (
                    (boundary_type == 'upper' and pool_durations[f] < dur_new) or
                    (boundary_type == 'lower' and pool_durations[f] > dur_new)
                )
            ]
            if not valid_removals:
                continue

            # 获取 F_new 的价格数据
            prices_new = self.index_processor.get_index_prices([F_new], start_date, end_date)
            if prices_new.empty:
                continue

            for F_remove in valid_removals:
                trial_factors = [f for f in current_factors if f != F_remove] + [F_new]

                # 获取试验因子池收益率
                trial_prices = self.index_processor.get_index_prices(
                    trial_factors, start_date, end_date)
                if trial_prices.empty:
                    continue
                trial_returns = trial_prices.pct_change()

                # 对齐
                aligned = pd.DataFrame({'fund': fund_returns}).join(
                    trial_returns, how='inner').dropna()
                if aligned.shape[0] < len(trial_factors):
                    continue

                X = aligned.iloc[:, 1:].values
                y = aligned['fund'].values
                w = self._get_time_weights(len(y))

                _, coefs = self._solve_qp_osqp(X, y, w)
                if coefs is None:
                    continue

                Q = float(np.sum(w * (y - X @ coefs) ** 2))

                if fund_code:
                    print(f"  [评估] 移除{F_remove}(dur={pool_durations[F_remove]:.2f}Y)"
                          f" → 添加{F_new}(dur={dur_new:.2f}Y): Q={Q:.6f}")

                if Q < best_Q:
                    best_Q           = Q
                    best_new_factors = trial_factors
                    best_remove      = F_remove
                    best_add         = F_new

        if best_new_factors is None:
            if fund_code:
                print(f"[无方案] {fund_code} {target_date}: 所有候选swap均无法评估，终止")
            return None, None, None

        if fund_code:
            print(f"[最优swap] {fund_code} {target_date}: "
                  f"移除{best_remove} → 添加{best_add}, Q={best_Q:.6f}")

        return best_new_factors, best_remove, best_add

    def _iterative_constrained_wls(self, fund_returns, index_returns,
                                  target_date=None, fund_code=None, max_iterations=10,
                                  all_index_codes=None):
        """
        迭代带约束的WLS，当解在边界上时通过 swap-and-evaluate 调整因子池。

        策略（统一用 WLS 目标函数评判）：
        - 候选方向：上界加入更长久期因子，下界加入更短久期因子
        - 移除约束：移除方向与候选因子方向相反的池内因子
        - 择优原则：枚举所有合规 (F_remove, F_new) 对，取使 WLS 目标最小的 swap

        参数:
            fund_returns: 基金收益率Series
            index_returns: 指数收益率DataFrame（Lasso 选出的初始池）
            target_date: 目标日期
            fund_code: 基金代码（用于日志）
            max_iterations: 最大迭代次数
            all_index_codes: 完整候选指数代码列表（含 Lasso 未选的因子）；
                             为 None 时回退到 index_returns.columns

        返回:
            dict: {factor_code: coefficient}
        """
        current_factors = index_returns.columns.tolist()
        # 边界 swap 候选池：优先用完整指数集（含 Lasso 未选的因子）
        all_candidate_codes = list(all_index_codes) if all_index_codes is not None \
            else current_factors.copy()

        # 初始化结构化日志
        log = {
            'fund_code': fund_code,
            'target_date': str(target_date) if target_date else None,
            'triggered': False,
            'total_iterations': 0,
            'convergence': None,
            'initial_factors': all_candidate_codes.copy(),
            'final_factors': None,
            'final_coefficients': None,
            'iterations': []
        }

        if fund_code:
            print(f"[开始] {fund_code} {target_date} 初始因子池({len(current_factors)}个): {current_factors}")

        coefficients = None  # 防止 data_error break 后引用未定义变量
        coef_dict = {}

        for iteration in range(max_iterations):
            # 当前因子池的收益率
            current_index_returns = index_returns[current_factors]

            # 对齐数据
            aligned_data = pd.DataFrame({
                'fund': fund_returns
            }).join(current_index_returns, how='inner').dropna()

            if aligned_data.shape[0] < len(current_factors):
                if fund_code:
                    print(f"[错误] {fund_code} {target_date} 第{iteration+1}轮: 观测数({aligned_data.shape[0]}) < 因子数({len(current_factors)})")
                log['total_iterations'] = iteration + 1
                log['convergence'] = 'data_error'
                log['final_factors'] = current_factors.copy()
                log['final_coefficients'] = None
                if fund_code:
                    self.iteration_logs[fund_code] = log
                break

            X = aligned_data.iloc[:, 1:].values
            y = aligned_data['fund'].values
            n_obs = X.shape[0]

            # 生成时间权重
            adjusted_weights = self._get_time_weights(n_obs)

            # 求解WLS
            intercept, coefficients = self._solve_qp_osqp(
                X, y, adjusted_weights
            )

            if coefficients is None:
                # 求解失败，使用等权兜底
                if fund_code:
                    print(f"[兜底] {fund_code} {target_date} 第{iteration+1}轮: OSQP求解失败，使用等权")
                equal_weight = (self.min_lev + self.max_lev) / 2 / len(current_factors)
                final_params = np.full(len(current_factors), equal_weight)
                log['total_iterations'] = iteration + 1
                log['convergence'] = 'fallback'
                log['final_factors'] = current_factors.copy()
                log['final_coefficients'] = dict(zip(current_factors, final_params))
                if fund_code:
                    self.iteration_logs[fund_code] = log
                return dict(zip(current_factors, final_params))

            # 打印当前回归结果
            sum_beta = np.sum(coefficients)
            coef_dict = dict(zip(current_factors, coefficients))

            if fund_code:
                coef_str = ", ".join([f"{k}={v:.3f}" for k, v in coef_dict.items()])
                print(f"[回归] {fund_code} {target_date} 第{iteration+1}轮: Σβ={sum_beta:.4f}, [{coef_str}]")

            # 检测边界状态
            boundary_status = self._detect_boundary_status(
                coefficients, self.min_lev, self.max_lev
            )

            # 构建本轮迭代日志
            iter_log = {
                'round': iteration + 1,
                'factors': current_factors.copy(),
                'boundary_type': boundary_status,
                'sum_beta': float(sum_beta),
                'coefficients': coef_dict.copy(),
                'factor_removed': None,
                'factor_added': None,
                'swap_objective': None,
            }

            if boundary_status == 'interior':
                # 解在内部，直接返回
                if fund_code:
                    print(f"[完成] {fund_code} {target_date} 解在边界内部，迭代结束")
                log['iterations'].append(iter_log)
                log['total_iterations'] = iteration + 1
                log['convergence'] = 'interior'
                log['final_factors'] = current_factors.copy()
                log['final_coefficients'] = coef_dict
                if fund_code:
                    self.iteration_logs[fund_code] = log
                return coef_dict

            # 在边界上，搜索最优 swap
            log['triggered'] = True
            if fund_code:
                print(f"[边界] {fund_code} {target_date} 第{iteration+1}轮: "
                      f"检测到{boundary_status}边界，Σβ={sum_beta:.4f}，搜索最优swap")

            new_factors, F_remove, F_add = self._find_best_swap(
                fund_returns.loc[aligned_data.index],
                current_factors,
                all_candidate_codes,
                boundary_status,
                target_date,
                fund_code
            )

            # 补充本轮日志
            iter_log['factor_removed'] = F_remove
            iter_log['factor_added']   = F_add

            if fund_code and F_remove is not None:
                print(f"[调整] {fund_code} {target_date} 第{iteration+1}轮: "
                      f"移除{{{F_remove}}}, 添加{{{F_add}}}, "
                      f"因子数{len(current_factors)}→{len(new_factors)}")

            log['iterations'].append(iter_log)

            if new_factors is None:
                # 久期方向上无合规 swap，当前解即为最优
                log['total_iterations'] = iteration + 1
                log['convergence'] = 'no_valid_swap'
                log['final_factors'] = current_factors.copy()
                log['final_coefficients'] = coef_dict
                if fund_code:
                    self.iteration_logs[fund_code] = log
                return coef_dict

            # 更新因子池，继续迭代
            current_factors = new_factors

        # 达到最大迭代次数（data_error break 时 log['convergence'] 已设置）
        if log['convergence'] != 'data_error':
            if fund_code:
                print(f"[警告] {fund_code} {target_date} 达到最大迭代次数({max_iterations})")
            log['total_iterations'] = max_iterations
            log['convergence'] = 'max_iter'
            log['final_factors'] = current_factors.copy()
            log['final_coefficients'] = dict(zip(current_factors, coefficients)) if coefficients is not None else None
            if fund_code:
                self.iteration_logs[fund_code] = log

        if coefficients is not None:
            return dict(zip(current_factors, coefficients))
        else:
            return {}

    def _constrained_wls(self, fund_returns, index_returns, time_weights,
                        target_date=None, fund_code=None, all_index_codes=None):
        """
        带约束的加权最小二乘法（使用OSQP求解器）
        当解在边界上时通过 swap-and-evaluate 动态调整因子池

        参数:
            fund_returns: 基金收益率Series
            index_returns: 指数收益率DataFrame（Lasso 选出的初始池）
            time_weights: 时间权重（已废弃，保留参数以兼容）
            target_date: 目标日期
            fund_code: 基金代码
            all_index_codes: 完整候选指数代码列表（含 Lasso 未选的因子）

        返回:
            dict: 回归系数
        """
        return self._iterative_constrained_wls(
            fund_returns, index_returns, target_date, fund_code,
            all_index_codes=all_index_codes
        )

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

        # 剔除回归离群点：(y, x1, x2, ...) 联合关系偏离的数据点
        fund_returns, index_returns_selected = self._remove_regression_outliers(
            fund_returns, index_returns_selected
        )

        # 生成时间权重
        time_weights = self._get_time_weights(len(fund_returns))

        # 带约束的WLS
        coefficients = self._constrained_wls(
            fund_returns, index_returns_selected, time_weights,
            target_date=target_date, fund_code=fund_code,
            all_index_codes=index_returns.columns.tolist()
        )

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
        # duration = total_duration / total_weight

        # return duration

        return total_duration  # 直接返回加权久期，不除以总权重，因为总权重可能不为1，且我们希望反映实际杠杆水平


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

    def export_iteration_logs(self, target_date, results_dict=None, output_path=None):
        """
        将迭代日志导出为 Excel 文件（3个Sheet）

        参数:
        target_date: 目标日期 'YYYY-MM-DD'，用于生成默认文件名
        results_dict: {fund_code: {duration, fund_type, bond_type, ...}}，用于补充分类信息
        output_path: 输出路径，默认为 '久期迭代日志_YYYYMMDD.xlsx'

        返回:
        str: 输出文件路径
        """
        if output_path is None:
            date_str = str(target_date).replace('-', '')
            output_path = f'久期迭代日志_{date_str}.xlsx'

        logs = self.duration_model.iteration_logs
        if not logs:
            print("没有迭代日志可导出（iteration_logs 为空）")
            return None

        # ── Sheet 1：全量汇总（所有有日志的基金） ──────────────────────────
        summary_rows = []
        for fund_code, log in logs.items():
            info = results_dict.get(fund_code, {}) if results_dict else {}
            row = {
                'fund_code': fund_code,
                'fund_type': info.get('fund_type', None),
                'bond_type': info.get('bond_type', None),
                'final_duration': info.get('duration', None),
                'triggered': log.get('triggered', False),
                'total_iterations': log.get('total_iterations', 0),
                'convergence': log.get('convergence', None),
                'initial_factor_count': len(log.get('initial_factors', [])),
                'final_factor_count': len(log.get('final_factors', []) or []),
                'initial_factors': ', '.join(log.get('initial_factors', [])),
                'final_factors': ', '.join(log.get('final_factors', []) or []),
            }
            summary_rows.append(row)
        summary_df = pd.DataFrame(summary_rows)

        # ── Sheet 2：触发汇总（triggered=True，含每轮 sum_beta） ──────────
        triggered_rows = []
        for fund_code, log in logs.items():
            if not log.get('triggered', False):
                continue
            info = results_dict.get(fund_code, {}) if results_dict else {}
            row = {
                'fund_code': fund_code,
                'fund_type': info.get('fund_type', None),
                'bond_type': info.get('bond_type', None),
                'final_duration': info.get('duration', None),
                'total_iterations': log.get('total_iterations', 0),
                'convergence': log.get('convergence', None),
                'initial_factors': ', '.join(log.get('initial_factors', [])),
                'final_factors': ', '.join(log.get('final_factors', []) or []),
            }
            # 每轮 sum_beta
            for it in log.get('iterations', []):
                row[f"round_{it['round']}_sum_beta"] = it.get('sum_beta', None)
                row[f"round_{it['round']}_boundary"] = it.get('boundary_type', None)
            triggered_rows.append(row)
        triggered_df = pd.DataFrame(triggered_rows)

        # ── Sheet 3：迭代详情（triggered=True，1行/轮次） ────────────────
        detail_rows = []
        for fund_code, log in logs.items():
            if not log.get('triggered', False):
                continue
            info = results_dict.get(fund_code, {}) if results_dict else {}
            for it in log.get('iterations', []):
                # 格式化系数为可读字符串
                coef_str = ', '.join([f"{k}={v:.4f}" for k, v in (it.get('coefficients') or {}).items()])
                row = {
                    'fund_code': fund_code,
                    'fund_type': info.get('fund_type', None),
                    'bond_type': info.get('bond_type', None),
                    'round': it.get('round', None),
                    'boundary_type': it.get('boundary_type', None),
                    'sum_beta': it.get('sum_beta', None),
                    'factors': ', '.join(it.get('factors', [])),
                    'factor_removed': it.get('factor_removed', None),
                    'factor_added': it.get('factor_added', None),
                    'coefficients_str': coef_str,
                    'swap_objective': it.get('swap_objective', None),
                }
                detail_rows.append(row)
        detail_df = pd.DataFrame(detail_rows)

        # ── 写入 Excel ──────────────────────────────────────────────────
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            summary_df.to_excel(writer, sheet_name='全量汇总', index=False)
            triggered_df.to_excel(writer, sheet_name='触发汇总', index=False)
            detail_df.to_excel(writer, sheet_name='迭代详情', index=False)

        triggered_count = summary_df['triggered'].sum() if not summary_df.empty else 0
        print(f"迭代日志已导出至: {output_path}")
        print(f"  全量汇总: {len(summary_df)} 只基金")
        print(f"  触发边界: {triggered_count} 只基金")
        print(f"  迭代详情: {len(detail_df)} 条记录")

        return output_path


if __name__ == '__main__':
    # 这里需要其他模块的支持，不单独测试
    pass
