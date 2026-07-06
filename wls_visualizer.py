"""
WLS回归可视化模块
功能：绘制基金久期测算中WLS回归的综合分析图表
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from scipy import stats

try:
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

# 中文字体配置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class WLSVisualizer:
    """WLS回归可视化类"""

    def __init__(self, duration_model, index_processor, log_file=None):
        """
        初始化

        参数:
        duration_model: DurationModel实例
        index_processor: BondIndexDataProcessor实例
        log_file: 迭代日志文件路径（可选，如果提供则从Excel加载日志）
        """
        self.duration_model = duration_model
        self.index_processor = index_processor
        self.log_file = log_file
        self._loaded_logs = None

        # 如果提供了日志文件，尝试加载
        if log_file and os.path.exists(log_file):
            self._load_logs_from_excel(log_file)

    def _load_logs_from_excel(self, log_file):
        """从Excel文件加载迭代日志"""
        try:
            import pandas as pd
            df_all = pd.read_excel(log_file, sheet_name='全量汇总')
            df_detail = pd.read_excel(log_file, sheet_name='迭代详情')

            # 构建日志字典
            self._loaded_logs = {}
            for _, row in df_all.iterrows():
                fund_code = row['fund_code']
                self._loaded_logs[fund_code] = {
                    'fund_code': fund_code,
                    'triggered': row.get('triggered', False),
                    'total_iterations': row.get('total_iterations', 0),
                    'convergence': row.get('convergence', None),
                    'initial_factors': self._parse_factor_list(row.get('initial_factors')),
                    'final_factors': self._parse_factor_list(row.get('final_factors')),
                    'final_coefficients': self._parse_coefficients(
                        self._parse_factor_list(row.get('final_factors')),
                        df_detail[df_detail['fund_code'] == fund_code]
                    )
                }

            # 从迭代详情中获取最终的系数
            for _, row in df_detail[df_detail['round'] == df_detail['round'].max()].iterrows():
                fund_code = row['fund_code']
                if fund_code in self._loaded_logs:
                    # 解析系数字符串
                    coef_str = row.get('coefficients_str', '')
                    if coef_str:
                        coefs = self._parse_coefficient_string(coef_str)
                        if coefs:
                            self._loaded_logs[fund_code]['final_coefficients'] = coefs

            print(f'从Excel加载日志: {len(self._loaded_logs)} 只基金')
        except Exception as e:
            print(f'加载Excel日志失败: {e}')
            self._loaded_logs = None

    def _parse_factor_list(self, factor_str):
        """解析因子列表字符串"""
        if pd.isna(factor_str) or not factor_str:
            return []
        return [f.strip() for f in str(factor_str).split(',') if f.strip()]

    def _parse_coefficient_string(self, coef_str):
        """解析系数字符串，格式: 'code1=0.123, code2=0.456'"""
        coefs = {}
        if not coef_str:
            return coefs
        for item in str(coef_str).split(','):
            if '=' in item:
                code, val = item.split('=')
                try:
                    coefs[code.strip()] = float(val.strip())
                except:
                    pass
        return coefs

    def _parse_coefficients(self, factors, df_detail):
        """从迭代详情中解析系数"""
        coefs = {}
        if df_detail.empty:
            return coefs
        # 获取最后一轮的系数
        last_round = df_detail[df_detail['round'] == df_detail['round'].max()]
        if not last_round.empty:
            coef_str = last_round.iloc[0].get('coefficients_str', '')
            return self._parse_coefficient_string(coef_str)
        return coefs

    def get_log(self, fund_code):
        """获取基金的迭代日志"""
        # 优先使用加载的日志
        if self._loaded_logs and fund_code in self._loaded_logs:
            return self._loaded_logs[fund_code]
        # 回退到运行时日志
        return self.duration_model.iteration_logs.get(fund_code)

    def get_regression_data(self, fund_code, fund_nav_df, index_codes,
                           target_date, reported_duration=None,
                           weight_method='linear'):
        """
        获取回归所需的全部数据

        参数:
            fund_code: 基金代码
            fund_nav_df: 基金净值DataFrame（包含return列）
            index_codes: 指数代码列表
            target_date: 目标日期
            reported_duration: Wind披露久期
            weight_method: 权重方法 ('linear', 'exponential', 'uniform')

        返回:
            dict: 包含回归数据的字典
        """
        # 1. 从迭代日志获取最终因子和系数
        log = self.get_log(fund_code)
        if log and log.get('final_coefficients'):
            final_factors = log['final_factors']
            final_coefs = log['final_coefficients']
        else:
            # 如果没有日志，返回None
            return None

        # 2. 获取基金收益率
        end_date = pd.to_datetime(target_date)
        start_date = end_date - pd.Timedelta(days=90)
        fund_returns = fund_nav_df['return'].loc[start_date:end_date].dropna()

        if len(fund_returns) < 5:
            return None

        # 3. 获取指数收益率
        index_prices = self.index_processor.get_index_prices(
            index_codes,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )

        if index_prices.empty:
            return None

        index_returns = index_prices.pct_change().dropna()

        # 4. 获取最终因子的收益率
        final_factor_returns = index_returns[final_factors].loc[fund_returns.index.intersection(index_returns.index)]

        # 5. 对齐数据
        common_dates = fund_returns.index.intersection(final_factor_returns.index)
        if len(common_dates) < 5:
            return None

        fund_returns = fund_returns.loc[common_dates]
        final_factor_returns = final_factor_returns.loc[common_dates]

        # 6. 计算拟合值和残差
        X = final_factor_returns.values
        coefficients = np.array([final_coefs.get(f, 0) for f in final_factors])
        fitted_values = pd.Series(X @ coefficients, index=common_dates)
        residuals = fund_returns.values - fitted_values.values
        residuals = pd.Series(residuals, index=common_dates)

        # 7. 计算权重
        weights = self.duration_model._get_time_weights(len(fund_returns), method=weight_method)

        # 8. 计算R²
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((fund_returns.values - fund_returns.values.mean())**2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # 9. 计算最终久期
        total_duration = 0
        for f in final_factors:
            idx_dur = self.index_processor.get_latest_duration(f, target_date)
            if idx_dur is not None and not np.isnan(idx_dur):
                total_duration += final_coefs.get(f, 0) * idx_dur

        return {
            'fund_code': fund_code,
            'fund_returns': fund_returns,
            'index_returns': final_factor_returns,
            'weights': weights,
            'coefficients': final_coefs,
            'final_factors': final_factors,
            'fitted_values': fitted_values,
            'residuals': residuals,
            'r_squared': r_squared,
            'calculated_duration': total_duration,
            'reported_duration': reported_duration
        }

    def plot_main_panel(self, fund_code, fund_name, data,
                       reported_duration=None, output_dir=None,
                        figsize=(14, 10), save=True, show=True):
        """
        绘制第一页：回归结果概览（2x2布局）

        参数:
            fund_code: 基金代码
            fund_name: 基金名称
            data: get_regression_data返回的数据字典
            reported_duration: Wind披露久期
            output_dir: 输出目录
            figsize: 图形大小
            save: 是否保存
            show: 是否显示
        """
        if data is None:
            print(f'  [警告] {fund_code} 数据不足，无法绘图')
            return None

        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.2)

        fund_returns = data['fund_returns']
        fitted_values = data['fitted_values']
        residuals = data['residuals']
        weights = data['weights']
        coefficients = data['coefficients']
        final_factors = data['final_factors']
        r_squared = data['r_squared']
        calc_dur = data['calculated_duration']
        report_dur = data.get('reported_duration', reported_duration)

        # 标题信息
        title = f'{fund_code} - {fund_name[:20]}'
        if len(fund_name) > 20:
            title = f'{fund_code} - {fund_name[:20]}...'

        # 子图1：时间序列图
        ax1 = fig.add_subplot(gs[0, 0])
        dates = fund_returns.index
        index_returns = data['index_returns']

        # 定义颜色列表（用于各指数）
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

        # 绘制各指数收益率（细线）
        for i, factor in enumerate(final_factors):
            if factor in index_returns.columns:
                color = colors[i % len(colors)]
                ax1.plot(dates, index_returns[factor].values, '-',
                        label=f'{self._format_factor_name(factor)}',
                        markersize=3, linewidth=0.8, alpha=0.5, color=color)

        # 绘制实际收益率和拟合收益率（粗线，突出显示）
        ax1.plot(dates, fund_returns.values, 'o-', label='实际收益率',
                markersize=4, linewidth=1.8, alpha=0.8, color='#2c3e50', zorder=10)
        ax1.plot(dates, fitted_values.values, 's--', label='拟合收益率',
                markersize=4, linewidth=1.8, alpha=0.8, color='#e74c3c', zorder=10)

        # 用点的大小表示权重
        normalized_weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-10)
        for i, (date, w) in enumerate(zip(dates, normalized_weights)):
            ax1.scatter(date, fund_returns.values[i], s=20 + w*80, alpha=0.3, color='#2c3e50', zorder=5)
            ax1.scatter(date, fitted_values.values[i], s=20 + w*80, alpha=0.3, color='#e74c3c', zorder=5)

        ax1.set_xlabel('日期', fontsize=10)
        ax1.set_ylabel('收益率', fontsize=10)
        ax1.set_title('收益率时间序列（基金 vs 指数 vs 拟合）', fontsize=11, fontweight='bold')
        ax1.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1, 1))
        ax1.grid(alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)

        # 子图2：残差分布直方图
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(residuals.values, bins=15, color='#3498db', alpha=0.7, edgecolor='white', density=True)

        # 添加正态分布拟合
        mu, sigma = residuals.mean(), residuals.std()
        x = np.linspace(residuals.min(), residuals.max(), 100)
        ax2.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='正态拟合')

        ax2.axvline(0, color='gray', linestyle='--', alpha=0.5)
        ax2.axvline(mu, color='red', linestyle='-', linewidth=1.5, label=f'均值={mu:.4f}')
        ax2.set_xlabel('残差', fontsize=10)
        ax2.set_ylabel('密度', fontsize=10)
        ax2.set_title(f'残差分布 (σ={sigma:.4f})', fontsize=11, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(alpha=0.3)

        # 子图3：因子系数条形图
        ax3 = fig.add_subplot(gs[1, 0])

        # 获取每个因子的久期
        factor_durations = []
        for f in final_factors:
            dur = self.index_processor.get_latest_duration(f, data.get('target_date', '2025-12-31'))
            factor_durations.append(dur if dur is not None else 0)

        # 按久期排序
        sorted_indices = np.argsort(factor_durations)
        sorted_factors = [final_factors[i] for i in sorted_indices]
        sorted_coefs = [coefficients.get(f, 0) for f in sorted_factors]
        sorted_durs = [factor_durations[i] for i in sorted_indices]

        y_pos = np.arange(len(sorted_factors))
        bars = ax3.barh(y_pos, sorted_coefs, color='#27ae60', alpha=0.8, edgecolor='white')

        # 标注数值和久期
        for i, (bar, coef, dur) in enumerate(zip(bars, sorted_coefs, sorted_durs)):
            ax3.text(coef + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{coef:.3f} (久期{dur:.2f}Y)',
                    va='center', fontsize=8)

        # 总权重
        sum_beta = sum(sorted_coefs)
        ax3.axvline(sum_beta, color='red', linestyle='--', alpha=0.7, label=f'Σβ={sum_beta:.3f}')

        ax3.set_yticks(y_pos)
        ax3.set_yticklabels([self._format_factor_name(f) for f in sorted_factors], fontsize=8)
        ax3.set_xlabel('系数', fontsize=10)
        ax3.set_title('因子回归系数', fontsize=11, fontweight='bold')
        ax3.legend(fontsize=9, loc='lower right')
        ax3.grid(axis='x', alpha=0.3)

        # 子图4：残差时间序列图
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(dates, residuals.values, 'o-', label='残差',
                markersize=4, linewidth=1, alpha=0.7, color='#9b59b6')

        # 点的大小表示权重
        for i, (date, w) in enumerate(zip(dates, normalized_weights)):
            ax4.scatter(date, residuals.values[i], s=20 + w*80, alpha=0.4, color='#9b59b6')

        ax4.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax4.axhline(mu, color='red', linestyle='-', linewidth=1, alpha=0.7)
        ax4.axhline(mu + 1.96*sigma, color='orange', linestyle=':', alpha=0.5, label='95%置信区间')
        ax4.axhline(mu - 1.96*sigma, color='orange', linestyle=':', alpha=0.5)

        ax4.set_xlabel('日期', fontsize=10)
        ax4.set_ylabel('残差', fontsize=10)
        ax4.set_title('残差时间序列', fontsize=11, fontweight='bold')
        ax4.legend(fontsize=9)
        ax4.grid(alpha=0.3)
        ax4.tick_params(axis='x', rotation=45)

        # 总标题
        fig.suptitle(f'{title}  WLS回归分析  R²={r_squared:.4f}',
                    fontsize=14, fontweight='bold', y=0.98)

        # 添加久期对比信息
        dur_text = f'Wind披露: {report_dur:.3f}年  |  测算: {calc_dur:.3f}年  |  偏差: {calc_dur - report_dur:.3f}年' if report_dur else f'测算久期: {calc_dur:.3f}年'
        fig.text(0.5, 0.94, dur_text, ha='center', fontsize=11,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.5))

        plt.tight_layout(rect=[0, 0, 1, 0.92])

        if save and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f'WLS_main_{fund_code.replace(".", "_")}.png')
            plt.savefig(output_path, bbox_inches='tight', dpi=120)
            print(f'  [保存] 主面板图: {output_path}')

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def plot_factor_scatters(self, fund_code, fund_name, data,
                            output_dir=None, figsize=(14, 8),
                            save=True, show=True):
        """
        绘制第二页：因子散点图（动态布局）

        参数:
            fund_code: 基金代码
            fund_name: 基金名称
            data: get_regression_data返回的数据字典
            output_dir: 输出目录
            figsize: 图形大小
            save: 是否保存
            show: 是否显示
        """
        if data is None:
            return None

        fund_returns = data['fund_returns']
        index_returns = data['index_returns']
        weights = data['weights']
        final_factors = data['final_factors']

        n_factors = len(final_factors)

        # 根据因子数量决定布局
        if n_factors <= 2:
            n_rows, n_cols = 1, 2
        elif n_factors <= 4:
            n_rows, n_cols = 2, 2
        else:
            n_rows, n_cols = 3, 2

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        fig.suptitle(f'{fund_code} - {fund_name[:30]}  各因子收益率 vs 基金收益率',
                    fontsize=14, fontweight='bold')

        # 确保axes是2D数组
        if n_factors == 1:
            axes = np.array([axes])
        elif n_rows == 1 or n_cols == 1:
            axes = axes.reshape(n_rows, n_cols)

        normalized_weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-10)

        for i, factor in enumerate(final_factors):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]

            factor_returns = index_returns[factor].values

            # 散点图，点大小表示权重
            scatter = ax.scatter(factor_returns, fund_returns.values,
                               s=20 + normalized_weights * 80,
                               alpha=0.5, c=normalized_weights,
                               cmap='viridis', edgecolors='none')

            # 计算回归线
            mask = ~np.isnan(factor_returns) & ~np.isnan(fund_returns.values)
            if mask.sum() > 2:
                z = np.polyfit(factor_returns[mask], fund_returns.values[mask], 1)
                p = np.poly1d(z)
                x_line = np.linspace(factor_returns[mask].min(), factor_returns[mask].max(), 100)
                ax.plot(x_line, p(x_line), "r--", linewidth=2, alpha=0.8)

                # 计算R²
                y_pred = p(factor_returns[mask])
                ss_res = np.sum((fund_returns.values[mask] - y_pred) ** 2)
                ss_tot = np.sum((fund_returns.values[mask] - fund_returns.values[mask].mean()) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            else:
                r2 = 0

            # 获取因子久期
            factor_dur = self.index_processor.get_latest_duration(factor, data.get('target_date', '2025-12-31'))
            dur_text = f' (久期{factor_dur:.2f}Y)' if factor_dur and not np.isnan(factor_dur) else ''

            ax.set_xlabel(f'{self._format_factor_name(factor)}收益率{dur_text}', fontsize=9)
            ax.set_ylabel('基金收益率', fontsize=9)
            ax.set_title(f'R²={r2:.4f}', fontsize=10)
            ax.grid(alpha=0.3)

        # 隐藏多余的子图
        for i in range(n_factors, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].axis('off')

        # 添加颜色条说明权重
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(scatter, cax=cbar_ax)
        cbar.set_label('权重 (越大越新)', fontsize=9)

        plt.tight_layout(rect=[0, 0, 0.9, 0.95])

        if save and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f'WLS_factors_{fund_code.replace(".", "_")}.png')
            plt.savefig(output_path, bbox_inches='tight', dpi=120)
            print(f'  [保存] 因子散点图: {output_path}')

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def plot_all(self, fund_code, fund_name, data,
                reported_duration=None, output_dir=None,
                show=True, save=True):
        """
        绘制两页完整图表

        参数:
            fund_code: 基金代码
            fund_name: 基金名称
            data: get_regression_data返回的数据字典
            reported_duration: Wind披露久期
            output_dir: 输出目录
            show: 是否显示
            save: 是否保存
        """
        if data is None:
            print(f'  [警告] {fund_code} 数据不足，无法绘图')
            return

        print(f'\n[绘图] {fund_code} - {fund_name}')

        # 第一页：主面板
        self.plot_main_panel(fund_code, fund_name, data,
                             reported_duration, output_dir,
                             save=save, show=False)

        # 第二页：因子散点图
        self.plot_factor_scatters(fund_code, fund_name, data,
                                  output_dir, save=save, show=show)

    def _format_factor_name(self, factor_code):
        """格式化指数代码为可读名称"""
        # 简化显示，去掉后缀（如.CS），保留代码部分
        return str(factor_code).split('.')[0]


if __name__ == '__main__':
    print('WLS可视化模块加载成功')
