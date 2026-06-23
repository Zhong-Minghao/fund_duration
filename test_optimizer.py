"""
测试优化器为什么返回初始值
"""
import numpy as np
from scipy.optimize import minimize

print('='*60)
print('测试优化器问题')
print('='*60)

# 模拟数据（根据实际情况调整）
np.random.seed(42)
n_obs = 30
n_factors = 2

# 创建有相关性的因子
X = np.random.randn(n_obs, n_factors)
X[:, 1] = 0.9 * X[:, 0] + 0.1 * np.random.randn(n_obs)  # 高度相关

# 创建y
true_params = np.array([0.5, 0.5])
y = X @ true_params + 0.01 * np.random.randn(n_obs)

# 时间权重
time_weights = np.linspace(0.5, 1.0, n_obs)
time_weights = time_weights / time_weights.sum() * n_obs

print(f'数据形状: X={X.shape}, y={y.shape}')
print(f'因子相关性: {np.corrcoef(X.T)[0, 1]:.4f}')

# 定义目标函数
def objective(params):
    residuals = y - X @ params
    weighted_residuals = residuals * np.sqrt(time_weights)
    return np.sum(weighted_residuals ** 2)

# 定义梯度（帮助调试）
def gradient(params):
    residuals = y - X @ params
    grad = -2 * X.T @ (time_weights * residuals)
    return grad

# 约束条件
constraints = [
    {'type': 'ineq', 'fun': lambda params: np.sum(params) - 0.8},
    {'type': 'ineq', 'fun': lambda params: 1.4 - np.sum(params)},
]
bounds = [(1e-6, None)] * n_factors

# 测试不同初始值
print('\n' + '='*60)
print('测试不同初始值')
print('='*60)

test_cases = [
    np.array([0.5, 0.5]),
    np.array([0.2, 0.8]),
    np.array([0.8, 0.2]),
    np.array([0.99, 0.01]),
]

for x0 in test_cases:
    print(f'\n初始值: {x0}')
    print(f'初始目标函数值: {objective(x0):.8f}')
    print(f'初始梯度: {gradient(x0)}')
    print(f'初始梯度范数: {np.linalg.norm(gradient(x0)):.2e}')

    # 方法1: SLSQP（原始方法）
    result_slsqp = minimize(
        objective, x0, method='SLSQP',
        bounds=bounds, constraints=constraints,
        options={'ftol': 1e-9, 'maxiter': 1000, 'disp': False}
    )
    print(f'SLSQP: x={result_slsqp.x}, fun={result_slsqp.fun:.8f}, success={result_slsqp.success}')
    print(f'       message={result_slsqp.message}')
    if result_slsqp.success:
        print(f'       nit={result_slsqp.nit}, nfev={result_slsqp.nfev}')

    # 方法2: SLSQP with gradient
    result_slsqp_grad = minimize(
        objective, x0, method='SLSQP', jac=gradient,
        bounds=bounds, constraints=constraints,
        options={'ftol': 1e-9, 'maxiter': 1000}
    )
    print(f'SLSQP+grad: x={result_slsqp_grad.x}, fun={result_slsqp_grad.fun:.8f}, success={result_slsqp_grad.success}')

    # 方法3: trust-constr
    result_trust = minimize(
        objective, x0, method='trust-constr', jac=gradient,
        bounds=bounds, constraints=constraints,
        options={'ftol': 1e-9, 'maxiter': 1000}
    )
    print(f'trust-constr: x={result_trust.x}, fun={result_trust.fun:.8f}, success={result_trust.success}')

    # 检查目标函数在初始值附近的变化
    print('目标函数在初始值附近:')
    for delta in [-0.01, -0.001, 0.001, 0.01]:
        x_test = x0.copy()
        x_test[0] += delta
        x_test[1] -= delta  # 保持总和不变
        if (x_test > 0).all() and 0.8 <= x_test.sum() <= 1.4:
            print(f'  x={x_test}, fun={objective(x_test):.8f}')

# 分析：为什么优化器可能返回初始值
print('\n' + '='*60)
print('问题分析')
print('='*60)

print('\n可能的原因：')
print('1. **初始值恰好是局部最优解**')
print('   - 如果初始值的梯度接近零，优化器会认为已达到最优')
print('')
print('2. **目标函数平坦**')
print('   - 如果目标函数在参数空间中非常平坦，优化器可能无法找到更好的方向')
print('')
print('3. **约束边界限制**')
print('   - 如果最优解在约束边界上，而初始值刚好在边界附近')
print('')
print('4. **数值精度问题**')
print('   - ftol设置可能不合适')

# 推荐的解决方案
print('\n' + '='*60)
print('推荐解决方案')
print('='*60)

print('\n方案1: 使用普通WLS作为初始值（推荐）')
print('  先用无约束WLS求解，如果结果满足约束则直接使用')
print('  如果不满足约束，再用约束优化，以WLS解作为初始值')

print('\n方案2: 添加梯度计算')
print('  显式提供梯度可以提高优化器性能')

print('\n方案3: 尝试不同的优化方法')
print('  SLSQP可能不是最适合的方法，可以尝试trust-constr')

print('\n方案4: 检查数据问题')
print('  如果因子高度相关，会导致病态问题，考虑:')
print('  - 使用岭回归')
print('  - 减少因子数量')
print('  - 使用PCA降维')

# 示例：方案1的实现
print('\n' + '='*60)
print('方案1示例代码')
print('='*60)

print('''
# 先用无约束WLS
from sklearn.linear_model import LinearRegression

W = time_weights
X_sqrtW = X * np.sqrt(W[:, None])
y_sqrtW = y * np.sqrt(W)

lr = LinearRegression(fit_intercept=False)
lr.fit(X_sqrtW, y_sqrtW)
wls_solution = lr.coef_

print(f'WLS解: {wls_solution}')
print(f'WLS解总和: {wls_solution.sum():.4f}')

# 检查是否满足约束
if 0.8 <= wls_solution.sum() <= 1.4 and (wls_solution > 0).all():
    print('WLS解满足约束，直接使用')
    final_params = wls_solution
else:
    print('WLS解不满足约束，使用约束优化')
    # 以WLS解作为初始值（先投影到可行域）
    x0 = wls_solution.copy()
    # 确保正值
    x0 = np.maximum(x0, 1e-6)
    # 调整总和到约束范围内
    if x0.sum() < 0.8:
        x0 = x0 / x0.sum() * 0.9
    elif x0.sum() > 1.4:
        x0 = x0 / x0.sum() * 1.1

    result = minimize(objective, x0, method='SLSQP',
                      bounds=bounds, constraints=constraints)
    final_params = result.x

print(f'最终参数: {final_params}')
''')
