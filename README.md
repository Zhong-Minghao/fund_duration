# 纯债基金久期测算系统

基于申万宏源《纯债基金久期全解析——债市机构行为研究系列之一》研报的实现。

## 系统概述

本系统用于高频测算纯债基金的久期，通过全市场基金久期的中枢和分歧度来跟踪市场情绪变化。

## 文件结构

```
├── data_preprocessing.py        # 数据预处理模块
├── fund_type_classifier.py      # 基金类型判断模块
├── wind_data_fetcher.py         # Wind数据获取模块
├── bond_index_data.py           # 中债指数数据处理模块
├── duration_model.py            # 久期测算模型
├── 纯债基金久期测算.ipynb       # 主程序notebook
├── 纯债基金持仓情况.xlsx         # 基金持仓数据
├── 短期纯债基金样本数据.xlsx     # 短期纯债基金样本
├── 中长期纯债基金样本数据.xlsx   # 中长期纯债基金样本
└── 中债财富指数.xlsx             # 中债财富指数价格和久期数据
```

## 模块说明

### 1. data_preprocessing.py - 数据预处理模块

功能：
- 读取短期和中长期纯债基金样本数据
- 根据以下条件筛选基金：
  - 基金已经成立且未到期
  - 基金估值方法不是摊余成本法
  - 不是定期开放基金
  - 是初始基金

### 2. fund_type_classifier.py - 基金类型判断模块

功能：
- 根据持仓情况表判断基金是利率债基金还是信用债基金
- 判断标准：政金债占比 + 国债占比 > 80% 为利率债，否则为信用债
- 支持按季度动态判断（基于不泄露未来数据原则）

### 3. wind_data_fetcher.py - Wind数据获取模块

功能：
- 通过Wind Python API获取基金后复权净值数据
- 支持平滑处理和异常值过滤
- 支持批量获取多只基金数据

使用方法：
```python
from WindPy import w
w.start()  # 需要先启动Wind终端

from wind_data_fetcher import WindDataFetcher
fetcher = WindDataFetcher()
nav_df = fetcher.get_fund_nav_smoothed('000033.OF', '2024-01-01', '2024-12-20')
```

### 4. bond_index_data.py - 中债指数数据处理模块

功能：
- 读取中债财富指数价格数据（Sheet1）
- 读取中债财富指数久期数据（Sheet2）
- 根据基金类型匹配相应的指数因子

指数分类：
- **中长期利率型**：国债及政金债指数（1-3年、3-5年、5-7年、7-10年、10年以上）+ 同业存单
- **中长期信用型**：信用债总指数（1年以下、1-3年、3-5年、5-7年、7-10年、10年以上）
- **短期利率型**：同业存单指数（0-3月、3-6月、6-9月、9-12月）+ 国债及政金债（1-3年）
- **短期信用型**：短融总指数（0-3月、3-6月、6-9月、9-12月）+ 中期票据（1-3年）

### 5. duration_model.py - 久期测算模型

功能：
- 使用Lasso回归筛选因子，处理多重共线性
- 使用带约束的加权最小二乘法（WLS）估算基金久期
- 约束条件：总仓位在0.8-1.4之间，每个因子权重非负

模型创新点：
1. **Lasso回归**：处理因子共线性问题
2. **时间权重矩阵**：赋予较近数据更高权重
3. **动态因子选择**：根据基金持仓风格调整指数因子
4. **参数约束**：模拟杠杆限制

## 使用方法

### 1. 确保环境配置

需要安装以下Python库：
- pandas
- numpy
- scikit-learn
- scipy
- WindPy (需要Wind终端支持)
- openpyxl (读取Excel文件)

建议使用backtrader conda虚拟环境。

### 2. 启动Wind终端

确保Wind终端正在运行，才能调用Wind API。

### 3. 运行主程序

打开 `纯债基金久期测算.ipynb`，按顺序执行各个cell：

```python
# 1. 导入模块
from data_preprocessing import FundDataPreprocessor
from fund_type_classifier import FundTypeClassifier
from wind_data_fetcher import WindDataFetcher
from bond_index_data import BondIndexDataProcessor
from duration_model import FundDurationCalculator

# 2. 初始化各模块
preprocessor = FundDataPreprocessor()
classifier = FundTypeClassifier()
wind_fetcher = WindDataFetcher()
index_processor = BondIndexDataProcessor()

# 3. 加载中债指数数据
index_processor.load_price_data()
index_processor.load_duration_data()
classifier.load_holdings_data()

# 4. 创建久期计算器
calculator = FundDurationCalculator(
    data_preprocessor=preprocessor,
    fund_classifier=classifier,
    wind_fetcher=wind_fetcher,
    index_processor=index_processor
)

# 5. 计算久期统计数据
target_date = '2024-12-20'
stats = calculator.calculate_duration_statistics(target_date)
```

## 输出结果

系统会输出以下统计数据：
- 各类型基金数量
- 久期中位数
- 久期均值
- 久期标准差
- 变异系数（分歧度）
- 久期最小值和最大值

## 数据说明

### 持仓情况表.xlsx

列结构：
- 第0-1列：Code, Name
- 第2-33列：政金债占比（32个季度）
- 第34-65列：国债占比（32个季度）

### 基金样本数据.xlsx

列名：
- Code：基金代码
- Name：基金名称
- fund_setupdate：基金成立日
- fund_maturitydate_2：基金到期日
- fund_type：基金类型
- fund_valuationmethod：估值方法
- fund_regulopenfundornot：是否定期开放基金
- fund_initial：是否初始基金

### 中债财富指数.xlsx

Sheet1：指数价格数据
- 从第5行开始是数据
- 第4行是指数代码

Sheet2：指数久期数据
- 从第5行开始是数据
- 第2行是指数代码

## 注意事项

1. **Wind连接**：确保Wind终端正在运行才能获取基金净值数据
2. **日期格式**：统一使用 'YYYY-MM-DD' 格式
3. **季度匹配**：基于不泄露未来数据原则，季度报告披露后决定接下来的3个月基金类型
4. **计算时间**：批量计算大量基金久期需要较长时间，建议分批处理

## 下一步优化

1. 添加并行计算功能，加快批量处理速度
2. 添加结果可视化功能
3. 支持时间序列久期计算
4. 添加与公告久期的对比验证功能

## 参考资料

申万宏源研究，《纯债基金久期全解析——债市机构行为研究系列之一》，2024年12月26日
