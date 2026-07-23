"""
Wind数据本地缓存模块
功能：将Wind API返回的基金净值和季报久期数据缓存为本地Pickle文件，
     避免重复调用Wind API消耗数据限额。

目录结构:
data/
  nav/
    000033.OF.pkl    # 每只基金一个文件，索引=date，列=NAV
    ...
  duration/
    duration_cache.pkl  # 联合索引=(fund_code, rpt_date)，列=duration
"""

import pandas as pd
import numpy as np
from pathlib import Path


class WindDataCache:
    """Wind数据本地Pickle缓存"""

    def __init__(self, cache_dir='data'):
        self.cache_dir = Path(cache_dir)
        self.nav_dir = self.cache_dir / 'nav'
        self.duration_file = self.cache_dir / 'duration' / 'duration_cache.pkl'
        self._ensure_dirs()

    def _ensure_dirs(self):
        self.nav_dir.mkdir(parents=True, exist_ok=True)
        self.duration_file.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  NAV 缓存
    # ------------------------------------------------------------------ #

    def _nav_path(self, fund_code: str) -> Path:
        safe_code = fund_code.replace('/', '_')
        return self.nav_dir / f'{safe_code}.pkl'

    def check_nav_coverage(self, fund_code: str, start_date: str, end_date: str) -> bool:
        """
        判断缓存是否覆盖 [start_date, end_date]。
        不要求每天都有数据（交易日缺口正常），只检查起止范围。
        """
        path = self._nav_path(fund_code)
        if not path.exists():
            return False
        try:
            import pickle
            with open(path, 'rb') as f:
                df = pickle.load(f)
            if df.empty:
                return False
            # 修复 datetime64[s] 索引格式兼容性问题
            if df.index.dtype == 'datetime64[s]':
                df.index = df.index.astype('datetime64[ns]')
            cache_min = df.index.min()
            cache_max = df.index.max()
            return cache_min <= pd.to_datetime(start_date) and cache_max >= pd.to_datetime(end_date)
        except Exception:
            return False

    def read_nav(self, fund_code: str, start_date: str, end_date: str) -> pd.DataFrame | None:
        """读取指定日期范围内的缓存NAV数据（仅含 NAV 列）"""
        path = self._nav_path(fund_code)
        if not path.exists():
            return None
        try:
            import pickle
            with open(path, 'rb') as f:
                df = pickle.load(f)
            # 修复 datetime64[s] 索引格式兼容性问题
            if hasattr(df.index, 'dtype') and str(df.index.dtype) == 'datetime64[s]':
                # 转换为兼容格式
                df.index = pd.to_datetime(df.index.astype(str))
            elif hasattr(df.index, 'dtype') and df.index.dtype.name.startswith('datetime64'):
                # 确保使用纳秒精度
                df.index = df.index.astype('datetime64[ns]')
            mask = (df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))
            result = df.loc[mask]
            return result if not result.empty else None
        except Exception as e:
            # 打印错误信息以便调试，但返回None让调用方从Wind获取
            print(f"[缓存警告] {fund_code}: 读取缓存失败 ({str(e)[:50]}...), 将从Wind重新获取")
            return None

    def write_nav(self, fund_code: str, df: pd.DataFrame):
        """
        写入NAV数据到缓存。
        若缓存已存在则合并（去重、按日期排序），保留历史数据。
        df 需含 DatetimeIndex 和 NAV 列。
        """
        if df is None or df.empty:
            return
        path = self._nav_path(fund_code)
        nav_df = df[['NAV']].copy()
        nav_df.index = pd.to_datetime(nav_df.index)

        if path.exists():
            try:
                existing = pd.read_pickle(path)
                nav_df = pd.concat([existing, nav_df])
                nav_df = nav_df[~nav_df.index.duplicated(keep='last')]
                nav_df = nav_df.sort_index()
            except Exception:
                pass

        nav_df.to_pickle(path)

    # ------------------------------------------------------------------ #
    #  Duration 缓存
    # ------------------------------------------------------------------ #

    def _load_duration_cache(self) -> pd.DataFrame:
        """加载 duration_cache.pkl，返回以 (fund_code, rpt_date) 为索引的 DataFrame"""
        if not self.duration_file.exists():
            return pd.DataFrame(columns=['duration'],
                                index=pd.MultiIndex.from_tuples([], names=['fund_code', 'rpt_date']))
        try:
            return pd.read_pickle(self.duration_file)
        except Exception:
            return pd.DataFrame(columns=['duration'],
                                index=pd.MultiIndex.from_tuples([], names=['fund_code', 'rpt_date']))

    def check_duration_exists(self, fund_code: str, rpt_date: str) -> bool:
        """检查指定基金在指定季报日期的久期是否已缓存"""
        cache = self._load_duration_cache()
        try:
            return (fund_code, rpt_date) in cache.index
        except Exception:
            return False

    def read_duration(self, fund_code: str, rpt_date: str) -> float | None:
        """读取单只基金的缓存季报久期，不存在则返回 None"""
        cache = self._load_duration_cache()
        try:
            val = cache.loc[(fund_code, rpt_date), 'duration']
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return None
            return float(val)
        except (KeyError, TypeError):
            return None

    def write_duration_batch(self, rpt_date: str, code_duration_dict: dict):
        """
        批量写入季报久期。
        code_duration_dict: {fund_code: duration_float_or_None}
        rpt_date: 'YYYYMMDD'
        """
        if not code_duration_dict:
            return

        rows = [
            {'fund_code': code, 'rpt_date': rpt_date, 'duration': val}
            for code, val in code_duration_dict.items()
        ]
        new_df = pd.DataFrame(rows).set_index(['fund_code', 'rpt_date'])

        existing = self._load_duration_cache()
        merged = pd.concat([existing, new_df])
        merged = merged[~merged.index.duplicated(keep='last')]
        merged = merged.sort_index()
        merged.to_pickle(self.duration_file)
