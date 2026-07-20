"""
清除所有缓存并重新生成
解决 pickle 兼容性问题
"""

import os
import shutil
from pathlib import Path

def clear_cache():
    """清除所有缓存文件"""
    cache_dir = Path('data')

    print("="*60)
    print("清除缓存文件")
    print("="*60)

    if cache_dir.exists():
        # 删除整个 data 目录
        shutil.rmtree(cache_dir)
        print(f"已删除: {cache_dir}")
    else:
        print(f"缓存目录不存在: {cache_dir}")

    # 重新创建目录结构
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / 'nav').mkdir(parents=True, exist_ok=True)
    (cache_dir / 'duration').mkdir(parents=True, exist_ok=True)

    print(f"已重新创建目录结构")
    print("\n" + "="*60)
    print("缓存清除完成！")
    print("="*60)

if __name__ == '__main__':
    clear_cache()
