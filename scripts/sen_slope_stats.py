# -*- coding: utf-8 -*-
"""统计全省 Sen 斜率分布: 最小值 / 最大值 / 中位数 / 第10与第90百分位数"""
import sys
from pathlib import Path
sys.path.insert(0, r"C:\pylibs")

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parent.parent
MK_DIR = ROOT / "output" / "Mann-Kendall 趋势检验"
FILES = {
    "浅层根区 (0-28cm)": MK_DIR / "河南浅层SSI_4-5月_MK_Sen斜率_2016-2025.tif",
    "表层 (0-7cm)": MK_DIR / "河南SSI_4-5月_MK_Sen斜率_2016-2025.tif",
}

for name, path in FILES.items():
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float64)
        arr[arr <= src.nodata / 2] = np.nan  # nodata -> NaN
    vals = arr[np.isfinite(arr)]
    print(f"===== {name} =====")
    print(f"有效像元数: {vals.size}")
    print(f"最小值  : {vals.min():+.4f}")
    print(f"最大值  : {vals.max():+.4f}")
    print(f"中位数  : {np.median(vals):+.4f}")
    print(f"第10百分位数 (最负10%对应值): {np.percentile(vals, 10):+.4f}")
    print(f"第90百分位数 (最正10%对应值): {np.percentile(vals, 90):+.4f}")
    print()
