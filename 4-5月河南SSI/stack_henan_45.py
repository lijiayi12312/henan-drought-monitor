# -*- coding: utf-8 -*-
"""
提取河南省 SSI 逐月栅格中每年 4 月、5 月的数据, 堆叠为一个多波段 tif
输入: d:\\大二\\干旱预测\\河南省SSI\\河南SSI_YYYYMM.tif (201601-202512)
输出: d:\\大二\\干旱预测\\4-5月河南SSI\\4-5月河南SSI_2016-2025.tif
      20 个波段, 顺序 2016-04, 2016-05, 2017-04, ..., 2025-05
依赖: rasterio
"""
from pathlib import Path

import numpy as np
import rasterio

SRC_DIR = Path(r"d:\大二\干旱预测\河南省SSI")
OUT_DIR = Path(r"d:\大二\干旱预测\4-5月河南SSI")
OUT_PATH = OUT_DIR / "4-5月河南SSI_2016-2025.tif"

YEARS = range(2016, 2026)
MONTHS = (4, 5)

# ---------------- 收集 4/5 月文件 ----------------
items = []
for year in YEARS:
    for month in MONTHS:
        path = SRC_DIR / f"河南SSI_{year}{month:02d}.tif"
        if not path.exists():
            raise FileNotFoundError(f"缺少文件: {path}")
        items.append((f"{year}-{month:02d}", path))

print(f"共收集 {len(items)} 个栅格: {items[0][0]} ~ {items[-1][0]}")

# ---------------- 校验网格一致性 ----------------
with rasterio.open(items[0][1]) as ref:
    ref_meta = (ref.width, ref.height, ref.transform, ref.crs, ref.nodata)
    profile = ref.profile.copy()

for label, path in items[1:]:
    with rasterio.open(path) as src:
        cur = (src.width, src.height, src.transform, src.crs, src.nodata)
        if cur != ref_meta:
            raise ValueError(f"网格不一致: {path.name}")

# ---------------- 堆叠写出多波段 tif ----------------
profile.update(
    count=len(items),
    dtype="float32",
    nodata=ref_meta[4],
    compress="deflate",
)

OUT_DIR.mkdir(parents=True, exist_ok=True)
with rasterio.open(OUT_PATH, "w", **profile) as dst:
    for idx, (label, path) in enumerate(items, start=1):
        with rasterio.open(path) as src:
            dst.write(src.read(1), idx)
        dst.set_band_description(idx, label)
        print(f"[{idx}/{len(items)}] 波段 {idx} <- {label}")

print(f"\n完成! 输出: {OUT_PATH}")