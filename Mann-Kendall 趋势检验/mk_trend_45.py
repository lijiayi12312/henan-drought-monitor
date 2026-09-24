# -*- coding: utf-8 -*-
"""
对 4-5 月河南省 SSI 做逐像元 Mann-Kendall 趋势检验
年序列构造: 每年取 4、5 月两波段的均值, 得 2016-2025 共 10 点
检验方法: pymannkendall.original_test, 输出 Z 值、p 值、Sen's slope
输入: d:\\大二\\干旱预测\\4-5月河南SSI\\4-5月河南SSI_2016-2025.tif
输出: d:\\大二\\干旱预测\\Mann-Kendall 趋势检验\\ 下三个 GeoTIFF
依赖: rasterio, numpy, pymannkendall
"""
from pathlib import Path

import numpy as np
import pymannkendall as mk
import rasterio

SRC_PATH = Path(r"d:\大二\干旱预测\4-5月河南SSI\4-5月河南SSI_2016-2025.tif")
OUT_DIR = Path(r"d:\大二\干旱预测\Mann-Kendall 趋势检验")
OUT_NODATA = -9999.0

YEARS = list(range(2016, 2026))

with rasterio.open(SRC_PATH) as src:
    cube = src.read().astype(np.float64)  # (20, 行, 列)
    profile = src.profile.copy()
    nodata = src.nodata
    labels = src.descriptions

n_band = cube.shape[0]
n_year = len(YEARS)
print(f"读取 {SRC_PATH.name}: {n_band} 个波段, 像元 {cube.shape[2]} x {cube.shape[1]}")

# ---------------- 构造年序列: 每年 4、5 月均值 ----------------
# 波段顺序为 2016-04, 2016-05, 2017-04, ... 与 labels 一致, 校验后重排为 (年, 月, 行, 列)
order = [f"{y}-{m:02d}" for y in YEARS for m in (4, 5)]
if tuple(labels) != tuple(order):
    raise ValueError(f"波段顺序与预期不符: {labels}")

cube = cube.reshape(n_year, 2, cube.shape[1], cube.shape[2])
valid = cube > nodata if nodata is not None else np.ones_like(cube, dtype=bool)

with np.errstate(invalid="ignore"):
    cube[~valid] = np.nan
    annual = np.nanmean(cube, axis=1)  # (年, 行, 列), 每年 4/5 月均值

n_valid_year = np.sum(~np.isnan(annual), axis=0)
print(f"年序列年数: {n_year}, 有效像元: {int((n_valid_year == n_year).sum())} / {n_valid_year.size}")

# ---------------- 逐像元 Mann-Kendall 检验 ----------------
z_arr = np.full(annual.shape[1:], OUT_NODATA, dtype=np.float32)
p_arr = np.full_like(z_arr, OUT_NODATA)
slope_arr = np.full_like(z_arr, OUT_NODATA)

rows, cols = np.where(n_valid_year == n_year)
print(f"待检验像元: {len(rows)} 个 ...")

n_sig, n_pos, n_neg = 0, 0, 0
for r, c in zip(rows, cols):
    res = mk.original_test(annual[:, r, c], alpha=0.05)
    z_arr[r, c] = res.z
    p_arr[r, c] = res.p
    slope_arr[r, c] = res.slope
    if res.p < 0.05:
        n_sig += 1
        n_pos += res.slope > 0
        n_neg += res.slope < 0

# ---------------- 输出三个 GeoTIFF ----------------
profile.update(count=1, dtype="float32", nodata=OUT_NODATA, compress="deflate")
OUT_DIR.mkdir(parents=True, exist_ok=True)

outputs = [
    (z_arr, "MK_Z值", "Mann-Kendall Z"),
    (p_arr, "MK_p值", "Mann-Kendall p"),
    (slope_arr, "MK_Sen斜率", "Sen's slope (SSI/年)"),
]
for arr, name, desc in outputs:
    out_path = OUT_DIR / f"河南SSI_4-5月_{name}_2016-2025.tif"
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr, 1)
        dst.set_band_description(1, desc)
    print(f"已输出: {out_path.name}")

# ---------------- 结果摘要 ----------------
sig = p_arr[p_arr > OUT_NODATA / 2] < 0.05
z_valid = z_arr[z_arr > OUT_NODATA / 2]
sl_valid = slope_arr[slope_arr > OUT_NODATA / 2]
print(f"\n结果摘要 (有效像元 {len(z_valid)} 个):")
print(f"  Z 值范围: {z_valid.min():.3f} ~ {z_valid.max():.3f}")
print(f"  Sen's slope 范围: {sl_valid.min():.4f} ~ {sl_valid.max():.4f} (SSI/年)")
print(f"  p < 0.05 显著像元: {len(z_valid[sig])} 个 (上升 {n_pos}, 下降 {n_neg})")
print(f"\n完成! 输出目录: {OUT_DIR}")