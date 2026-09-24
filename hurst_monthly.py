# -*- coding: utf-8 -*-
"""
对河南省 SSI 全部 120 个月逐月序列 (2016-01 -> 2025-12) 逐像元计算 Hurst 指数
序列长度 120, 满足 hurst 包 compute_Hc 的长度要求 (>= 100)
方法: hurst 包 compute_Hc, kind='random_walk', simplified=True
输入: d:\\大二\\干旱预测\\河南省SSI\\河南SSI_YYYYMM.tif (201601-202512)
输出: d:\\大二\\干旱预测\\hurst指数\\河南SSI_逐月_Hurst指数_2016-2025.tif
依赖: rasterio, numpy, hurst
"""
from pathlib import Path

import numpy as np
import rasterio
from hurst import compute_Hc

SRC_DIR = Path(r"d:\大二\干旱预测\河南省SSI")
OUT_DIR = Path(r"d:\大二\干旱预测\hurst指数")
OUT_PATH = OUT_DIR / "河南SSI_逐月_Hurst指数_2016-2025.tif"
OUT_NODATA = -9999.0

YEARS = list(range(2016, 2026))
MONTHS = list(range(1, 13))

# ---------------- 读取 120 个月栅格 ----------------
items = []
for y in YEARS:
    for m in MONTHS:
        path = SRC_DIR / f"河南SSI_{y}{m:02d}.tif"
        if not path.exists():
            raise FileNotFoundError(f"缺少文件: {path}")
        items.append((y, m, path))

with rasterio.open(items[0][2]) as ref:
    profile = ref.profile.copy()
    ref_grid = (ref.width, ref.height, ref.transform, ref.crs, ref.nodata)
    cube = np.empty((len(items), ref.height, ref.width), dtype=np.float64)

for idx, (y, m, path) in enumerate(items):
    with rasterio.open(path) as src:
        cur = (src.width, src.height, src.transform, src.crs, src.nodata)
        if cur != ref_grid:
            raise ValueError(f"网格不一致: {path.name}")
        cube[idx] = src.read(1)

print(f"读取 {len(items)} 个月度栅格, 像元 {cube.shape[2]} x {cube.shape[1]}")

# ---------------- 标记无效值 ----------------
nodata = ref_grid[4]
valid = cube > nodata if nodata is not None else np.ones_like(cube, dtype=bool)
n_valid = valid.sum(axis=0)
rows, cols = np.where(n_valid == cube.shape[0])
print(f"待计算像元: {len(rows)} 个 (需 {cube.shape[0]} 个波段全部有效) ...")

# ---------------- 逐像元 Hurst 指数 ----------------
h_arr = np.full(cube.shape[1:], OUT_NODATA, dtype=np.float32)

n_ok, n_fail = 0, 0
for r, c in zip(rows, cols):
    series = cube[:, r, c]
    try:
        h, c_val, snr = compute_Hc(series, kind="random_walk", simplified=True)
        if np.isfinite(h):
            h_arr[r, c] = h
            n_ok += 1
        else:
            n_fail += 1
    except Exception as e:
        n_fail += 1
        print(f"  像元 ({r},{c}) 计算失败: {e}")

# ---------------- 输出 ----------------
profile.update(count=1, dtype="float32", nodata=OUT_NODATA, compress="deflate")
OUT_DIR.mkdir(parents=True, exist_ok=True)
with rasterio.open(OUT_PATH, "w", **profile) as dst:
    dst.write(h_arr, 1)
    dst.set_band_description(1, "Hurst 指数 (compute_Hc, 120 个月)")

m = h_arr > OUT_NODATA / 2
print(f"\n结果摘要 (成功 {n_ok} 个, 失败 {n_fail} 个):")
if m.any():
    print(f"  Hurst 指数范围: {h_arr[m].min():.4f} ~ {h_arr[m].max():.4f}, 均值 {h_arr[m].mean():.4f}")
    print(f"  H > 0.5 (持续性): {int((h_arr[m] > 0.5).sum())} 个")
    print(f"  H < 0.5 (反持续性): {int((h_arr[m] < 0.5).sum())} 个")
    print(f"  |H - 0.5| < 0.05 (近随机): {int((np.abs(h_arr[m] - 0.5) < 0.05).sum())} 个")
else:
    print("  无有效结果")
print(f"\n完成! 输出: {OUT_PATH}")