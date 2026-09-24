# -*- coding: utf-8 -*-
"""
对 4-5 月河南省 SSI 逐像元计算 Hurst 指数
时间序列: 每个像元 20 个波段 (2016-04 -> 2025-05)
方法: 手写 R/S 分析 (重标极差法)
      注: hurst 包 compute_Hc 要求序列长度 >= 100, 不适用于 20 点序列, 故按预案手写
      与 hurst 包 kind='random_walk' 同语义: 直接对序列去均值后做 R/S, 不做累加
输入: d:\\大二\\干旱预测\\4-5月河南SSI\\4-5月河南SSI_2016-2025.tif
输出: d:\\大二\\干旱预测\\hurst指数\\河南SSI_4-5月_Hurst指数_2016-2025.tif
依赖: rasterio, numpy
"""
from pathlib import Path

import numpy as np
import rasterio

SRC_PATH = Path(r"d:\大二\干旱预测\4-5月河南SSI\4-5月河南SSI_2016-2025.tif")
OUT_DIR = Path(r"d:\大二\干旱预测\hurst指数")
OUT_PATH = OUT_DIR / "河南SSI_4-5月_Hurst指数_2016-2025.tif"
OUT_NODATA = -9999.0


def hurst_rs(x, min_n=4):
    """手写 R/S 分析估计 Hurst 指数
    x: 一维序列; 段长 n 取所有满足 N % n == 0 且 n >= min_n 的值
    每段: 去均值 -> 累计离差 -> 极差 R 与标准差 S -> R/S
    log(R/S) 对 log(n) 最小二乘拟合, 斜率即 H
    """
    x = np.asarray(x, dtype=np.float64)
    n_total = len(x)
    if np.allclose(x, x[0]):  # 常数序列无意义
        return np.nan
    ns = [n for n in range(min_n, n_total + 1) if n_total % n == 0]
    log_n, log_rs = [], []
    for n in ns:
        segs = x.reshape(-1, n)
        rs_vals = []
        for seg in segs:
            dev = np.cumsum(seg - seg.mean())
            r = dev.max() - dev.min()
            s = seg.std()
            if s > 0:
                rs_vals.append(r / s)
        if rs_vals:
            log_n.append(np.log(n))
            log_rs.append(np.log(np.mean(rs_vals)))
    if len(log_n) < 2:
        return np.nan
    slope, _ = np.polyfit(log_n, log_rs, 1)
    return slope


with rasterio.open(SRC_PATH) as src:
    cube = src.read()  # (20, 行, 列)
    profile = src.profile.copy()
    nodata = src.nodata

print(f"读取 {SRC_PATH.name}: {cube.shape[0]} 个波段, 像元 {cube.shape[2]} x {cube.shape[1]}")

# ---------------- 标记无效值 ----------------
valid = cube > nodata if nodata is not None else np.ones_like(cube, dtype=bool)
n_valid = valid.sum(axis=0)
rows, cols = np.where(n_valid == cube.shape[0])
print(f"待计算像元: {len(rows)} 个 (需 {cube.shape[0]} 个波段全部有效) ...")

# ---------------- 逐像元 Hurst 指数 ----------------
h_arr = np.full(cube.shape[1:], OUT_NODATA, dtype=np.float32)

n_ok, n_fail = 0, 0
for r, c in zip(rows, cols):
    h = hurst_rs(cube[:, r, c])
    if np.isfinite(h):
        h_arr[r, c] = h
        n_ok += 1
    else:
        n_fail += 1

# ---------------- 输出 ----------------
profile.update(count=1, dtype="float32", nodata=OUT_NODATA, compress="deflate")
OUT_DIR.mkdir(parents=True, exist_ok=True)
with rasterio.open(OUT_PATH, "w", **profile) as dst:
    dst.write(h_arr, 1)
    dst.set_band_description(1, "Hurst 指数 (R/S)")

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