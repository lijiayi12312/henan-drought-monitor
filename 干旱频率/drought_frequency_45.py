# -*- coding: utf-8 -*-
"""
基于 4-5 月河南省 SSI 多波段栅格, 计算逐像元干旱频率
干旱等级划分:
    SSI > 0                     不干旱
    -1.0 < SSI <= 0             轻度干旱
    -1.5 < SSI <= -1.0          中度干旱
    -2.0 < SSI <= -1.5          重度干旱
    SSI <= -2.0                 极端干旱
干旱频率 = 像元 SSI <= 0 的波段数 / 有效波段数
输入: d:\\大二\\干旱预测\\4-5月河南SSI\\4-5月河南SSI_2016-2025.tif
输出: d:\\大二\\干旱预测\\干旱频率\\河南SSI_4-5月_干旱频率_2016-2025.tif
依赖: rasterio, numpy
"""
from pathlib import Path

import numpy as np
import rasterio

SRC_PATH = Path(r"d:\大二\干旱预测\4-5月河南SSI\4-5月河南SSI_2016-2025.tif")
OUT_DIR = Path(r"d:\大二\干旱预测\干旱频率")
OUT_PATH = OUT_DIR / "河南SSI_4-5月_干旱频率_2016-2025.tif"
OUT_NODATA = -9999.0

with rasterio.open(SRC_PATH) as src:
    cube = src.read()  # (波段, 行, 列)
    profile = src.profile.copy()
    nodata = src.nodata
    labels = src.descriptions

n_band = cube.shape[0]
print(f"读取 {SRC_PATH.name}: {n_band} 个波段, 像元 {cube.shape[2]} x {cube.shape[1]}")

# ---------------- 标记无效值 ----------------
valid = cube > nodata if nodata is not None else np.ones_like(cube, dtype=bool)
if not valid.all(axis=0).all():
    n_valid_band = valid.sum(axis=0)
    print(f"存在无效值像元: {int((n_valid_band < n_band).sum())} 个 (不足 {n_band} 个有效波段)")

# ---------------- 干旱等级划分 ----------------
# 1=不干旱, 2=轻度, 3=中度, 4=重度, 5=极端, 0=无效
level = np.zeros_like(cube, dtype=np.uint8)
level[valid] = 1
level[valid & (cube <= 0)] = 2
level[valid & (cube <= -1.0)] = 3
level[valid & (cube <= -1.5)] = 4
level[valid & (cube <= -2.0)] = 5

names = {1: "不干旱", 2: "轻度干旱", 3: "中度干旱", 4: "重度干旱", 5: "极端干旱"}
for code, name in names.items():
    print(f"  {name}: {int((level == code).sum())} 个 (波段,像元)")

# ---------------- 干旱频率 ----------------
dry = valid & (cube <= 0)
n_dry = dry.sum(axis=0)
n_ok = valid.sum(axis=0)

freq = np.full(n_dry.shape, OUT_NODATA, dtype=np.float32)
mask = n_ok > 0
freq[mask] = n_dry[mask] / n_ok[mask]

profile.update(count=1, dtype="float32", nodata=OUT_NODATA, compress="deflate")
OUT_DIR.mkdir(parents=True, exist_ok=True)
with rasterio.open(OUT_PATH, "w", **profile) as dst:
    dst.write(freq, 1)
    dst.set_band_description(1, "干旱频率(SSI<=0)")

print(f"\n干旱频率统计 (有效像元 {int(mask.sum())} / {mask.size}):")
print(f"  最小 {freq[mask].min():.4f}  最大 {freq[mask].max():.4f}  均值 {freq[mask].mean():.4f}")
print(f"\n完成! 输出: {OUT_PATH}")