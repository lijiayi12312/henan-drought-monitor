# -*- coding: utf-8 -*-
"""
对 4-5 月河南省 SSI 用游程理论 (run theory) 逐像元识别干旱事件 (按真实时间)
序列: 每个像元 20 个波段 (2016-04 -> 2025-05), 每年 [4月, 5月] 成对
规则:
    干旱月: SSI <= -1.0
    同年 4 月与 5 月均 <= -1.0: 一次持续 2 个月的事件
    仅 4 月或仅 5 月 <= -1.0: 一次持续 1 个月的独立事件
    不同年份之间绝不合并 (跨年 5 月与次年 4 月视为不相邻)
    累积强度: -sum(SSI) (事件内各月, 正值越大越强)
输入: d:\\大二\\干旱预测\\4-5月河南SSI\\4-5月河南SSI_2016-2025.tif
输出: d:\\大二\\干旱预测\\游程理论\\河南SSI_4-5月_干旱事件_2016-2025.csv (覆盖旧版)
依赖: rasterio, numpy
"""
import csv
from pathlib import Path

import numpy as np
import rasterio

SRC_PATH = Path(r"d:\大二\干旱预测\4-5月河南SSI\4-5月河南SSI_2016-2025.tif")
OUT_DIR = Path(r"d:\大二\干旱预测\游程理论")
OUT_PATH = OUT_DIR / "河南SSI_4-5月_干旱事件_按真实时间_2016-2025.csv"

DRY_THR = -1.0  # 干旱阈值 (含)

with rasterio.open(SRC_PATH) as src:
    cube = src.read()  # (20, 行, 列)
    nodata = src.nodata
    labels = list(src.descriptions)
    transform = src.transform

print(f"读取 {SRC_PATH.name}: {cube.shape[0]} 个波段, 像元 {cube.shape[2]} x {cube.shape[1]}")

# 波段顺序校验: 必须为 2016-04, 2016-05, ..., 2025-05
expected = [f"{y}-{m:02d}" for y in range(2016, 2026) for m in (4, 5)]
if labels != expected:
    raise ValueError(f"波段顺序与预期不符: {labels}")

valid = cube > nodata if nodata is not None else np.ones_like(cube, dtype=bool)
n_valid = valid.sum(axis=0)
rows, cols = np.where(n_valid == cube.shape[0])
print(f"待分析像元: {len(rows)} 个 ...")

# ---------------- 逐像元识别事件 ----------------
events_rows = []
n_event_px = 0
for r, c in zip(rows, cols):
    x = cube[:, r, c].astype(np.float64)
    paired = x.reshape(10, 2)  # 每年 [4月, 5月]

    px_events = []
    for yi in range(10):
        apr_d = paired[yi, 0] <= DRY_THR
        may_d = paired[yi, 1] <= DRY_THR
        if apr_d and may_d:
            s, e = 2 * yi, 2 * yi + 1  # 4-5 月连旱, 持续 2 个月
        elif apr_d:
            s = e = 2 * yi            # 仅 4 月, 持续 1 个月
        elif may_d:
            s = e = 2 * yi + 1        # 仅 5 月, 持续 1 个月
        else:
            continue
        px_events.append((s, e))

    if px_events:
        n_event_px += 1
    lon, lat = transform * (c + 0.5, r + 0.5)
    for s, e in px_events:
        dur = e - s + 1
        severity = float(-x[s : e + 1].sum())
        events_rows.append(
            [int(r), int(c), round(lon, 4), round(lat, 4),
             labels[s], labels[e], dur, round(severity, 4)]
        )

# ---------------- 输出 CSV ----------------
OUT_DIR.mkdir(parents=True, exist_ok=True)
header = ["行", "列", "经度", "纬度", "起始月份", "结束月份", "持续月数", "累积强度"]
with open(OUT_PATH, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(header)
    w.writerows(events_rows)

# ---------------- 统计摘要 ----------------
durs = np.array([row[6] for row in events_rows], dtype=float)
sevs = np.array([row[7] for row in events_rows], dtype=float)
n_1m = int((durs == 1).sum())
n_2m = int((durs == 2).sum())
print(f"\n统计摘要:")
print(f"  发生事件的像元: {n_event_px} / {len(rows)}")
print(f"  事件总数: {len(events_rows)}")
if len(events_rows):
    print(f"  持续 1 个月: {n_1m} 个 | 持续 2 个月 (同年 4-5 月连旱): {n_2m} 个")
    print(f"  累积强度: 1 月事件 {sevs[durs == 1].min():.2f} ~ {sevs[durs == 1].max():.2f} (均值 {sevs[durs == 1].mean():.2f})"
          if n_1m else "  无 1 月事件")
    if n_2m:
        print(f"  累积强度: 2 月事件 {sevs[durs == 2].min():.2f} ~ {sevs[durs == 2].max():.2f} (均值 {sevs[durs == 2].mean():.2f})")
    from collections import Counter
    by_year = Counter(row[4][:4] for row in events_rows)
    print(f"  分年事件数: " + ", ".join(f"{y}年{by_year[y]}" for y in sorted(by_year)))
print(f"\n完成! 输出: {OUT_PATH}")