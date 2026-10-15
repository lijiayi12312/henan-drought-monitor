# -*- coding: utf-8 -*-
"""
符号审计: 五层 SSI x 产量距平的相关系数 r 究竟是 -0.296 还是 +0.296
四条独立路径交叉验证 (以浅层根区 0-28cm 为主, 辅以 swvl1):
  A. 原始脚本复跑 (yield_ssi_analysis_layers.py, 20 波段堆叠)  -> r = -0.296 (已确认)
  B. 绕过 20 波段堆叠: 直接读 120 个月度 SSI tif 中的 4/5 月 -> 独立重算 r
  C. 已生成时序 CSV 直接 pearsonr (算术一致性检查)
  D. 像元级验证: 用原始浅层土壤水分月度 tif 现算 z=(x-mean)/std (ddof=0),
     与 SSI tif 逐像元比对 -> 证明 SSI = +z (链路中不存在取反)
最后打印全省逐年 SSI 均值 x 产量距平均值 + 单市明细, 定位负相关的驱动年份。
"""
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from scipy import stats

ROOT = Path(r"D:\大二\干旱预测")
YIELD_XLSX = Path(r"d:\桌面\henan_wheat_yield.xlsx")
CITY_SHP = ROOT / "河南省市shp" / "河南省.shp"
TS_CSV = ROOT / "产量关联分析" / "city_yield_ssi_timeseries_shallow.csv"

MONTHLY = {"shallow": ROOT / "河南省SSI" / "swvl_shallow_ssi",
           "swvl1": ROOT / "河南省SSI" / "swvl1"}
RAW_SM = ROOT / "河南省SSI" / "swvl_shallow"      # 浅层原始土壤水分月度 tif


def norm_city(name):
    n = str(name).strip()
    for suf in ("示范区", "市"):
        if n.endswith(suf) and len(n) > len(suf):
            return n[:-len(suf)]
    return n


def read_tif_nan(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float64)
        nod = src.nodata if src.nodata is not None else -3.4e38
        meta = (src.transform, src.height, src.width)
    arr[arr <= nod] = np.nan
    return arr, meta


# ---------- 产量距平 (与前期脚本完全一致的公式: 距平 = 实际 - 线性趋势) ----------
df = pd.read_excel(YIELD_XLSX)
df.columns = ["年份", "城市", "产量"]
df["年份"] = df["年份"].astype(int)
df["key"] = df["城市"].map(norm_city)
recs = []
for key, g in df.groupby("key"):
    g = g.sort_values("年份")
    x = g["年份"].to_numpy(float)
    y = g["产量"].to_numpy(float)
    a, b = np.polyfit(x, y, 1)
    for yr, yi, ti in zip(x.astype(int), y, a * x + b):
        recs.append({"年份": yr, "key": key, "实际产量": yi, "趋势产量": ti,
                     "距平": yi - ti, "减产率%": (yi - ti) / ti * 100.0})
yield_df = pd.DataFrame(recs)

# ---------- 城市边界 ----------
gdf = gpd.read_file(CITY_SHP, encoding="utf-8")
if gdf.crs and gdf.crs.to_epsg() != 4326:
    gdf = gdf.to_crs(4326)
CITIES = [(str(r["name"]), norm_city(r["name"]), r.geometry) for _, r in gdf.iterrows()]


def city_means(paths):
    """读若干单波段 tif -> 各市全时段均值 (all_touched, 与前期脚本一致)"""
    cube = []
    meta = None
    for p in paths:
        arr, meta = read_tif_nan(p)
        cube.append(arr)
    cube = np.stack(cube)
    transform, h, w = meta
    out = {}
    for _, key, geom in CITIES:
        mask = geometry_mask([geom], out_shape=(h, w), transform=transform,
                             invert=True, all_touched=True)
        vals = cube[:, mask]
        vals = vals[np.isfinite(vals)]
        out[key] = float(vals.mean()) if vals.size else np.nan
    return out


# ---------- 路径 B: 月度 tif 直读 -> 逐年各市 4-5 月 SSI 均值 ----------
print("=" * 76)
print("B. 绕过 20 波段堆叠, 直接读月度 SSI tif 重算相关")
print("=" * 76)
for tag, folder in MONTHLY.items():
    rows = []
    for yr in range(2016, 2026):
        means = city_means([folder / f"河南SSI_{yr}04.tif", folder / f"河南SSI_{yr}05.tif"])
        rows += [{"年份": yr, "key": k, "SSI": v} for k, v in means.items()]
    ssi_df = pd.DataFrame(rows)
    mg = yield_df.merge(ssi_df, on=["年份", "key"]).dropna(subset=["SSI"])
    r, p = stats.pearsonr(mg["SSI"], mg["距平"])
    print(f"  {tag:<8} n={len(mg)}  r={r:+.4f}  p={p:.3e}")
    if tag == "shallow":
        ssi_shallow = ssi_df
        mg_shallow = mg

# ---------- 路径 C: 现成时序 CSV 直接 pearsonr ----------
ts = pd.read_csv(TS_CSV, encoding="utf-8-sig")
tsy = ts.dropna(subset=["距平"])
rC, pC = stats.pearsonr(tsy["SSI"], tsy["距平"])
print(f"\nC. 时序CSV直算: n={len(tsy)}  r={rC:+.4f}  p={pC:.3e}")

# ---------- 路径 D: 像元级验证 SSI = +z (2025年4月, 浅层) ----------
print("\n" + "=" * 76)
print("D. 像元级验证: 用原始浅层土壤水分现算 z, 与 SSI tif 比对 (2025-04)")
print("=" * 76)
april_sm = []
for yr in range(2016, 2026):
    arr, meta = read_tif_nan(RAW_SM / f"河南浅层SM_{yr}04.tif")
    april_sm.append(arr)
samples = np.stack(april_sm)
mean = np.nanmean(samples, axis=0)
std = np.nanstd(samples, axis=0, ddof=0)
z = (samples[-1] - mean) / std          # 2025-04 的 z 分数
ssi_2504, _ = read_tif_nan(MONTHLY["shallow"] / "河南SSI_202504.tif")
ok = np.isfinite(z) & np.isfinite(ssi_2504)
diff = np.abs(z[ok] - ssi_2504[ok])
print(f"  有效像元 {ok.sum()} 个, |z现算 - SSI.tif| 最大 {diff.max():.4f}, 均值 {diff.mean():.2e}")
print(f"  corr(z现算, SSI.tif) = {np.corrcoef(z[ok], ssi_2504[ok])[0, 1]:.4f}  (≈1 且同号 => SSI=+z, 无取反)")
for i, (r_, c_) in enumerate(zip(*np.where(ok)[:3])):
    print(f"  像元样例 {i+1} (行{r_},列{c_}): 现算 z={z[r_, c_]:+.3f}  vs  SSI tif={ssi_2504[r_, c_]:+.3f}")

# ---------- 驱动分析: 全省逐年 SSI 均值 x 产量距平均值 ----------
print("\n" + "=" * 76)
print("驱动分析: 全省逐年 4-5月浅层SSI均值 x 产量距平均值 (2016-2024)")
print("=" * 76)
yearly = (mg_shallow.groupby("年份")
          .agg(SSI均值=("SSI", "mean"), 距平均值=("距平", "mean"),
               减产率均值=("减产率%", "mean"), n=("距平", "size"))
          .round(3))
print(yearly.to_string())
ry, py = stats.pearsonr(yearly["SSI均值"], yearly["距平均值"])
print(f"\n  逐年省级均值的相关: r={ry:+.3f} (n={len(yearly)}年)")

# 单市明细: 周口 (浅层 r=-0.613 的代表市)
print("\n周口市逐年明细 (浅层):")
zk = mg_shallow[mg_shallow["key"] == "周口"].sort_values("年份")
print(zk[["年份", "SSI", "实际产量", "趋势产量", "距平", "减产率%"]].round(2).to_string(index=False))
