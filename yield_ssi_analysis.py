# -*- coding: utf-8 -*-
"""
SSI 与冬小麦(夏粮)产量关联分析
----------------------------------
1. 各市产量去趋势 (一元线性回归): 产量距平 = 实际产量 - 趋势产量
2. SSI 栅格 (4-5月, 2016-2025) 按地级市边界提取每年 4-5 月平均 SSI
3. 按 年份+城市 合并 -> Pearson 相关 (r, p)
4. 按干旱等级分组平均减产率: 轻度 [-1.0,-0.5) / 中度 [-1.5,-1.0) / 重度 (<=-1.5)
5. 输出 city_yield_analysis.csv + 控制台明细
"""
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
from scipy import stats

ROOT = Path(__file__).resolve().parent
YIELD_XLSX = Path(r"d:\桌面\henan_wheat_yield.xlsx")   # 年份/城市/夏粮产量(万吨), 2016-2024
SSI_TIF = ROOT / "4-5月河南SSI" / "4-5月河南SSI_2016-2025.tif"   # 20 波段: 每年 4,5 月
CITY_SHP = ROOT / "河南省市shp" / "河南省.shp"
OUT_CSV = ROOT / "city_yield_analysis.csv"
OUT_TS = ROOT / "city_yield_ssi_timeseries.csv"   # 逐年 SSI x 产量距平 (平台"产量影响评估"模块数据源)


def norm_city(name):
    """城市名规整: 去"示范区"/"市"后缀, 对齐 Excel/shp 两侧 (济源示范区/济源市 -> 济源)"""
    n = str(name).strip()
    for suf in ("示范区", "市"):
        if n.endswith(suf) and len(n) > len(suf):
            return n[: -len(suf)]
    return n


# ---------- 1. 产量数据: 去趋势 -> 距平与减产率 ----------
def load_yield_anomaly():
    """各市产量对年份线性回归剔除技术趋势, 返回 DataFrame[年份, 城市, 实际产量, 趋势产量, 距平, 减产率%]"""
    df = pd.read_excel(YIELD_XLSX)
    df.columns = ["年份", "城市", "产量"]
    df["年份"] = df["年份"].astype(int)
    df["key"] = df["城市"].map(norm_city)
    records = []
    for key, g in df.groupby("key"):
        g = g.sort_values("年份")
        n = len(g)
        note = ""
        if n < 3:  # 点太少无法拟合趋势
            note = f"样本不足({n}年), 跳过趋势剔除"
            print(f"[缺失] {g['城市'].iloc[0]}: {note}")
            continue
        y = g["产量"].to_numpy(float)
        x = g["年份"].to_numpy(float)
        a, b = np.polyfit(x, y, 1)          # 线性趋势产量
        trend = a * x + b
        anom = y - trend                     # 产量距平 (气象产量)
        loss = anom / trend * 100.0          # 减产率% (负=减产)
        for yr, yi, ti, ai, li in zip(x.astype(int), y, trend, anom, loss):
            records.append({"年份": yr, "key": key, "城市": g["城市"].iloc[0],
                            "实际产量": yi, "趋势产量": ti, "距平": ai, "减产率%": li})
    return pd.DataFrame(records)

# ---------- 2. SSI 栅格: 按市边界提取每年 4-5 月平均 ----------
def load_city_ssi():
    """返回 DataFrame[年份, key, SSI]; SSI = 该市像元当年 4月+5月波段均值"""
    gdf = gpd.read_file(CITY_SHP, encoding="utf-8")
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)

    with rasterio.open(SSI_TIF) as src:
        cube = src.read().astype(np.float64)          # (20, rows, cols)
        nodata = src.nodata
        transform, h, w = src.transform, src.height, src.width
    if nodata is not None:
        cube[cube <= nodata] = np.nan

    years = list(range(2016, 2026))                    # 波段 2i=4月, 2i+1=5月
    records = []
    for _, row in gdf.iterrows():
        name = str(row["name"])
        key = name[:-1] if name.endswith("市") else name
        # all_touched=True: 网格较粗, 保证小市(如济源)也能覆盖到像元
        mask = geometry_mask([row.geometry], out_shape=(h, w),
                             transform=transform, invert=True, all_touched=True)
        for i, yr in enumerate(years):
            band = cube[2 * i: 2 * i + 2]              # 当年 4、5 月
            vals = band[:, mask]
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue                               # 该市无有效像元 -> 该年缺测
            records.append({"年份": yr, "key": key,
                            "SSI": float(vals.mean()), "有效像元": int(vals.size // 2)})
    return pd.DataFrame(records)

# ---------- 主流程 ----------
yield_df = load_yield_anomaly()
ssi_df = load_city_ssi()
mg = yield_df.merge(ssi_df[["年份", "key", "SSI"]], on=["年份", "key"], how="left")

# ---------- 逐年时间序列表 (含 2025 年仅有 SSI 的行, 供平台双Y轴图使用) ----------
ts = ssi_df[["年份", "key", "SSI"]].merge(
    yield_df[["年份", "key", "城市", "实际产量", "趋势产量", "距平", "减产率%"]],
    on=["年份", "key"], how="left")
_name_map = yield_df.drop_duplicates("key").set_index("key")["城市"]
ts["城市"] = ts["城市"].fillna(ts["key"].map(_name_map))
ts = ts.sort_values(["key", "年份"])
ts.to_csv(OUT_TS, index=False, encoding="utf-8-sig")
print(f"时间序列表已保存: {OUT_TS}  ({len(ts)} 行)")

print("\n================ SSI x 产量距平: 各市结果 ================")
rows = []
for key, g in mg.groupby("key"):
    city = g["城市"].iloc[0]
    g = g.dropna(subset=["SSI"])
    n = len(g)
    if n < 3:
        print(f"{city:<6} [缺失] 有效配对样本 {n} 年, 无法计算相关系数")
        rows.append({"城市": city, "有效年数": n, "Pearson_r": np.nan, "p值": np.nan,
                     "显著性(p<0.05)": "数据不足",
                     "轻度平均减产率%": np.nan, "轻度样本": 0,
                     "中度平均减产率%": np.nan, "中度样本": 0,
                     "重度平均减产率%": np.nan, "重度样本": 0, "备注": "有效配对样本不足"})
        continue
    r, p = stats.pearsonr(g["SSI"], g["距平"])
    sig = "*" if p < 0.05 else ""
    # 干旱等级分组 (只统计 SSI<-0.5 的年份): 轻 [-1.0,-0.5) 中 [-1.5,-1.0) 重 <=-1.5
    bins = {"轻度": (-1.0, -0.5), "中度": (-1.5, -1.0), "重度": (-np.inf, -1.5)}
    grp = {}
    for lvl, (lo, hi) in bins.items():
        sub = g[(g["SSI"] >= lo) & (g["SSI"] < hi)]
        grp[lvl] = (sub["减产率%"].mean() if len(sub) else np.nan, len(sub))
    print(f"{city:<6} r={r:+.3f}  p={p:.3f}{sig}  n={n}年  "
          f"轻度:{grp['轻度'][0] if np.isnan(grp['轻度'][0]) else format(grp['轻度'][0], '+.1f')}%({grp['轻度'][1]})  "
          f"中度:{grp['中度'][0] if np.isnan(grp['中度'][0]) else format(grp['中度'][0], '+.1f')}%({grp['中度'][1]})  "
          f"重度:{grp['重度'][0] if np.isnan(grp['重度'][0]) else format(grp['重度'][0], '+.1f')}%({grp['重度'][1]})")
    rows.append({"城市": city, "有效年数": n, "Pearson_r": r, "p值": p,
                 "显著性(p<0.05)": "显著" if p < 0.05 else "不显著",
                 "轻度平均减产率%": grp["轻度"][0], "轻度样本": grp["轻度"][1],
                 "中度平均减产率%": grp["中度"][0], "中度样本": grp["中度"][1],
                 "重度平均减产率%": grp["重度"][0], "重度样本": grp["重度"][1],
                 "备注": "" if n == 9 else f"仅{n}年有效配对"})

# 全省汇总 (所有市-年观测按等级合并, 统计意义更强)
print("\n---------------- 全省汇总 (全部市-年观测) ----------------")
g_all = mg.dropna(subset=["SSI"])
r_all, p_all = stats.pearsonr(g_all["SSI"], g_all["距平"])
print(f"全部观测 n={len(g_all)}, Pearson r={r_all:+.3f}, p={p_all:.2e}")
for lvl, (lo, hi) in {"轻度": (-1.0, -0.5), "中度": (-1.5, -1.0), "重度": (-np.inf, -1.5)}.items():
    sub = g_all[(g_all["SSI"] >= lo) & (g_all["SSI"] < hi)]
    if len(sub):
        print(f"{lvl}干旱 (SSI∈[{lo},{hi})): 平均减产率 {sub['减产率%'].mean():+.1f}%  (n={len(sub)})")
    else:
        print(f"{lvl}干旱 (SSI∈[{lo},{hi})): 无样本")
rows.append({"城市": "全省汇总", "有效年数": len(g_all), "Pearson_r": r_all, "p值": p_all,
             "显著性(p<0.05)": "显著" if p_all < 0.05 else "不显著",
             "轻度平均减产率%": g_all[(g_all.SSI >= -1.0) & (g_all.SSI < -0.5)]["减产率%"].mean(),
             "轻度样本": int(((g_all.SSI >= -1.0) & (g_all.SSI < -0.5)).sum()),
             "中度平均减产率%": g_all[(g_all.SSI >= -1.5) & (g_all.SSI < -1.0)]["减产率%"].mean(),
             "中度样本": int(((g_all.SSI >= -1.5) & (g_all.SSI < -1.0)).sum()),
             "重度平均减产率%": g_all[g_all.SSI < -1.5]["减产率%"].mean(),
             "重度样本": int((g_all.SSI < -1.5).sum()), "备注": "全部市-年观测合并"})

out = pd.DataFrame(rows)
out.insert(1, "key", out["城市"].map(norm_city))   # 规整名, 供平台按 shp 城市匹配
out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
print(f"\n结果已保存: {OUT_CSV}")
