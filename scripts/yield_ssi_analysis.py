# -*- coding: utf-8 -*-
"""
SSI 与冬小麦单产关联分析 (单产口径)
----------------------------------
0. 实际单产(吨/公顷) = 冬小麦总产(万吨) / 播种面积(千公顷) × 10
   (数据源: henan_wheat_yield_clean.csv 勘误版总产 + 河南省各市冬小麦播种面积.xlsx, 同论文口径)
1. 各市单产去趋势 (一元线性回归): 单产距平 = 实际单产 - 趋势单产
2. SSI 栅格 (4-5月, 2016-2025) 按地级市边界提取每年 4-5 月平均 SSI
3. 按 年份+城市 合并 -> Pearson 相关 (r, p)
4. 按干旱等级分组平均减产率: 轻度 [-1.0,-0.5) / 中度 [-1.5,-1.0) / 重度 (<=-1.5)
5. 输出 city_yield_analysis.csv + 时序表 city_yield_ssi_timeseries.csv (平台双Y轴图数据源)
"""
from pathlib import Path
import re

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
YIELD_CSV = ROOT / "data" / "henan_wheat_yield_clean.csv"   # 冬小麦总产(万吨), 18市×2016-2024, 勘误版
AREA_XLSX = ROOT / "data" / "河南省各市冬小麦播种面积.xlsx"   # 宽表: 行=各市, 列=2016-2024年, 值=千公顷
SSI_TIF = ROOT / "data" / "4-5月河南SSI" / "4-5月河南SSI_swvl1_2016-2025.tif"   # 20 波段: 每年 4,5 月 (原 4-5月河南SSI_2016-2025.tif, 已并入 _swvl1 命名)
CITY_SHP = ROOT / "data" / "河南省市shp" / "河南省.shp"
OUT_CSV = ROOT / "output" / "产量关联分析" / "city_yield_analysis.csv"
OUT_TS = ROOT / "output" / "产量关联分析" / "city_yield_ssi_timeseries.csv"   # 逐年 SSI x 产量距平 (平台"产量影响评估"模块数据源)


def norm_city(name):
    """城市名规整: 去"示范区"/"市"后缀, 对齐 Excel/shp 两侧 (济源示范区/济源市 -> 济源)"""
    n = "".join(str(name).split())   # 去内部空格(如"郑  州  市"->"郑州市")
    for suf in ("示范区", "市"):
        if n.endswith(suf) and len(n) > len(suf):
            return n[: -len(suf)]
    return n


# ---------- 0. 播种面积宽表 -> 长表 (逻辑同论文项目 12_plots_final.load_area) ----------
def load_area():
    """返回 DataFrame[key, 年份, 面积(千公顷)]"""
    raw = pd.read_excel(AREA_XLSX, header=None)
    year_cols = {}
    for j, v in enumerate(raw.iloc[1]):
        m = re.fullmatch(r"(\d{4})年", str(v).strip()) if pd.notna(v) else None
        if m:
            year_cols[int(m.group(1))] = j
    assert sorted(year_cols) == list(range(2016, 2025)), f"面积表年份列异常: {sorted(year_cols)}"
    records = []
    for i in range(2, len(raw)):
        name = raw.iloc[i, 0]
        if pd.isna(name):
            continue
        key = norm_city(name)
        if key in ("全省", "各市之和"):
            continue
        for yr, j in year_cols.items():
            records.append({"key": key, "年份": yr,
                            "面积": pd.to_numeric(raw.iloc[i, j], errors="coerce")})
    df = pd.DataFrame(records)
    assert df["key"].nunique() == 18 and len(df) == 162, f"面积表应为18市×9年=162行, 实际{len(df)}"
    assert df["面积"].notna().all() and (df["面积"] > 0).all(), "面积存在缺失或非正值"
    return df


# ---------- 1. 单产数据: 去趋势 -> 距平与减产率 ----------
def load_yield_anomaly():
    """单产口径 (同论文): 实际单产(吨/公顷) = 总产(万吨)/面积(千公顷)×10;
    各市对年份线性回归剔除技术趋势, 返回 DataFrame[年份, key, 城市, 实际单产, 趋势单产, 单产距平, 减产率%]"""
    ydf = pd.read_csv(YIELD_CSV)
    ydf = ydf[["city", "key", "year", "yield"]].rename(
        columns={"city": "城市", "year": "年份", "yield": "总产"})
    ydf["年份"] = ydf["年份"].astype(int)
    mg = ydf.merge(load_area(), on=["key", "年份"], how="inner")
    assert len(mg) == 162, f"产量×面积合并后应为162行, 实际{len(mg)}"
    mg["实际单产"] = mg["总产"] / mg["面积"] * 10.0
    records = []
    for key, g in mg.groupby("key"):
        g = g.sort_values("年份")
        if len(g) < 3:  # 点太少无法拟合趋势
            print(f"[缺失] {g['城市'].iloc[0]}: 样本不足({len(g)}年), 跳过趋势剔除")
            continue
        y = g["实际单产"].to_numpy(float)
        x = g["年份"].to_numpy(float)
        a, b = np.polyfit(x, y, 1)          # 线性趋势单产
        trend = a * x + b
        anom = y - trend                     # 单产距平 (气象单产)
        loss = anom / trend * 100.0          # 减产率% (负=减产)
        for yr, yi, ti, ai, li in zip(x.astype(int), y, trend, anom, loss):
            records.append({"年份": yr, "key": key, "城市": g["城市"].iloc[0],
                            "实际单产": yi, "趋势单产": ti, "单产距平": ai, "减产率%": li})
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
    yield_df[["年份", "key", "城市", "实际单产", "趋势单产", "单产距平", "减产率%"]],
    on=["年份", "key"], how="left")
_name_map = yield_df.drop_duplicates("key").set_index("key")["城市"]
ts["城市"] = ts["城市"].fillna(ts["key"].map(_name_map))
ts = ts.sort_values(["key", "年份"])
ts.to_csv(OUT_TS, index=False, encoding="utf-8-sig")
print(f"时间序列表已保存: {OUT_TS}  ({len(ts)} 行)")

print("\n================ SSI x 单产距平: 各市结果 ================")
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
    r, p = stats.pearsonr(g["SSI"], g["单产距平"])
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
r_all, p_all = stats.pearsonr(g_all["SSI"], g_all["单产距平"])
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
