# -*- coding: utf-8 -*-
"""
四套 4-5 月关键期 SSI (swvl1/swvl2/swvl3/根区加权) 与冬小麦产量关联分析
逻辑与 yield_ssi_analysis.py 完全一致, 仅参数化为多数据集:
  1. 各市产量线性去趋势 -> 产量距平 / 减产率%
  2. 按 20 波段 SSI tif + 市边界提取每市每年 4-5 月 SSI 均值 (all_touched)
  3. 每市 Pearson r / p 值; 旱涝双端分组平均减产率
     (干旱端 轻[-1.0,-0.5) 中[-1.5,-1.0) 重<-1.5 | 偏湿端 轻[+0.5,+1.0) 中[+1.0,+1.5) 重>=+1.5)
输出:
  产量关联分析\\city_yield_analysis_{swvl1,swvl2,swvl3,root}.csv  (含全省汇总行)
  产量关联分析\\层级产量关联对比表.csv  (四层汇总)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
from scipy import stats

ROOT = Path(r"D:\大二\干旱预测")
YIELD_XLSX = Path(r"d:\桌面\henan_wheat_yield.xlsx")   # 年份/城市/夏粮产量(万吨), 2016-2024
CITY_SHP = ROOT / "河南省市shp" / "河南省.shp"
OUT_DIR = ROOT / "产量关联分析"

DATASETS = [
    ("swvl1", "swvl1（表层 0-7cm）", ROOT / "4-5月河南SSI" / "4-5月河南SSI_swvl1_2016-2025.tif"),
    ("swvl2", "swvl2（浅层 7-28cm）", ROOT / "4-5月河南SSI" / "4-5月河南SSI_swvl2_2016-2025.tif"),
    ("shallow", "浅层根区（0-28cm 平均）", ROOT / "4-5月河南SSI" / "4-5月河南SSI_shallow_2016-2025.tif"),
    ("swvl3", "swvl3（深层 28-100cm）", ROOT / "4-5月河南SSI" / "4-5月河南SSI_swvl3_2016-2025.tif"),
    ("root", "根区（0-100cm 加权）", ROOT / "4-5月河南SSI" / "4-5月河南SSI_root_2016-2025.tif"),
]


def norm_city(name):
    """城市名规整: 去"示范区"/"市"后缀, 对齐 Excel/shp 两侧 (济源示范区/济源市 -> 济源)"""
    n = str(name).strip()
    for suf in ("示范区", "市"):
        if n.endswith(suf) and len(n) > len(suf):
            return n[: -len(suf)]
    return n


def load_yield_anomaly():
    """各市产量对年份线性回归剔除技术趋势 (逻辑同 yield_ssi_analysis.py)"""
    df = pd.read_excel(YIELD_XLSX)
    df.columns = ["年份", "城市", "产量"]
    df["年份"] = df["年份"].astype(int)
    df["key"] = df["城市"].map(norm_city)
    records = []
    for key, g in df.groupby("key"):
        g = g.sort_values("年份")
        n = len(g)
        if n < 3:
            print(f"[缺失] {g['城市'].iloc[0]}: 样本不足({n}年), 跳过趋势剔除")
            continue
        y = g["产量"].to_numpy(float)
        x = g["年份"].to_numpy(float)
        a, b = np.polyfit(x, y, 1)
        trend = a * x + b
        for yr, yi, ti in zip(x.astype(int), y, trend):
            records.append({"年份": yr, "key": key, "城市": g["城市"].iloc[0],
                            "实际产量": yi, "趋势产量": ti,
                            "距平": yi - ti, "减产率%": (yi - ti) / ti * 100.0})
    return pd.DataFrame(records)


def load_city_ssi(ssi_tif):
    """按市边界提取每年 4-5 月 SSI 均值 (逻辑同 yield_ssi_analysis.py, 参数化 tif 路径)"""
    gdf = gpd.read_file(CITY_SHP, encoding="utf-8")
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)

    with rasterio.open(ssi_tif) as src:
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
        mask = geometry_mask([row.geometry], out_shape=(h, w),
                             transform=transform, invert=True, all_touched=True)
        for i, yr in enumerate(years):
            band = cube[2 * i: 2 * i + 2]
            vals = band[:, mask]
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            records.append({"年份": yr, "key": key, "SSI": float(vals.mean())})
    return pd.DataFrame(records)


def analyze_one(tag, label, ssi_tif, yield_df):
    """单数据集全流程: 提取 SSI -> 合并 -> 各市 r/p + 分级减产率 + 全省汇总 -> CSV"""
    ssi_df = load_city_ssi(ssi_tif)
    mg = yield_df.merge(ssi_df[["年份", "key", "SSI"]], on=["年份", "key"], how="left")

    print(f"\n================ {label}: 各市 SSI x 产量距平 ================")
    rows = []
    # 旱涝双端分组: 干旱端 (SSI 负) 与 偏湿端 (SSI 正), 完整刻画水分异常两端对产量的影响
    bins = {"轻度": (-1.0, -0.5), "中度": (-1.5, -1.0), "重度": (-np.inf, -1.5)}
    wet_bins = {"轻度偏湿": (0.5, 1.0), "中度偏湿": (1.0, 1.5), "重度偏湿": (1.5, np.inf)}
    all_bins = {**bins, **wet_bins}
    for key, g in mg.groupby("key"):
        city = g["城市"].iloc[0]
        g = g.dropna(subset=["SSI"])
        n = len(g)
        if n < 3:
            print(f"{city:<6} [缺失] 有效配对样本 {n} 年")
            rows.append({"城市": city, "key": key, "有效年数": n, "Pearson_r": np.nan, "p值": np.nan,
                         "显著性(p<0.05)": "数据不足",
                         **{f"{lvl}平均减产率%": np.nan for lvl in all_bins},
                         **{f"{lvl}样本": 0 for lvl in all_bins}, "备注": "有效配对样本不足"})
            continue
        r, p = stats.pearsonr(g["SSI"], g["距平"])
        grp = {}
        for lvl, (lo, hi) in all_bins.items():
            sub = g[(g["SSI"] >= lo) & (g["SSI"] < hi)]
            grp[lvl] = (sub["减产率%"].mean() if len(sub) else np.nan, len(sub))

        def fmt_group(lv):
            return f"{lv}:{'—' if np.isnan(grp[lv][0]) else format(grp[lv][0], '+.1f')}%({grp[lv][1]})"

        print(f"{city:<6} r={r:+.3f}  p={p:.3f}{'*' if p < 0.05 else ''}  n={n}年")
        print("        旱端 " + "  ".join(fmt_group(lvl) for lvl in bins)
              + "   |   湿端 " + "  ".join(fmt_group(lvl) for lvl in wet_bins))
        rows.append({"城市": city, "key": key, "有效年数": n, "Pearson_r": r, "p值": p,
                     "显著性(p<0.05)": "显著" if p < 0.05 else "不显著",
                     **{f"{lvl}平均减产率%": grp[lvl][0] for lvl in all_bins},
                     **{f"{lvl}样本": grp[lvl][1] for lvl in all_bins},
                     "备注": "" if n == 9 else f"仅{n}年有效配对"})

    # 全省汇总 (全部市-年观测合并)
    g_all = mg.dropna(subset=["SSI"])
    r_all, p_all = stats.pearsonr(g_all["SSI"], g_all["距平"])
    print(f"---- 全省汇总: n={len(g_all)}, r={r_all:+.3f}, p={p_all:.2e} ----")
    mod = g_all[(g_all.SSI >= -1.5) & (g_all.SSI < -1.0)]["减产率%"]
    sev = g_all[g_all.SSI < -1.5]["减产率%"]
    modw = g_all[(g_all.SSI >= 1.0) & (g_all.SSI < 1.5)]["减产率%"]
    sevw = g_all[g_all.SSI >= 1.5]["减产率%"]

    def _pct(s):
        return "无样本" if not len(s) else f"{s.mean():+.1f}% (n={len(s)})"

    print(f"  干旱端: 中度 {_pct(mod)} | 重度 {_pct(sev)}")
    print(f"  偏湿端: 中度 {_pct(modw)} | 重度 {_pct(sevw)}")
    rows.append({"城市": "全省汇总", "key": "", "有效年数": len(g_all),
                 "Pearson_r": r_all, "p值": p_all,
                 "显著性(p<0.05)": "显著" if p_all < 0.05 else "不显著",
                 "轻度平均减产率%": g_all[(g_all.SSI >= -1.0) & (g_all.SSI < -0.5)]["减产率%"].mean(),
                 "轻度样本": int(((g_all.SSI >= -1.0) & (g_all.SSI < -0.5)).sum()),
                 "中度平均减产率%": mod.mean() if len(mod) else np.nan, "中度样本": len(mod),
                 "重度平均减产率%": sev.mean() if len(sev) else np.nan, "重度样本": len(sev),
                 "轻度偏湿平均减产率%": g_all[(g_all.SSI >= 0.5) & (g_all.SSI < 1.0)]["减产率%"].mean(),
                 "轻度偏湿样本": int(((g_all.SSI >= 0.5) & (g_all.SSI < 1.0)).sum()),
                 "中度偏湿平均减产率%": modw.mean() if len(modw) else np.nan, "中度偏湿样本": len(modw),
                 "重度偏湿平均减产率%": sevw.mean() if len(sevw) else np.nan, "重度偏湿样本": len(sevw),
                 "备注": "全部市-年观测合并"})

    out = pd.DataFrame(rows)
    out_path = OUT_DIR / f"city_yield_analysis_{tag}.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"已保存: {out_path}")

    return {"层级": label, "相关系数r": r_all, "p值": p_all,
            "中度干旱平均减产率%": mod.mean() if len(mod) else np.nan,
            "重度干旱平均减产率%": sev.mean() if len(sev) else np.nan,
            "中度偏湿平均减产率%": modw.mean() if len(modw) else np.nan,
            "重度偏湿平均减产率%": sevw.mean() if len(sevw) else np.nan}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    yield_df = load_yield_anomaly()
    summary = []
    for tag, label, ssi_tif in DATASETS:
        if not ssi_tif.exists():
            print(f"[跳过] 找不到输入: {ssi_tif}")
            continue
        summary.append(analyze_one(tag, label, ssi_tif, yield_df))

    # ---- 四层汇总对比表 ----
    comp = pd.DataFrame(summary)
    comp_path = OUT_DIR / "层级产量关联对比表.csv"
    comp.to_csv(comp_path, index=False, encoding="utf-8-sig")
    print("\n" + "=" * 72)
    print("四套 SSI x 冬小麦产量关联: 汇总对比表 (全省汇总, 全部市-年观测)")
    print("=" * 72)
    for _, row in comp.iterrows():
        def _cell(col):
            return "无样本" if pd.isna(row[col]) else f"{row[col]:+.1f}%"
        print(f"  {row['层级']:<18} r={row['相关系数r']:+.3f}  p={row['p值']:.2e}  "
              f"中度干旱: {_cell('中度干旱平均减产率%'):<10} 重度干旱: {_cell('重度干旱平均减产率%'):<10} "
              f"中度偏湿: {_cell('中度偏湿平均减产率%'):<10} 重度偏湿: {_cell('重度偏湿平均减产率%')}")
    print(f"\n对比表已保存: {comp_path}")
    print(f"全部输出目录: {OUT_DIR}")


if __name__ == "__main__":
    main()
