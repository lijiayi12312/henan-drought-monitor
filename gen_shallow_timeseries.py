# -*- coding: utf-8 -*-
"""
生成浅层根区 (0-28cm) 各市逐年 产量距平 x 4-5月SSI 时间序列 CSV
逻辑与 yield_ssi_analysis_layers.py 完全一致 (产量线性去趋势 + all_touched 城市 SSI 提取),
输出结构与表层 city_yield_ssi_timeseries.csv 相同, 供平台产量影响评估模块按版本切换读取。
输出: 产量关联分析\\city_yield_ssi_timeseries_shallow.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask

ROOT = Path(r"D:\大二\干旱预测")
YIELD_XLSX = Path(r"d:\桌面\henan_wheat_yield.xlsx")
CITY_SHP = ROOT / "河南省市shp" / "河南省.shp"
SSI_TIF = ROOT / "4-5月河南SSI" / "4-5月河南SSI_shallow_2016-2025.tif"
OUT = ROOT / "产量关联分析" / "city_yield_ssi_timeseries_shallow.csv"


def norm_city(name):
    n = str(name).strip()
    for suf in ("示范区", "市"):
        if n.endswith(suf) and len(n) > len(suf):
            return n[: -len(suf)]
    return n


def main():
    # 产量去趋势 (同 yield_ssi_analysis_layers.load_yield_anomaly)
    df = pd.read_excel(YIELD_XLSX)
    df.columns = ["年份", "城市", "产量"]
    df["年份"] = df["年份"].astype(int)
    df["key"] = df["城市"].map(norm_city)
    yrecs = []
    for key, g in df.groupby("key"):
        g = g.sort_values("年份")
        if len(g) < 3:
            continue
        y = g["产量"].to_numpy(float)
        x = g["年份"].to_numpy(float)
        a, b = np.polyfit(x, y, 1)
        trend = a * x + b
        for yr, yi, ti in zip(x.astype(int), y, trend):
            yrecs.append({"年份": yr, "key": key, "城市": g["城市"].iloc[0],
                          "实际产量": yi, "趋势产量": ti,
                          "距平": yi - ti, "减产率%": (yi - ti) / ti * 100.0})
    yield_df = pd.DataFrame(yrecs)

    # 城市 SSI 提取 (同 yield_ssi_analysis_layers.load_city_ssi, 2025 无产量仅 SSI)
    gdf = gpd.read_file(CITY_SHP, encoding="utf-8")
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    with rasterio.open(SSI_TIF) as src:
        cube = src.read().astype(np.float64)
        nodata = src.nodata
        transform, h, w = src.transform, src.height, src.width
    if nodata is not None:
        cube[cube <= nodata] = np.nan
    srecs = []
    for _, row in gdf.iterrows():
        name = str(row["name"])
        key = name[:-1] if name.endswith("市") else name
        mask = geometry_mask([row.geometry], out_shape=(h, w),
                             transform=transform, invert=True, all_touched=True)
        for i, yr in enumerate(range(2016, 2026)):
            vals = cube[2 * i: 2 * i + 2][:, mask]
            vals = vals[np.isfinite(vals)]
            if vals.size:
                srecs.append({"年份": yr, "key": key, "SSI": float(vals.mean())})
    ssi_df = pd.DataFrame(srecs)

    # 合并: 左连 SSI (保留 2025 的 SSI 行, 产量列为 NaN), 列序与表层时序一致
    out = ssi_df.merge(yield_df, on=["年份", "key"], how="left")
    out = out[["年份", "key", "SSI", "城市", "实际产量", "趋势产量", "距平", "减产率%"]]
    out = out.sort_values(["key", "年份"]).reset_index(drop=True)
    out.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"已输出: {OUT}  ({len(out)} 行, {out['key'].nunique()} 市)")
    print(f"年份范围: {out['年份'].min()}~{out['年份'].max()}; 含产量年份: 2016-2024")
    print(out.head(3).to_string())


if __name__ == "__main__":
    main()
