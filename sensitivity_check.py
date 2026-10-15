# -*- coding: utf-8 -*-
r"""敏感性检验: 极端年份对 SSI x 产量距平 相关性的贡献
对比 全样本 / 剔除 2023 / 剔除 2018+2023 三种情形的 Pearson r 与 Spearman rho,
用于判断 "2023 烂场雨是否主导整体相关性" (链路与 yield_ssi_analysis_layers.py 一致)
运行:  设 PYTHONPATH=C:\pylibs 后 python sensitivity_check.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
from scipy import stats

ROOT = Path(r"D:\大二\干旱预测")
YIELD_XLSX = Path(r"d:\桌面\henan_wheat_yield.xlsx")
CITY_SHP = ROOT / "河南省市shp" / "河南省.shp"
SSI_TIF = ROOT / "4-5月河南SSI" / "4-5月河南SSI_shallow_2016-2025.tif"


def norm_city(name):
    n = str(name).strip()
    for suf in ("示范区", "市"):
        if n.endswith(suf) and len(n) > len(suf):
            return n[: -len(suf)]
    return n


def main():
    gdf = gpd.read_file(CITY_SHP, encoding="utf-8")
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)

    df = pd.read_excel(YIELD_XLSX)
    df.columns = ["年份", "城市", "产量"]
    df["年份"] = df["年份"].astype(int)
    df["key"] = df["城市"].map(norm_city)
    records = []
    for key, g in df.groupby("key"):
        g = g.sort_values("年份")
        y = g["产量"].to_numpy(float)
        x = g["年份"].to_numpy(float)
        a, b = np.polyfit(x, y, 1)
        for yr, yi in zip(x.astype(int), y):
            records.append({"年份": yr, "key": key, "距平": yi - (a * yr + b)})
    yield_df = pd.DataFrame(records)

    with rasterio.open(SSI_TIF) as src:
        cube = src.read().astype(np.float64)
        nodata = src.nodata
        transform, h, w = src.transform, src.height, src.width
    if nodata is not None:
        cube[cube <= nodata] = np.nan
    ssi_recs = []
    for _, rrow in gdf.iterrows():
        name = str(rrow["name"])
        key = name[:-1] if name.endswith("市") else name
        m = geometry_mask([rrow.geometry], out_shape=(h, w), transform=transform,
                          invert=True, all_touched=True)
        for i, yr in enumerate(range(2016, 2026)):
            b = cube[2 * i: 2 * i + 2][:, m]
            b = b[np.isfinite(b)]
            if b.size == 0:
                continue
            ssi_recs.append({"年份": yr, "key": key, "SSI": float(b.mean())})
    mg = yield_df.merge(pd.DataFrame(ssi_recs)[["年份", "key", "SSI"]],
                        on=["年份", "key"], how="left").dropna(subset=["SSI"])

    cases = [
        ("全样本", mg),
        ("剔除 2023", mg[mg["年份"] != 2023]),
        ("剔除 2018+2023", mg[~mg["年份"].isin([2018, 2023])]),
    ]
    print(f"{'情形':<16}{'n':>5}{'Pearson r':>12}{'p':>12}{'Spearman rho':>14}{'p':>12}")
    for label, d in cases:
        r, p = stats.pearsonr(d["SSI"], d["距平"])
        rho, ps = stats.spearmanr(d["SSI"], d["距平"])
        print(f"{label:<16}{len(d):>5}{r:>12.4f}{p:>12.2e}{rho:>14.4f}{ps:>12.2e}")
    print("\n判读: 若剔除极端年后 r 大幅减弱且不再显著 -> 极端年主导;")
    print("      若仍显著为负 -> 负相关为整体趋势, 极端年仅增强其强度。")


if __name__ == "__main__":
    main()
