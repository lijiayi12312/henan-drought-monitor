# -*- coding: utf-8 -*-
"""
生成浅层根区 (0-28cm) 各市逐年 单产距平 x 4-5月SSI 时间序列 CSV
单产口径 (同论文): 实际单产(吨/公顷) = 总产(万吨)/播种面积(千公顷)×10, 分市线性去趋势;
逻辑与 yield_ssi_analysis_layers.py 完全一致 (all_touched 城市 SSI 提取),
输出结构与表层 city_yield_ssi_timeseries.csv 相同, 供平台产量影响评估模块按版本切换读取。
输出: 产量关联分析\\city_yield_ssi_timeseries_shallow.csv
"""
from pathlib import Path
import re

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask

ROOT = Path(__file__).resolve().parent.parent
YIELD_CSV = ROOT / "data" / "henan_wheat_yield_clean.csv"   # 冬小麦总产(万吨), 18市×2016-2024, 勘误版
AREA_XLSX = ROOT / "data" / "河南省各市冬小麦播种面积.xlsx"   # 宽表: 行=各市, 列=2016-2024年, 值=千公顷
CITY_SHP = ROOT / "data" / "河南省市shp" / "河南省.shp"
SSI_TIF = ROOT / "data" / "4-5月河南SSI" / "4-5月河南SSI_shallow_2016-2025.tif"
OUT = ROOT / "output" / "产量关联分析" / "city_yield_ssi_timeseries_shallow.csv"


def norm_city(name):
    n = "".join(str(name).split())   # 去内部空格(如"郑  州  市"->"郑州市")
    for suf in ("示范区", "市"):
        if n.endswith(suf) and len(n) > len(suf):
            return n[: -len(suf)]
    return n


def main():
    # 单产换算 + 去趋势 (同 yield_ssi_analysis.load_yield_anomaly, 单产口径)
    ydf = pd.read_csv(YIELD_CSV)
    ydf = ydf[["city", "key", "year", "yield"]].rename(
        columns={"city": "城市", "year": "年份", "yield": "总产"})
    ydf["年份"] = ydf["年份"].astype(int)
    raw = pd.read_excel(AREA_XLSX, header=None)   # 播种面积宽表 -> 长表 [key, 年份, 面积(千公顷)]
    year_cols = {}
    for j, v in enumerate(raw.iloc[1]):
        m = re.fullmatch(r"(\d{4})年", str(v).strip()) if pd.notna(v) else None
        if m:
            year_cols[int(m.group(1))] = j
    assert sorted(year_cols) == list(range(2016, 2025)), f"面积表年份列异常: {sorted(year_cols)}"
    arecs = []
    for i in range(2, len(raw)):
        name = raw.iloc[i, 0]
        if pd.isna(name):
            continue
        k = norm_city(name)
        if k in ("全省", "各市之和"):
            continue
        for yr, j in year_cols.items():
            arecs.append({"key": k, "年份": yr,
                         "面积": pd.to_numeric(raw.iloc[i, j], errors="coerce")})
    adf = pd.DataFrame(arecs)
    assert adf["key"].nunique() == 18 and len(adf) == 162 \
        and adf["面积"].notna().all() and (adf["面积"] > 0).all(), "播种面积表数据不完整"
    mg = ydf.merge(adf, on=["key", "年份"], how="inner")
    assert len(mg) == 162, f"产量×面积合并后应为162行, 实际{len(mg)}"
    mg["实际单产"] = mg["总产"] / mg["面积"] * 10.0
    yrecs = []
    for key, g in mg.groupby("key"):
        g = g.sort_values("年份")
        if len(g) < 3:
            continue
        y = g["实际单产"].to_numpy(float)
        x = g["年份"].to_numpy(float)
        a, b = np.polyfit(x, y, 1)
        trend = a * x + b
        for yr, yi, ti in zip(x.astype(int), y, trend):
            yrecs.append({"年份": yr, "key": key, "城市": g["城市"].iloc[0],
                          "实际单产": yi, "趋势单产": ti,
                          "单产距平": yi - ti, "减产率%": (yi - ti) / ti * 100.0})
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
    out = out[["年份", "key", "SSI", "城市", "实际单产", "趋势单产", "单产距平", "减产率%"]]
    out = out.sort_values(["key", "年份"]).reset_index(drop=True)
    out.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"已输出: {OUT}  ({len(out)} 行, {out['key'].nunique()} 市)")
    print(f"年份范围: {out['年份'].min()}~{out['年份'].max()}; 含产量年份: 2016-2024")
    print(out.head(3).to_string())


if __name__ == "__main__":
    main()
