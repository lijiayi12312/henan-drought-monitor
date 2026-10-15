# -*- coding: utf-8 -*-
"""
SSI x 产量距平 数据链路诊断 (只打印中间量, 不做相关计算)
复现 yield_ssi_analysis_layers.py 的完整处理链路, 逐项排查:
  步骤1  读取 河南地级市边界 shp + 2016-2024 产量 Excel, 检查城市名匹配
  步骤2  读取 浅层根区 20 波段 SSI tif, 健康检查 (波段顺序/值域/有效像元)
  步骤3  商丘市 2023 关键样本: ① SSI 均值 ② 实际产量/趋势产量/距平
  步骤4  2016-2024 逐年: 全省平均 SSI 与 全省平均产量距平
运行:  $env:PYTHONPATH="C:\pylibs"; python diagnose_ssi_yield.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask

ROOT = Path(r"D:\大二\干旱预测")
YIELD_XLSX = Path(r"d:\桌面\henan_wheat_yield.xlsx")   # 年份/城市/夏粮产量(万吨), 2016-2024
CITY_SHP = ROOT / "河南省市shp" / "河南省.shp"
SSI_TIF = ROOT / "4-5月河南SSI" / "4-5月河南SSI_shallow_2016-2025.tif"  # 浅层根区 20 波段


def norm_city(name):
    """城市名规整 (与原链路一致): 去 '示范区'/'市' 后缀"""
    n = str(name).strip()
    for suf in ("示范区", "市"):
        if n.endswith(suf) and len(n) > len(suf):
            return n[: -len(suf)]
    return n


def city_mask(gdf_row, transform, h, w):
    """市界掩膜 (与原链路一致: all_touched=True, invert=True)"""
    return geometry_mask([gdf_row.geometry], out_shape=(h, w),
                         transform=transform, invert=True, all_touched=True)


def main():
    # ================= 步骤1: shp + 产量 Excel =================
    print("=" * 72)
    print("步骤1  读取市界 shp 与产量 Excel")
    print("=" * 72)
    gdf = gpd.read_file(CITY_SHP, encoding="utf-8")
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    print(f"shp: {CITY_SHP}")
    print(f"  地级市 {len(gdf)} 个, CRS = {gdf.crs}")
    print(f"  城市名: {'、'.join(str(n) for n in gdf['name'])}")

    raw = pd.read_excel(YIELD_XLSX)
    print(f"\nExcel: {YIELD_XLSX}")
    print(f"  原始列名: {list(raw.columns)}, 行数: {len(raw)}")
    print(raw.head(5).to_string(index=False))

    df = raw.copy()
    df.columns = ["年份", "城市", "产量"]                    # 与原链路一致: 取前三列
    df["年份"] = df["年份"].astype(int)
    df["key"] = df["城市"].map(norm_city)
    yrs = sorted(df["年份"].unique())
    cnt = df.groupby("key")["年份"].count()
    print(f"\n  年份覆盖: {yrs[0]} ~ {yrs[-1]} ({len(yrs)} 年), 城市 {df['key'].nunique()} 个")
    print(f"  各市样本年数: min={cnt.min()}, max={cnt.max()}"
          + ("" if cnt.nunique() == 1 else f", 不一致! 分布:\n{cnt.to_string()}"))

    shp_keys = {str(n)[:-1] if str(n).endswith("市") else str(n) for n in gdf["name"]}
    xls_keys = set(df["key"])
    only_shp, only_xls = shp_keys - xls_keys, xls_keys - shp_keys
    print(f"  名称匹配: shp 侧 {len(shp_keys)} 个, Excel 侧 {len(xls_keys)} 个, "
          + ("全部匹配" if not (only_shp or only_xls)
             else f"未匹配! 仅shp: {sorted(only_shp)}  仅Excel: {sorted(only_xls)}"))

    # 产量距平 (与原链路一致: 逐市对年份一元线性回归去趋势)
    records = []
    for key, g in df.groupby("key"):
        g = g.sort_values("年份")
        y = g["产量"].to_numpy(float)
        x = g["年份"].to_numpy(float)
        a, b = np.polyfit(x, y, 1)
        trend = a * x + b
        for yr, yi, ti in zip(x.astype(int), y, trend):
            records.append({"年份": yr, "key": key, "城市": g["城市"].iloc[0],
                            "实际产量": yi, "趋势产量": ti,
                            "距平": yi - ti, "减产率%": (yi - ti) / ti * 100.0})
    yield_df = pd.DataFrame(records)
    print(f"\n  产量距平计算完成 (逐市线性去趋势): {len(yield_df)} 条 市-年 记录")

    # ================= 步骤2: SSI 栅格健康检查 =================
    print("\n" + "=" * 72)
    print("步骤2  浅层根区 SSI 栅格健康检查")
    print("=" * 72)
    print(f"文件: {SSI_TIF}")
    with rasterio.open(SSI_TIF) as src:
        cube = src.read().astype(np.float64)                # (20, rows, cols)
        nodata = src.nodata
        transform, h, w = src.transform, src.height, src.width
        crs = src.crs
        descs = list(src.descriptions)
    if nodata is not None:
        cube[cube <= nodata] = np.nan
    print(f"  形状: {cube.shape} (应为 (20, {h}, {w})), CRS = {crs}, "
          f"分辨率 {abs(transform.a):.3f} 度, nodata = {nodata}")
    print(f"  波段顺序: {', '.join(f'{i + 1}:{d}' for i, d in enumerate(descs))}")
    print(f"  全部 20 波段值域: {np.nanmin(cube):+.3f} ~ {np.nanmax(cube):+.3f} "
          f"(SSI 合理区间约 [-3, 3])")
    print(f"  有效像元占比: {np.isfinite(cube).mean() * 100:.1f}%")
    print("  各波段摘要 (均值 / 值域 / 有效像元数):")
    for i in range(cube.shape[0]):
        b = cube[i]
        n_ok = int(np.isfinite(b).sum())
        print(f"    {descs[i]}: 均值 {np.nanmean(b):+.3f}  "
              f"值域 {np.nanmin(b):+.2f} ~ {np.nanmax(b):+.2f}  像元 {n_ok}")

    # ================= 步骤3: 商丘市 2023 关键样本 =================
    print("\n" + "=" * 72)
    print("步骤3  商丘市 2023 年 关键样本诊断")
    print("=" * 72)
    hit = gdf[gdf["name"].astype(str).str.contains("商丘")]
    if hit.empty:
        print("  [错误] shp 中找不到商丘市, 终止")
        return
    brow = hit.iloc[0]
    mask = city_mask(brow, transform, h, w)
    i23 = 2023 - 2016                                        # 波段 2i=4月, 2i+1=5月
    v_all = cube[2 * i23: 2 * i23 + 2][:, mask]
    v_all = v_all[np.isfinite(v_all)]
    v_apr = cube[2 * i23][mask]
    v_apr = v_apr[np.isfinite(v_apr)]
    v_may = cube[2 * i23 + 1][mask]
    v_may = v_may[np.isfinite(v_may)]
    print(f"  掩膜内像元 {int(mask.sum())} 个 (all_touched=True); "
          f"有效值: 4-5 月合并 {v_all.size} 个 (4 月 {v_apr.size}, 5 月 {v_may.size})")
    print(f"  ① 商丘 2023 年 4-5 月 SSI 均值 = {v_all.mean():+.4f}")
    print(f"     分月: 4 月 = {v_apr.mean():+.4f}   5 月 = {v_may.mean():+.4f}")

    sq = yield_df[yield_df["key"] == "商丘"].sort_values("年份")
    print("\n  ② 商丘市 2016-2024 产量序列 (趋势由该市 9 年数据线性拟合):")
    print(sq[["年份", "实际产量", "趋势产量", "距平", "减产率%"]].round(2).to_string(index=False))
    row23 = sq[sq["年份"] == 2023].iloc[0]
    print(f"\n  ② 商丘 2023: 实际产量 = {row23['实际产量']:.2f} 万吨, "
          f"趋势产量 = {row23['趋势产量']:.2f} 万吨, "
          f"距平 = {row23['距平']:+.2f} 万吨 (减产率 {row23['减产率%']:+.2f}%)")

    print("\n  [辅助] 商丘市逐年 4-5 月 SSI 均值 (2016-2025, 同法提取):")
    for i, yr in enumerate(range(2016, 2026)):
        b = cube[2 * i: 2 * i + 2][:, mask]
        b = b[np.isfinite(b)]
        s = f"{b.mean():+.3f}" if b.size else "nan (无有效像元)"
        print(f"    {yr}: {s}" + ("   <-- 2023" if yr == 2023 else ""))

    # ================= 步骤4: 逐年全省平均 =================
    print("\n" + "=" * 72)
    print("步骤4  2016-2024 逐年: 全省平均 SSI 与 全省平均产量距平")
    print("=" * 72)
    ssi_recs = []
    for _, rrow in gdf.iterrows():
        name = str(rrow["name"])
        key = name[:-1] if name.endswith("市") else name      # 与原链路一致
        m = city_mask(rrow, transform, h, w)
        for i, yr in enumerate(range(2016, 2026)):
            b = cube[2 * i: 2 * i + 2][:, m]
            b = b[np.isfinite(b)]
            if b.size == 0:
                continue
            ssi_recs.append({"年份": yr, "key": key, "SSI": float(b.mean())})
    ssi_df = pd.DataFrame(ssi_recs)
    mg = yield_df.merge(ssi_df[["年份", "key", "SSI"]], on=["年份", "key"], how="left")
    n_miss = int(mg["SSI"].isna().sum())
    if n_miss:
        print(f"  [警告] {n_miss} 条 市-年 记录未匹配到 SSI (将不计入下表)")
    ann = (mg.dropna(subset=["SSI"]).groupby("年份")
             .agg(全省平均SSI=("SSI", "mean"),
                  全省平均产量距平=("距平", "mean"),
                  全省平均减产率=("减产率%", "mean"),
                  配对市数=("key", "count"))
             .reset_index()
             .round({"全省平均SSI": 3, "全省平均产量距平": 2, "全省平均减产率": 2}))
    print(ann.to_string(index=False))
    print("\n符号约定: SSI 负 = 偏干, 正 = 偏湿; 距平负 = 减产(低于趋势), 正 = 增产。")
    print("诊断结束 —— 按要求未做相关系数等后续计算。")


if __name__ == "__main__":
    main()
