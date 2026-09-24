# -*- coding: utf-8 -*-
"""
批量裁剪全国逐月 SSI 栅格到河南省
输入: d:\\大二\\练习数据\\SSI计算结果\\SSI_YYYYMM.tif (201601-202512, 共120个月)
矢量: 河南省 (1).shp (EPSG:4326)
输出: d:\\大二\\干旱预测\\河南省SSI\\河南SSI_YYYYMM.tif
依赖: rioxarray, geopandas, shapely
"""
import re
from pathlib import Path

import geopandas as gpd
import rioxarray as rxr
from shapely.geometry import mapping

# ---------------- 路径配置 ----------------
SSI_DIR = Path(r"d:\大二\练习数据\SSI计算结果")
SHP_PATH = Path(r"d:\大二\干旱预测\河南省shp\河南省 (1).shp")
OUT_DIR = Path(r"d:\大二\干旱预测\河南省SSI")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- 读取河南省矢量 ----------------
henan = gpd.read_file(SHP_PATH)
# 融合为单个多边形，裁剪时作为一个整体
henan = henan[["geometry"]].dissolve()
henan_geoms = henan.geometry.apply(mapping).tolist()

# ---------------- 批量裁剪全部时间层 ----------------
tif_files = sorted(SSI_DIR.glob("SSI_*.tif"))
print(f"共发现 {len(tif_files)} 个 SSI 逐月栅格，开始裁剪 ...")

success, skipped = 0, 0
for i, tif_path in enumerate(tif_files, 1):
    match = re.search(r"SSI_(\d{6})\.tif$", tif_path.name)
    if not match:
        skipped += 1
        continue
    year_month = match.group(1)  # 年份+月份, 如 201601

    rds = rxr.open_rasterio(tif_path)

    # 坐标系不一致时将矢量重投影到栅格坐标系 (本例均为 EPSG:4326)
    if henan.crs != rds.rio.crs:
        geoms = henan.to_crs(rds.rio.crs).geometry.apply(mapping).tolist()
        crs = rds.rio.crs
    else:
        geoms, crs = henan_geoms, henan.crs

    # 按河南省边界裁剪, drop=True 仅保留与河南相交的窗口
    clipped = rds.rio.clip(geoms, crs=crs, drop=True, all_touched=False)

    out_path = OUT_DIR / f"河南SSI_{year_month}.tif"
    clipped.rio.to_raster(
        out_path,
        dtype="float32",
        nodata=rds.rio.nodata,
        compress="deflate",
    )

    rds.close()
    clipped.close()
    success += 1
    print(f"[{i}/{len(tif_files)}] {out_path.name} 已输出")

print(f"\n完成! 成功 {success} 个, 跳过 {skipped} 个")
print(f"输出目录: {OUT_DIR}")
