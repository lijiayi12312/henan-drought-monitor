# -*- coding: utf-8 -*-
"""
逐月计算不同等级干旱的加权重心坐标（SSI 栅格）。

干旱等级标准：
    SSI > 0           不干旱
    -1.0 < SSI <= 0    轻度干旱
    -1.5 < SSI <= -1.0 中度干旱
    -2.0 < SSI <= -1.5 重度干旱
    SSI <= -2.0        极端干旱

输出 4 类干旱重心：All_Drought / Moderate_Plus / Severe_Plus / Extreme
"""

import os
import re
import glob
import warnings

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.crs import CRS
from rasterio.transform import xy

# ==================== 用户配置区 ====================
RASTER_FOLDER = os.path.dirname(os.path.abspath(__file__))                       # SSI 栅格所在目录 (脚本同目录)
OUTPUT_CSV = os.path.join(RASTER_FOLDER, "drought_centroid_by_grade.csv")

TARGET_CRS = "EPSG:4326"      # 目标坐标系：WGS 1984 地理坐标系
NODATA_VALUES = [-9999.0]     # 额外需要视为无效值的哨兵值
# 默认模式匹配 SSI_YYYYMM.tif
FILE_PATTERN = "SSI_*.tif"
# ===================================================


# 干旱等级定义：(名称, SSI 上界, SSI 下界是否包含)
GRADE_DEFS = [
    ("All_Drought",   None,   0.0),    # SSI <= 0
    ("Moderate_Plus", None,  -1.0),    # SSI <= -1.0
    ("Severe_Plus",   None,  -1.5),    # SSI <= -1.5
    ("Extreme",       None,  -2.0),    # SSI <= -2.0
]

# SSI 理论取值范围很小，低于该阈值的值一定是 NoData 填充值（如 -9999 或 float32 最小值）
SENTINEL_LOWER_BOUND = -100.0


def read_raster(path, target_crs=TARGET_CRS):
    """读取单波段栅格，统一转为 np.nan 表示无效值；必要时重投影到 target_crs。

    返回 (data, transform, crs)，data 为 float32 二维数组。
    """
    with rasterio.open(path) as src:
        data = src.read(1).astype("float32")
        transform = src.transform
        crs = src.crs
        nodata = src.nodata

        # 统一无效值 -> np.nan（NaN、nodata、明确列出的哨兵值、异常小的填充值）
        invalid = ~np.isfinite(data)
        if nodata is not None:
            invalid |= (data == np.float32(nodata))
        for nv in NODATA_VALUES:
            invalid |= (data == np.float32(nv))
        invalid |= (data < SENTINEL_LOWER_BOUND)
        if invalid.any():
            data[invalid] = np.nan

        if crs is None:
            warnings.warn(f"{os.path.basename(path)} 缺少坐标系信息，跳过重投影")
            return data, transform, crs

        # 已经是目标坐标系则跳过重投影
        if crs.to_epsg() != CRS.from_user_input(target_crs).to_epsg():
            dst_crs = CRS.from_user_input(target_crs)
            dst_transform, dst_width, dst_height = calculate_default_transform(
                crs, dst_crs, src.width, src.height, *src.bounds
            )
            dst = np.full((dst_height, dst_width), np.nan, dtype="float32")
            reproject(
                source=data,
                destination=dst,
                src_transform=transform,
                src_crs=crs,
                src_nodata=np.nan,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )
            data, transform, crs = dst, dst_transform, dst_crs

    return data, transform, crs


def compute_centroid(data, transform, upper_bound):
    """计算 SSI <= upper_bound 像元的加权重心（权重 = |SSI|）。

    返回 (X, Y, count, sum_weight)；若无有效像元则坐标与权重为 NaN、count 为 0。
    """
    valid = np.isfinite(data)
    mask = valid & (data <= upper_bound)
    if not mask.any():
        return np.nan, np.nan, 0, np.nan

    # 只取满足条件的像元，避免对大数组做冗余运算
    ssi = data[mask]
    rows, cols = np.nonzero(mask)
    xs, ys = xy(transform, rows, cols, offset="center")
    xs = np.asarray(xs, dtype="float64")
    ys = np.asarray(ys, dtype="float64")

    weight = np.abs(ssi).astype("float64")
    sum_weight = float(weight.sum())
    if sum_weight == 0.0:  # 例如全部像元 SSI 恰好为 0
        return np.nan, np.nan, int(mask.sum()), 0.0

    X = float((xs * weight).sum() / sum_weight)
    Y = float((ys * weight).sum() / sum_weight)
    return X, Y, int(mask.sum()), sum_weight


def haversine(lon1, lat1, lon2, lat2):
    """基于经纬度计算两点间球面距离（公里）。任一参数为 NaN 时返回 NaN。"""
    if any(pd.isna(v) for v in (lon1, lat1, lon2, lat2)):
        return np.nan
    radius = 6371.0088  # 地球平均半径 km
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return float(2.0 * radius * np.arcsin(np.sqrt(a)))


def list_raster_files(folder):
    """按 YYYYMM 升序返回 (year_month, 文件路径) 列表。"""
    items = []
    for path in glob.glob(os.path.join(folder, FILE_PATTERN)):
        m = re.search(r"(\d{6})", os.path.basename(path))
        if m:
            items.append((m.group(1), path))
    items.sort(key=lambda t: t[0])
    return items


def main():
    files = list_raster_files(RASTER_FOLDER)
    if not files:
        print(f"未在 {RASTER_FOLDER} 中找到匹配 {FILE_PATTERN} 的栅格文件")
        return

    records = []
    total = len(files)

    for i, (year_month, path) in enumerate(files, start=1):
        try:
            data, transform, crs = read_raster(path)
        except Exception as exc:  # 单个文件失败不中断整体流程
            print(f"[警告] 读取失败，已跳过 {os.path.basename(path)}：{exc}")
            continue

        # 校验像元是否为经纬度单位（EPSG:4326）
        if crs is not None and crs.to_epsg() != 4326:
            print(f"[警告] {os.path.basename(path)} 重投影后坐标系为 {crs}，结果可能不是经纬度")

        for grade, _, upper_bound in GRADE_DEFS:
            X, Y, count, sum_weight = compute_centroid(data, transform, upper_bound)
            records.append({
                "year_month": year_month,
                "grade": grade,
                "X": X,
                "Y": Y,
                "count": count,
                "sum_weight": sum_weight,
            })

        print(f"已完成 {year_month}，共 {total} 个月")

    df = pd.DataFrame.from_records(
        records,
        columns=["year_month", "grade", "X", "Y", "count", "sum_weight"],
    )

    # 按 grade 分组，计算相邻月份重心迁移距离
    df = df.sort_values(["grade", "year_month"]).reset_index(drop=True)
    df["migration_distance_km"] = np.nan
    for grade, idx in df.groupby("grade", sort=False).groups.items():
        idx = list(idx)
        for prev, cur in zip(idx[:-1], idx[1:]):
            df.at[cur, "migration_distance_km"] = haversine(
                df.at[prev, "X"], df.at[prev, "Y"],
                df.at[cur, "X"], df.at[cur, "Y"],
            )

    # 按 year_month、grade 升序排列后输出
    df = df.sort_values(["year_month", "grade"]).reset_index(drop=True)
    df["X"] = df["X"].round(6)
    df["Y"] = df["Y"].round(6)

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"完成，共处理 {total} 个月，输出：{OUTPUT_CSV}")


if __name__ == "__main__":
    main()