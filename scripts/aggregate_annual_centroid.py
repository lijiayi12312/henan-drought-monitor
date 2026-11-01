# -*- coding: utf-8 -*-
"""
将逐月干旱重心数据聚合为逐年数据，用于绘制年度干旱重心迁移轨迹。

输入：drought_centroid_by_grade.csv（逐月，120 个月 × 4 等级）
输出：annual_drought_centroid.csv（逐年，10 年 × 4 等级）
"""

import numpy as np
import pandas as pd

# ==================== 用户配置区 ====================
import os as _os
_DIR = _os.path.dirname(_os.path.abspath(__file__))                              # 脚本所在目录
INPUT_CSV = _os.path.join(_DIR, "drought_centroid_by_grade.csv")
OUTPUT_CSV = _os.path.join(_DIR, "annual_drought_centroid.csv")
# ===================================================


def load_monthly(path):
    """读取逐月重心 CSV，提取年份，并返回按 grade、year 排序的数据框。"""
    df = pd.read_csv(path, dtype={"year_month": str})
    df["year"] = df["year_month"].str[:4].astype(int)
    return df.sort_values(["grade", "year", "year_month"]).reset_index(drop=True)


def aggregate_annual(df):
    """按 year + grade 聚合为年度加权重心。

    X / Y 以 sum_weight 为权重加权平均；count 与 sum_weight 直接累加。
    """
    work = df.copy()
    work["X_w"] = work["X"] * work["sum_weight"]
    work["Y_w"] = work["Y"] * work["sum_weight"]

    grouped = work.groupby(["grade", "year"], sort=True, dropna=False)
    annual = grouped.agg(
        X_w_sum=("X_w", "sum"),
        Y_w_sum=("Y_w", "sum"),
        count=("count", "sum"),
        sum_weight=("sum_weight", "sum"),
    ).reset_index()

    # 权重为 0 或缺失的组无法计算加权重心，记为 NaN 并提示
    bad = annual["sum_weight"].isna() | (annual["sum_weight"] <= 0)
    for _, row in annual[bad].iterrows():
        print(f"[警告] {row['year']} 年 {row['grade']} 的总权重为 0 或缺失，重心记为 NaN，已跳过该组计算")

    weight = annual["sum_weight"].where(~bad)
    annual["X"] = annual["X_w_sum"] / weight
    annual["Y"] = annual["Y_w_sum"] / weight
    annual = annual.drop(columns=["X_w_sum", "Y_w_sum"])
    return annual


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


def add_migration_distance(annual):
    """按 grade 分组，计算相邻年份重心之间的迁移距离，首个年份记为 NaN。"""
    annual = annual.sort_values(["grade", "year"]).reset_index(drop=True)
    annual["migration_distance_km"] = np.nan
    for _, idx in annual.groupby("grade", sort=False).groups.items():
        idx = list(idx)
        for prev, cur in zip(idx[:-1], idx[1:]):
            annual.at[cur, "migration_distance_km"] = haversine(
                annual.at[prev, "X"], annual.at[prev, "Y"],
                annual.at[cur, "X"], annual.at[cur, "Y"],
            )
    return annual


def main():
    df = load_monthly(INPUT_CSV)
    annual = aggregate_annual(df)
    annual = add_migration_distance(annual)

    # 输出字段顺序、排序与精度
    annual = annual[["year", "grade", "X", "Y", "count", "sum_weight", "migration_distance_km"]]
    annual = annual.sort_values(["grade", "year"]).reset_index(drop=True)
    annual["X"] = annual["X"].round(6)
    annual["Y"] = annual["Y"].round(6)
    annual["migration_distance_km"] = annual["migration_distance_km"].round(2)
    annual["year"] = annual["year"].astype(int)

    annual.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"完成，输出：{OUTPUT_CSV}（共 {len(annual)} 行）\n")
    print("前 10 行预览：")
    print(annual.head(10).to_string(index=False))


if __name__ == "__main__":
    main()