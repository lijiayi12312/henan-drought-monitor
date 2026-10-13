# -*- coding: utf-8 -*-
"""
对 swvl2 / swvl3 / swvl4 逐月土壤水分计算 SSI（标准化土壤水分指数）。

方法与原 swvl1 完全一致（经数据反推验证）：
    对每个像元、每个日历月份（1-12 月），取 2016-2025 该月共 10 年的值，
    计算同期均值和标准差（总体标准差 ddof=0），
    SSI = (当月值 - 同期均值) / 同期标准差；标准差为 0 或样本不足时置 NaN。

输入：
    D:\\大二\\练习数据\\土壤水分输出数据\\{swvl2,swvl3,swvl4}\\YYYYMM.tif（河南网格，120 个月）
输出（格式参照 河南省SSI\\swvl1）：
    D:\\大二\\干旱预测\\河南省SSI\\{swvl2,swvl3,swvl4}\\河南SSI_YYYYMM.tif
    float32、nodata=-3.4e38、deflate 压缩、EPSG:4326
"""

import os
import re
import warnings

import numpy as np
import rasterio

# ==================== 用户配置区 ====================
INPUT_ROOT = r"D:\大二\练习数据\土壤水分输出数据"   # 输入根目录（swvl2/swvl3/swvl4 子文件夹）
OUTPUT_ROOT = r"D:\大二\干旱预测\河南省SSI"          # 输出根目录（生成 swvl2/swvl3/swvl4 子文件夹）
VARS = ["swvl2", "swvl3", "swvl4"]
NODATA = np.float32(-3.4028234663852886e+38)          # 与 swvl1 SSI 输出一致
# ===================================================


def list_monthly_tifs(var_dir):
    """列出某变量的 120 个 YYYYMM.tif，按年月升序，返回 [(yyyymm, 路径), ...]。"""
    items = []
    for name in os.listdir(var_dir):
        m = re.fullmatch(r"(\d{6})\.tif", name)
        if m:
            items.append((m.group(1), os.path.join(var_dir, name)))
    return sorted(items)


def load_stack(var_dir):
    """读取全部月份数据，返回 (data (T,H,W) float64, yyyymms, profile)。"""
    items = list_monthly_tifs(var_dir)
    if not items:
        raise FileNotFoundError(f"{var_dir} 下未找到 YYYYMM.tif 文件")
    yyyymms = [ym for ym, _ in items]
    profile = None
    stack = []
    for ym, path in items:
        with rasterio.open(path) as ds:
            arr = ds.read(1).astype("float64")
            profile = ds.profile.copy()
        stack.append(arr)
    return np.stack(stack), yyyymms, profile


def compute_ssi(data, yyyymms):
    """按日历月份逐像元标准化，返回 SSI 数组 (T,H,W)，无效处置 NaN。

    与原方法一致：每个日历月用 10 年样本的均值和总体标准差（ddof=0）。
    """
    months = np.array([int(ym[4:6]) for ym in yyyymms])
    ssi = np.full_like(data, np.nan)

    for m in range(1, 13):
        idx = np.where(months == m)[0]
        if idx.size < 2:
            continue
        samples = data[idx]                        # (10, H, W)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # 全 NaN 像元的空切片警告，无害
            mean = np.nanmean(samples, axis=0)
            std = np.nanstd(samples, axis=0, ddof=0)
        n_valid = np.isfinite(samples).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            z = (samples - mean) / std
        z[:, std == 0] = np.nan                   # 标准差为 0 无法标准化
        z[:, n_valid < 2] = np.nan                # 有效样本不足
        ssi[idx] = z
    return ssi


def export_tifs(ssi, yyyymms, profile, out_dir):
    """逐月写出 tif：float32、nodata=-3.4e38、deflate 压缩。"""
    os.makedirs(out_dir, exist_ok=True)
    out_profile = profile.copy()
    out_profile.update(
        dtype="float32",
        nodata=float(NODATA),
        count=1,
        compress="deflate",
    )
    n = 0
    for i, ym in enumerate(yyyymms):
        arr = ssi[i].astype(np.float32)
        arr = np.where(np.isnan(arr), NODATA, arr)  # NaN → nodata，与原输出一致
        out_path = os.path.join(out_dir, f"河南SSI_{ym}.tif")
        with rasterio.open(out_path, "w", **out_profile) as dst:
            dst.write(arr, 1)
        n += 1
    return n


def main():
    total = 0
    for var in VARS:
        var_dir = os.path.join(INPUT_ROOT, var)
        out_dir = os.path.join(OUTPUT_ROOT, var)
        print(f"===== 处理 {var} =====")
        try:
            data, yyyymms, profile = load_stack(var_dir)
            print(f"  读入 {len(yyyymms)} 个月（{yyyymms[0]} ~ {yyyymms[-1]}），网格 "
                  f"{data.shape[2]}x{data.shape[1]}")
            ssi = compute_ssi(data, yyyymms)
            n = export_tifs(ssi, yyyymms, profile, out_dir)
            total += n
            print(f"  已完成 {var}，输出 {n} 个 tif")
        except Exception as exc:
            print(f"  [警告] {var} 处理失败，跳过：{exc}")
    print(f"\n全部完成，共输出 {total} 个 SSI tif，目录：{OUTPUT_ROOT}")


if __name__ == "__main__":
    main()