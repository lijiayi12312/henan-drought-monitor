# -*- coding: utf-8 -*-
"""
0-28cm 浅层根区土壤水分 与 浅层 SSI 计算
----------------------------------
1. 读取逐月土壤水分 swvl1 (0-7cm, 全国网格) / swvl2 (7-28cm, 河南网格)
   - 检查时间维度 (120 个月, 201601~202512) 与空间维度; 不一致时将 swvl1 窗口裁剪到河南网格
     (两者同为 0.25 度且像元精确对齐, 无需重采样)
2. 浅层根区土壤水分 = (swvl1 + swvl2) / 2   (两层直接平均, 不加深度权重)
3. 输出逐月浅层根区土壤水分 -> 河南省SSI\\swvl_shallow\\河南浅层SM_YYYYMM.tif
4. 浅层 SSI: 方法与 compute_ssi_layers.py 一致 —— 每像元每日历月取 2016-2025 共 10 年样本,
   SSI = (当月值 - 同期均值) / 同期总体标准差 (ddof=0); 样本不足或标准差为 0 置 NaN
   -> 河南省SSI\\swvl_shallow_ssi\\河南SSI_YYYYMM.tif
输出格式与既有 SSI 一致: float32, nodata=-3.4e38, deflate, EPSG:4326

用法:
    python shallow_zone_ssi.py --limit 3    # 只输出前 3 个月验证 (检查 SSI 值域 [-3, 3])
    python shallow_zone_ssi.py              # 输出全部 120 个月
注意: SSI 标准化始终使用全量 120 个月样本 (计算需要), --limit 只控制写出文件数。
"""
import argparse
import os
import re
import warnings

import numpy as np
import rasterio
from rasterio.windows import from_bounds

# ==================== 配置 ====================
RAW_ROOT = r"D:\大二\练习数据\土壤水分输出数据"   # 原始土壤水分 (swvl1/2/YYYYMM.tif)
OUT_ROOT = r"D:\大二\干旱预测\河南省SSI"          # 输出根目录
OUT_SM_DIR = os.path.join(OUT_ROOT, "swvl_shallow")       # 浅层根区土壤水分
OUT_SSI_DIR = os.path.join(OUT_ROOT, "swvl_shallow_ssi")  # 浅层根区 SSI
NODATA = np.float32(-3.4028234663852886e+38)      # 与既有 SSI 输出一致
# =============================================


def list_monthly_tifs(var_dir):
    """列出 YYYYMM.tif (按年月升序), 返回 [(yyyymm, 路径), ...]。"""
    items = []
    for name in os.listdir(var_dir):
        m = re.fullmatch(r"(\d{6})\.tif", name)
        if m:
            items.append((m.group(1), os.path.join(var_dir, name)))
    return sorted(items)


def load_henan_stack(var_dir, ref_profile=None):
    """读取一个变量全部月份。若提供 ref_profile 且网格不同, 按参考网格窗口裁剪。
    返回 (data (T,H,W) float64, yyyymms, profile)。"""
    items = list_monthly_tifs(var_dir)
    if not items:
        raise FileNotFoundError(f"{var_dir} 下未找到 YYYYMM.tif")
    yyyymms = [ym for ym, _ in items]
    stack, profile = [], None
    for _, path in items:
        with rasterio.open(path) as ds:
            profile = ds.profile.copy()
            if ref_profile is not None and (ds.width, ds.height) != (ref_profile["width"], ref_profile["height"]):
                # 空间对齐: 同为 0.25 度 EPSG:4326, 用参考网格 bounds 开窗读取 (精确对齐, 无重采样)
                rb = ref_profile
                west = rb["transform"].c
                north = rb["transform"].f
                east = west + rb["width"] * 0.25
                south = north - rb["height"] * 0.25
                win = from_bounds(west, south, east, north, transform=ds.transform)
                win = win.round_offsets().round_lengths()
                arr = ds.read(1, window=win).astype("float64")
                if arr.shape != (rb["height"], rb["width"]):
                    raise ValueError(f"{path} 裁剪后 {arr.shape} 与参考网格 "
                                     f"({rb['height']},{rb['width']}) 不符")
            else:
                arr = ds.read(1).astype("float64")
        stack.append(arr)
    return np.stack(stack), yyyymms, profile


def compute_ssi(data, yyyymms):
    """按日历月份逐像元标准化 (与 compute_ssi_layers.py 一致, ddof=0)。"""
    months = np.array([int(ym[4:6]) for ym in yyyymms])
    ssi = np.full_like(data, np.nan)
    for m in range(1, 13):
        idx = np.where(months == m)[0]
        if idx.size < 2:
            continue
        samples = data[idx]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean = np.nanmean(samples, axis=0)
            std = np.nanstd(samples, axis=0, ddof=0)
        n_valid = np.isfinite(samples).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            z = (samples - mean) / std
        z[:, std == 0] = np.nan
        z[:, n_valid < 2] = np.nan
        ssi[idx] = z
    return ssi


def export_tif(arr, path, profile):
    """写出单波段 tif: float32, nodata, deflate。"""
    arr = arr.astype("float32")
    arr = np.where(np.isnan(arr), NODATA, arr)
    p = profile.copy()
    p.update(dtype="float32", nodata=float(NODATA), count=1, compress="deflate")
    with rasterio.open(path, "w", **p) as dst:
        dst.write(arr, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=120, help="只输出前 N 个月 (SSI 计算仍用全量样本)")
    args = ap.parse_args()

    # ---- 读取两层 (swvl2 为参考网格: 河南裁剪版) ----
    print("===== 读取原始土壤水分 =====")
    v2, yms2, prof2 = load_henan_stack(os.path.join(RAW_ROOT, "swvl2"))
    v1, yms1, _ = load_henan_stack(os.path.join(RAW_ROOT, "swvl1"), ref_profile=prof2)
    print(f"  swvl2: {v2.shape[0]} 个月, 网格 {v2.shape[2]}x{v2.shape[1]}")
    print(f"  swvl1: {v1.shape[0]} 个月, 裁剪后网格 {v1.shape[2]}x{v1.shape[1]}")

    # ---- 维度检查 ----
    ok_t = yms1 == yms2
    ok_s = v1.shape == v2.shape
    print(f"  时间维度一致: {ok_t} ({yms1[0]} ~ {yms1[-1]})")
    print(f"  空间维度一致: {ok_s} ({v1.shape[1]} 行 x {v1.shape[2]} 列)")
    if not (ok_t and ok_s):
        raise SystemExit("[错误] 维度对齐失败, 终止")

    # ---- 浅层根区土壤水分 (任一层无效 -> 该像元无效) ----
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        shallow_sm = (v1 + v2) / 2.0
    n_valid = int(np.isfinite(shallow_sm).sum(axis=(1, 2)).mean())
    print(f"\n===== 浅层根区加权 (swvl1 + swvl2) / 2 =====")
    print(f"  平均每期有效像元: {n_valid} / {shallow_sm[0].size}")
    print(f"  全部月份值域: {np.nanmin(shallow_sm):.4f} ~ {np.nanmax(shallow_sm):.4f} "
          f"(合理区间约 0.1~0.5 m3/m3)")

    # ---- 浅层 SSI (全量样本标准化) ----
    shallow_ssi = compute_ssi(shallow_sm, yms1)
    print(f"\n===== 浅层 SSI (逐日历月 10 年标准化) =====")
    print(f"  值域: {np.nanmin(shallow_ssi):.3f} ~ {np.nanmax(shallow_ssi):.3f}")
    in_range = np.nanmin(shallow_ssi) >= -3.0 and np.nanmax(shallow_ssi) <= 3.0
    print(f"  值域在 [-3, 3] 内: {'是' if in_range else '否 <-- 请检查!'}")

    # ---- 输出前 limit 个月 ----
    os.makedirs(OUT_SM_DIR, exist_ok=True)
    os.makedirs(OUT_SSI_DIR, exist_ok=True)
    n = min(args.limit, len(yms1))
    for i in range(n):
        ym = yms1[i]
        export_tif(shallow_sm[i], os.path.join(OUT_SM_DIR, f"河南浅层SM_{ym}.tif"), prof2)
        export_tif(shallow_ssi[i], os.path.join(OUT_SSI_DIR, f"河南SSI_{ym}.tif"), prof2)
    print(f"\n===== 已输出前 {n} 个月 (验证模式: {args.limit < len(yms1)}) =====")
    for tag, d in (("浅层水分", OUT_SM_DIR), ("浅层SSI", OUT_SSI_DIR)):
        fs = sorted(os.listdir(d))
        print(f"  {d}")
        print(f"    文件数: {len(fs)}, 范围: {fs[0]} ~ {fs[-1]}" if fs else "    (空)")
    # 每月摘要, 便于人工抽查
    print("\n---- 前 3 个月逐月摘要 ----")
    for i in range(min(3, len(yms1))):
        sm, ss = shallow_sm[i], shallow_ssi[i]
        print(f"  {yms1[i]}: 水分 {np.nanmin(sm):.4f}~{np.nanmax(sm):.4f} 均值 {np.nanmean(sm):.4f} | "
              f"SSI {np.nanmin(ss):+.2f}~{np.nanmax(ss):+.2f} 均值 {np.nanmean(ss):+.3f}")


if __name__ == "__main__":
    main()
