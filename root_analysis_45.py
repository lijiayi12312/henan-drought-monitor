# -*- coding: utf-8 -*-
"""
对 4-5 月河南 SSI 各层土壤水分数据 (swvl2 / swvl3, 可扩展) 重跑表层 SSI 的全部五项分析
(逻辑与原脚本逐行一致, 仅改输入输出路径; 输出文件名前缀见 DATASETS 配置)

输入: D:\\大二\\干旱预测\\4-5月河南SSI\\4-5月河南SSI_{swvl2,swvl3}_2016-2025.tif (各 20 波段)
输出 (与表层 SSI 分析同目录, 文件名以 河南根区SSI 标记):
  1. 干旱频率  -> 干旱频率\\河南根区SSI_4-5月_干旱频率_2016-2025.tif       (原: drought_frequency_45.py)
  2. MK 检验   -> Mann-Kendall 趋势检验\\河南根区SSI_4-5月_MK_{Z值,p值,Sen斜率}_2016-2025.tif (原: mk_trend_45.py)
  3. Hurst     -> hurst指数\\河南根区SSI_4-5月_Hurst指数_2016-2025.tif      (原: hurst_45.py)
  4. 游程事件  -> 游程理论\\河南根区SSI_4-5月_干旱事件_按真实时间_2016-2025.csv (原: run_theory_45.py)
  5. 高风险区  -> 未来高风险区\\河南根区SSI_4-5月_未来高风险_2016-2025.tif   (判据与 app.py load_high_risk 一致:
                   Sen斜率 <= 全省最负 20% 分位 且 Hurst > 0.5)
"""
import csv
from collections import Counter
from pathlib import Path

import numpy as np
import rasterio

# ==================== 配置 (仅此处与原脚本不同) ====================
ROOT = Path(r"D:\大二\干旱预测")
YEARS = list(range(2016, 2026))
OUT_NODATA = -9999.0
# 待分析数据集: (输入 tif, 输出文件名前缀); 根区版已跑过, 需要重跑时取消注释
DATASETS = [
    (ROOT / "4-5月河南SSI" / "4-5月河南SSI_shallow_2016-2025.tif", "河南浅层SSI"),
    # (ROOT / "4-5月河南SSI" / "4-5月河南SSI_swvl2_2016-2025.tif", "河南swvl2SSI"),
    # (ROOT / "4-5月河南SSI" / "4-5月河南SSI_swvl3_2016-2025.tif", "河南swvl3SSI"),
    # (ROOT / "4-5月河南SSI" / "4-5月河南SSI_root_2016-2025.tif", "河南根区SSI"),
]
SRC_PATH = None  # 由 main 按数据集设置
TAG = None
# ==================================================================


def open_src():
    ds = rasterio.open(SRC_PATH)
    return ds


# ==================== 1. 干旱频率 (drought_frequency_45.py 原逻辑) ====================
def part1_frequency():
    print("=" * 60)
    print("1. 干旱频率统计 (SSI <= 0 波段占比, 逻辑同 drought_frequency_45.py)")
    print("=" * 60)
    out_dir = ROOT / "干旱频率"
    out_path = out_dir / f"{TAG}_4-5月_干旱频率_2016-2025.tif"

    with rasterio.open(SRC_PATH) as src:
        cube = src.read()  # (波段, 行, 列)
        profile = src.profile.copy()
        nodata = src.nodata
        labels = src.descriptions

    n_band = cube.shape[0]
    print(f"读取 {SRC_PATH.name}: {n_band} 个波段, 像元 {cube.shape[2]} x {cube.shape[1]}")

    valid = cube > nodata if nodata is not None else np.ones_like(cube, dtype=bool)
    if not valid.all(axis=0).all():
        n_valid_band = valid.sum(axis=0)
        print(f"存在无效值像元: {int((n_valid_band < n_band).sum())} 个 (不足 {n_band} 个有效波段)")

    # 干旱等级划分: 1=不干旱, 2=轻度, 3=中度, 4=重度, 5=极端, 0=无效
    level = np.zeros_like(cube, dtype=np.uint8)
    level[valid] = 1
    level[valid & (cube <= 0)] = 2
    level[valid & (cube <= -1.0)] = 3
    level[valid & (cube <= -1.5)] = 4
    level[valid & (cube <= -2.0)] = 5

    names = {1: "不干旱", 2: "轻度干旱", 3: "中度干旱", 4: "重度干旱", 5: "极端干旱"}
    for code, name in names.items():
        print(f"  {name}: {int((level == code).sum())} 个 (波段,像元)")

    dry = valid & (cube <= 0)
    n_dry = dry.sum(axis=0)
    n_ok = valid.sum(axis=0)

    freq = np.full(n_dry.shape, OUT_NODATA, dtype=np.float32)
    mask = n_ok > 0
    freq[mask] = n_dry[mask] / n_ok[mask]

    profile.update(count=1, dtype="float32", nodata=OUT_NODATA, compress="deflate")
    out_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(freq, 1)
        dst.set_band_description(1, "干旱频率(SSI<=0)")

    print(f"\n干旱频率统计 (有效像元 {int(mask.sum())} / {mask.size}):")
    print(f"  最小 {freq[mask].min():.4f}  最大 {freq[mask].max():.4f}  均值 {freq[mask].mean():.4f}")
    print(f"完成! 输出: {out_path}\n")
    return out_path


# ==================== 2. MK 趋势检验 (mk_trend_45.py 原逻辑) ====================
def part2_mk():
    print("=" * 60)
    print("2. Mann-Kendall 趋势检验 (年序列=4/5月均值, 逻辑同 mk_trend_45.py)")
    print("=" * 60)
    import pymannkendall as mk

    out_dir = ROOT / "Mann-Kendall 趋势检验"

    with rasterio.open(SRC_PATH) as src:
        cube = src.read().astype(np.float64)  # (20, 行, 列)
        profile = src.profile.copy()
        nodata = src.nodata
        labels = src.descriptions

    n_band = cube.shape[0]
    n_year = len(YEARS)
    print(f"读取 {SRC_PATH.name}: {n_band} 个波段, 像元 {cube.shape[2]} x {cube.shape[1]}")

    order = [f"{y}-{m:02d}" for y in YEARS for m in (4, 5)]
    if tuple(labels) != tuple(order):
        raise ValueError(f"波段顺序与预期不符: {labels}")

    cube = cube.reshape(n_year, 2, cube.shape[1], cube.shape[2])
    valid = cube > nodata if nodata is not None else np.ones_like(cube, dtype=bool)

    with np.errstate(invalid="ignore"):
        cube[~valid] = np.nan
        annual = np.nanmean(cube, axis=1)  # (年, 行, 列)

    n_valid_year = np.sum(~np.isnan(annual), axis=0)
    print(f"年序列年数: {n_year}, 有效像元: {int((n_valid_year == n_year).sum())} / {n_valid_year.size}")

    z_arr = np.full(annual.shape[1:], OUT_NODATA, dtype=np.float32)
    p_arr = np.full_like(z_arr, OUT_NODATA)
    slope_arr = np.full_like(z_arr, OUT_NODATA)

    rows, cols = np.where(n_valid_year == n_year)
    print(f"待检验像元: {len(rows)} 个 ...")

    n_sig, n_pos, n_neg = 0, 0, 0
    for r, c in zip(rows, cols):
        res = mk.original_test(annual[:, r, c], alpha=0.05)
        z_arr[r, c] = res.z
        p_arr[r, c] = res.p
        slope_arr[r, c] = res.slope
        if res.p < 0.05:
            n_sig += 1
            n_pos += res.slope > 0
            n_neg += res.slope < 0

    profile.update(count=1, dtype="float32", nodata=OUT_NODATA, compress="deflate")
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = [
        (z_arr, "MK_Z值", "Mann-Kendall Z"),
        (p_arr, "MK_p值", "Mann-Kendall p"),
        (slope_arr, "MK_Sen斜率", "Sen's slope (SSI/年)"),
    ]
    for arr, name, desc in outputs:
        out_path = out_dir / f"{TAG}_4-5月_{name}_2016-2025.tif"
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(arr, 1)
            dst.set_band_description(1, desc)
        print(f"已输出: {out_path.name}")

    sig = p_arr[p_arr > OUT_NODATA / 2] < 0.05
    z_valid = z_arr[z_arr > OUT_NODATA / 2]
    sl_valid = slope_arr[slope_arr > OUT_NODATA / 2]
    print(f"\n结果摘要 (有效像元 {len(z_valid)} 个):")
    print(f"  Z 值范围: {z_valid.min():.3f} ~ {z_valid.max():.3f}")
    print(f"  Sen's slope 范围: {sl_valid.min():.4f} ~ {sl_valid.max():.4f} (SSI/年)")
    print(f"  p < 0.05 显著像元: {len(z_valid[sig])} 个 (上升 {n_pos}, 下降 {n_neg})")
    print(f"完成! 输出目录: {out_dir}\n")
    return out_dir


# ==================== 3. Hurst 指数 (hurst_45.py 原逻辑) ====================
def hurst_rs(x, min_n=4):
    """手写 R/S 分析估计 Hurst 指数 (与原 hurst_45.py 完全一致)"""
    x = np.asarray(x, dtype=np.float64)
    n_total = len(x)
    if np.allclose(x, x[0]):  # 常数序列无意义
        return np.nan
    ns = [n for n in range(min_n, n_total + 1) if n_total % n == 0]
    log_n, log_rs = [], []
    for n in ns:
        segs = x.reshape(-1, n)
        rs_vals = []
        for seg in segs:
            dev = np.cumsum(seg - seg.mean())
            r = dev.max() - dev.min()
            s = seg.std()
            if s > 0:
                rs_vals.append(r / s)
        if rs_vals:
            log_n.append(np.log(n))
            log_rs.append(np.log(np.mean(rs_vals)))
    if len(log_n) < 2:
        return np.nan
    slope, _ = np.polyfit(log_n, log_rs, 1)
    return slope


def part3_hurst():
    print("=" * 60)
    print("3. Hurst 指数 (手写 R/S, 逻辑同 hurst_45.py)")
    print("=" * 60)
    out_dir = ROOT / "hurst指数"
    out_path = out_dir / f"{TAG}_4-5月_Hurst指数_2016-2025.tif"

    with rasterio.open(SRC_PATH) as src:
        cube = src.read()  # (20, 行, 列)
        profile = src.profile.copy()
        nodata = src.nodata

    print(f"读取 {SRC_PATH.name}: {cube.shape[0]} 个波段, 像元 {cube.shape[2]} x {cube.shape[1]}")

    valid = cube > nodata if nodata is not None else np.ones_like(cube, dtype=bool)
    n_valid = valid.sum(axis=0)
    rows, cols = np.where(n_valid == cube.shape[0])
    print(f"待计算像元: {len(rows)} 个 (需 {cube.shape[0]} 个波段全部有效) ...")

    h_arr = np.full(cube.shape[1:], OUT_NODATA, dtype=np.float32)

    n_ok, n_fail = 0, 0
    for r, c in zip(rows, cols):
        h = hurst_rs(cube[:, r, c])
        if np.isfinite(h):
            h_arr[r, c] = h
            n_ok += 1
        else:
            n_fail += 1

    profile.update(count=1, dtype="float32", nodata=OUT_NODATA, compress="deflate")
    out_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(h_arr, 1)
        dst.set_band_description(1, "Hurst 指数 (R/S)")

    m = h_arr > OUT_NODATA / 2
    print(f"\n结果摘要 (成功 {n_ok} 个, 失败 {n_fail} 个):")
    if m.any():
        print(f"  Hurst 指数范围: {h_arr[m].min():.4f} ~ {h_arr[m].max():.4f}, 均值 {h_arr[m].mean():.4f}")
        print(f"  H > 0.5 (持续性): {int((h_arr[m] > 0.5).sum())} 个")
        print(f"  H < 0.5 (反持续性): {int((h_arr[m] < 0.5).sum())} 个")
        print(f"  |H - 0.5| < 0.05 (近随机): {int((np.abs(h_arr[m] - 0.5) < 0.05).sum())} 个")
    else:
        print("  无有效结果")
    print(f"完成! 输出: {out_path}\n")
    return out_path


# ==================== 4. 游程理论干旱事件 (run_theory_45.py 原逻辑) ====================
def part4_run_theory():
    print("=" * 60)
    print("4. 游程理论干旱事件识别 (阈值 SSI <= -1.0, 逻辑同 run_theory_45.py)")
    print("=" * 60)
    out_dir = ROOT / "游程理论"
    out_path = out_dir / f"{TAG}_4-5月_干旱事件_按真实时间_2016-2025.csv"

    DRY_THR = -1.0  # 干旱阈值 (含)

    with rasterio.open(SRC_PATH) as src:
        cube = src.read()  # (20, 行, 列)
        nodata = src.nodata
        labels = list(src.descriptions)
        transform = src.transform

    print(f"读取 {SRC_PATH.name}: {cube.shape[0]} 个波段, 像元 {cube.shape[2]} x {cube.shape[1]}")

    expected = [f"{y}-{m:02d}" for y in range(2016, 2026) for m in (4, 5)]
    if labels != expected:
        raise ValueError(f"波段顺序与预期不符: {labels}")

    valid = cube > nodata if nodata is not None else np.ones_like(cube, dtype=bool)
    n_valid = valid.sum(axis=0)
    rows, cols = np.where(n_valid == cube.shape[0])
    print(f"待分析像元: {len(rows)} 个 ...")

    events_rows = []
    n_event_px = 0
    for r, c in zip(rows, cols):
        x = cube[:, r, c].astype(np.float64)
        paired = x.reshape(10, 2)  # 每年 [4月, 5月]

        px_events = []
        for yi in range(10):
            apr_d = paired[yi, 0] <= DRY_THR
            may_d = paired[yi, 1] <= DRY_THR
            if apr_d and may_d:
                s, e = 2 * yi, 2 * yi + 1  # 4-5 月连旱, 持续 2 个月
            elif apr_d:
                s = e = 2 * yi            # 仅 4 月, 持续 1 个月
            elif may_d:
                s = e = 2 * yi + 1        # 仅 5 月, 持续 1 个月
            else:
                continue
            px_events.append((s, e))

        if px_events:
            n_event_px += 1
        lon, lat = transform * (c + 0.5, r + 0.5)
        for s, e in px_events:
            dur = e - s + 1
            severity = float(-x[s : e + 1].sum())
            events_rows.append(
                [int(r), int(c), round(lon, 4), round(lat, 4),
                 labels[s], labels[e], dur, round(severity, 4)]
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    header = ["行", "列", "经度", "纬度", "起始月份", "结束月份", "持续月数", "累积强度"]
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(events_rows)

    durs = np.array([row[6] for row in events_rows], dtype=float)
    sevs = np.array([row[7] for row in events_rows], dtype=float)
    n_1m = int((durs == 1).sum())
    n_2m = int((durs == 2).sum())
    print(f"\n统计摘要:")
    print(f"  发生事件的像元: {n_event_px} / {len(rows)}")
    print(f"  事件总数: {len(events_rows)}")
    if len(events_rows):
        print(f"  持续 1 个月: {n_1m} 个 | 持续 2 个月 (同年 4-5 月连旱): {n_2m} 个")
        print(f"  累积强度: 1 月事件 {sevs[durs == 1].min():.2f} ~ {sevs[durs == 1].max():.2f} (均值 {sevs[durs == 1].mean():.2f})"
              if n_1m else "  无 1 月事件")
        if n_2m:
            print(f"  累积强度: 2 月事件 {sevs[durs == 2].min():.2f} ~ {sevs[durs == 2].max():.2f} (均值 {sevs[durs == 2].mean():.2f})")
        by_year = Counter(row[4][:4] for row in events_rows)
        print(f"  分年事件数: " + ", ".join(f"{y}年{by_year[y]}" for y in sorted(by_year)))
    print(f"完成! 输出: {out_path}\n")
    return out_path


# ==================== 5. 未来高风险区 (判据与 app.py load_high_risk 一致) ====================
def part5_high_risk(mk_dir, hurst_path):
    print("=" * 60)
    print("5. 未来高风险区筛选 (Sen斜率 <= 最负20%分位 且 Hurst > 0.5, 判据与 app.py 一致)")
    print("=" * 60)
    out_dir = ROOT / "未来高风险区"
    out_path = out_dir / f"{TAG}_4-5月_未来高风险_2016-2025.tif"

    def load_layer(path):  # 与 app.py load_layer 相同的读取方式
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float64)
            nodata = src.nodata
            arr[arr <= nodata / 2] = np.nan
            return arr

    s = load_layer(mk_dir / f"{TAG}_4-5月_MK_Sen斜率_2016-2025.tif")
    h = load_layer(hurst_path)
    valid = np.isfinite(s) & np.isfinite(h)
    q20 = float(np.quantile(s[valid], 0.2))  # Sen 斜率最负的前 20% 分位
    risk = (s <= q20) & (h > 0.5)
    print(f"判据: Sen斜率 <= 最负20%分位 ({q20:.3f}) 且 Hurst > 0.5")

    # 输出掩膜 tif: 1=高风险, 0=非风险, nodata=无效
    with rasterio.open(hurst_path) as ref:
        profile = ref.profile.copy()
    profile.update(count=1, dtype="uint8", nodata=255, compress="deflate")
    out_arr = np.full(risk.shape, 255, dtype=np.uint8)
    out_arr[valid] = 0
    out_arr[risk] = 1
    out_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(out_arr, 1)
        dst.set_band_description(1, "未来高风险区 (1=风险)")

    n_risk, n_valid = int(risk.sum()), int(valid.sum())
    print(f"统计摘要 (有效像元 {n_valid}):")
    print(f"  高风险像元: {n_risk} ({n_risk / n_valid * 100:.1f}%)")
    print(f"  Sen 斜率有效范围: {np.nanmin(s):.4f} ~ {np.nanmax(s):.4f}")
    print(f"  Hurst 有效范围: {np.nanmin(h):.4f} ~ {np.nanmax(h):.4f}")
    print(f"完成! 输出: {out_path}")
    return out_path


if __name__ == "__main__":
    for SRC_PATH, TAG in DATASETS:  # 模块级赋值: 各 part 函数读全局 SRC_PATH/TAG
        print("\n" + "#" * 60)
        print(f"# 数据集: {SRC_PATH.name}  (输出前缀: {TAG})")
        print("#" * 60)
        if not SRC_PATH.exists():
            print(f"[跳过] 找不到输入: {SRC_PATH}")
            continue
        freq_path = part1_frequency()
        mk_dir = part2_mk()
        hurst_path = part3_hurst()
        events_path = part4_run_theory()
        risk_path = part5_high_risk(mk_dir, hurst_path)

        print("\n" + "=" * 60)
        print(f"{TAG} 全部 5 项分析完成, 输出清单:")
        print("=" * 60)
        for p in [freq_path, mk_dir, hurst_path, events_path, risk_path]:
            print(f"  {p}")
