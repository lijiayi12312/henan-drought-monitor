# -*- coding: utf-8 -*-
r"""浅层根区 SSI x 产量距平 相关性诊断 (n=162 市-年配对)
链路与 yield_ssi_analysis_layers.py 完全一致, 在此基础上:
  1. 提取 浅层根区 SSI x 产量距平 配对数据
  2. Pearson r + Spearman rho 双指标 (检验极端年份是否主导整体相关性)
  3. 散点图 (x=SSI, y=产量距平) + 线性拟合线
  4. 异常年份 2018 (霜冻) / 2023 (烂场雨) 用不同颜色标出
  5. 打印斜率与截距; 图存 output/浅层SSI_产量距平_散点诊断.png
运行:  设 PYTHONPATH=C:\pylibs 后 python corr_scatter_diagnosis.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
from scipy import stats

ROOT = Path(r"D:\大二\干旱预测")
YIELD_XLSX = Path(r"d:\桌面\henan_wheat_yield.xlsx")   # 年份/城市/夏粮产量(万吨), 2016-2024
CITY_SHP = ROOT / "河南省市shp" / "河南省.shp"
SSI_TIF = ROOT / "4-5月河南SSI" / "4-5月河南SSI_shallow_2016-2025.tif"  # 浅层根区 20 波段
OUT_PNG = ROOT / "output" / "浅层SSI_产量距平_散点诊断.png"

# 深色主题 (与 plot_layer_corr_bar.py 一致)
FONT = ROOT / "assets" / "NotoSansCJKsc-Regular.otf"
if FONT.exists():
    fm.fontManager.addfont(str(FONT))
    matplotlib.rcParams["font.sans-serif"] = [fm.FontProperties(fname=str(FONT)).get_name(),
                                             "Microsoft YaHei", "SimHei", "DejaVu Sans"]
else:
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

BG = "#0e1117"
C_TXT = "#e6e6e6"
C_AXIS = "#b3b3b3"
C_NORM = (150 / 255, 150 / 255, 165 / 255)   # 常规年份: 灰
C_2018 = (235 / 255, 160 / 255, 60 / 255)    # 2018: 橙
C_2023 = (225 / 255, 75 / 255, 75 / 255)     # 2023: 红
C_FIT = (90 / 255, 160 / 255, 225 / 255)     # 拟合线: 蓝


def norm_city(name):
    n = str(name).strip()
    for suf in ("示范区", "市"):
        if n.endswith(suf) and len(n) > len(suf):
            return n[: -len(suf)]
    return n


def main():
    # ============ 1. 提取配对数据 (链路复刻 yield_ssi_analysis_layers.py) ============
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
        trend = a * x + b
        for yr, yi, ti in zip(x.astype(int), y, trend):
            records.append({"年份": yr, "key": key, "距平": yi - ti})
    yield_df = pd.DataFrame(records)

    with rasterio.open(SSI_TIF) as src:
        cube = src.read().astype(np.float64)                # (20, rows, cols)
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
        for i, yr in enumerate(range(2016, 2026)):          # 波段 2i=4月, 2i+1=5月
            b = cube[2 * i: 2 * i + 2][:, m]
            b = b[np.isfinite(b)]
            if b.size == 0:
                continue
            ssi_recs.append({"年份": yr, "key": key, "SSI": float(b.mean())})
    ssi_df = pd.DataFrame(ssi_recs)

    mg = yield_df.merge(ssi_df[["年份", "key", "SSI"]], on=["年份", "key"], how="left")
    mg = mg.dropna(subset=["SSI"]).reset_index(drop=True)
    print(f"配对样本: n = {len(mg)} (市-年, 2016-2024)")

    x = mg["SSI"].to_numpy(float)
    y = mg["距平"].to_numpy(float)

    # ============ 2. Pearson + Spearman ============
    r_p, p_p = stats.pearsonr(x, y)
    rho_s, p_s = stats.spearmanr(x, y)
    slope, intercept = np.polyfit(x, y, 1)
    print(f"\nPearson  r   = {r_p:+.4f}   (p = {p_p:.3e})")
    print(f"Spearman rho = {rho_s:+.4f}   (p = {p_s:.3e})")
    print(f"线性拟合: 产量距平 = {slope:.4f} × SSI + {intercept:.4f}")

    # ============ 3-5. 散点图 + 拟合线 + 异常年标注 ============
    fig, ax = plt.subplots(figsize=(9.5, 6.8), dpi=150)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    yr_arr = mg["年份"].to_numpy()
    idx_norm = np.asarray([i for i in range(len(mg)) if yr_arr[i] not in (2018, 2023)])
    idx_2018 = np.where(yr_arr == 2018)[0]
    idx_2023 = np.where(yr_arr == 2023)[0]
    for label, color, idx in (
        ("常规年份", C_NORM, idx_norm),
        ("2018 (偏湿·霜冻年)", C_2018, idx_2018),
        ("2023 (偏湿·烂场雨年)", C_2023, idx_2023),
    ):
        ax.scatter(x[idx], y[idx], s=48, color=[color], alpha=0.85,
                   edgecolors=BG, linewidths=0.7, zorder=3,
                   label=f"{label} (n={len(idx)})")

    xx = np.linspace(x.min() - 0.1, x.max() + 0.1, 200)
    sign = "+" if intercept >= 0 else "-"
    ax.plot(xx, slope * xx + intercept, color=C_FIT, lw=2.2, zorder=2,
            label=f"线性拟合: y = {slope:.2f}x {sign} {abs(intercept):.1f}")

    # 零参考线 (偏干/偏湿、减产/增产 分界)
    ax.axhline(0, color="#555555", lw=1.0, ls="--", zorder=1)
    ax.axvline(0, color="#555555", lw=1.0, ls="--", zorder=1)
    ax.text(0.985, 0.045, "偏湿·减产区", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9.5, color="#777777")
    ax.text(0.015, 0.955, "偏旱·增产区", transform=ax.transAxes, ha="left", va="top",
            fontsize=9.5, color="#777777")

    leg = ax.legend(loc="upper right", fontsize=10, facecolor="#1a212b",
                    edgecolor="#3a4350", labelcolor=C_TXT, framealpha=0.9)
    leg.get_frame().set_alpha(0.9)

    ax.set_xlabel("4-5 月 浅层根区 SSI (负=偏干, 正=偏湿)", fontsize=13, color=C_AXIS)
    ax.set_ylabel("产量距平 (万吨, 负=减产)", fontsize=13, color=C_AXIS)
    ax.set_title("浅层根区 SSI x 冬小麦产量距平 散点诊断 (2016-2024, 市-年)",
                 fontsize=14.5, color=C_TXT, pad=12)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(C_AXIS)
    ax.tick_params(colors=C_AXIS, labelsize=11)
    ax.grid(True, color="#252b36", lw=0.6, alpha=0.6, zorder=0)

    fig.text(0.5, 0.015,
             f"Pearson r = {r_p:+.3f} (p={p_p:.1e})    Spearman rho = {rho_s:+.3f} (p={p_s:.1e})    "
             f"拟合: 距平 = {slope:.2f}×SSI {sign} {abs(intercept):.1f}    n = {len(mg)}",
             ha="center", va="bottom", fontsize=12, color=C_TXT)

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.25)
    print(f"\n散点图已保存: {OUT_PNG}")
    print("诊断结束 —— 按要求停在相关计算与散点图, 未做其他后续计算。")


if __name__ == "__main__":
    main()
