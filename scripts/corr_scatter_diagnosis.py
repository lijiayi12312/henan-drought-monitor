# -*- coding: utf-8 -*-
r"""浅层根区 SSI x 单产距平 相关性诊断 (n=162 市-年配对)
配对数据直接取自 city_yield_ssi_timeseries_shallow.csv (gen_shallow_timeseries.py 生成,
单产口径: 实际单产(吨/公顷)=总产(万吨)/播种面积(千公顷)×10, 分市线性去趋势):
  1. 提取 浅层根区 SSI x 单产距平 配对数据
  2. Pearson r + Spearman rho 双指标 (检验极端年份是否主导整体相关性)
  3. 散点图 (x=SSI, y=单产距平) + 线性拟合线
  4. 异常年份 2018 (霜冻) / 2023 (烂场雨) 用不同颜色标出
  5. 打印斜率与截距; 图存 output/浅层SSI_单产距平_散点诊断.png
运行:  设 PYTHONPATH=C:\pylibs 后 python corr_scatter_diagnosis.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
SERIES_CSV = ROOT / "output" / "产量关联分析" / "city_yield_ssi_timeseries_shallow.csv"
OUT_PNG = ROOT / "output" / "浅层SSI_单产距平_散点诊断.png"

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
    # ============ 1. 提取配对数据 (读单产口径时序表, 与主链路 gen_shallow_timeseries.py 一致) ============
    mg = pd.read_csv(SERIES_CSV)
    mg = mg.dropna(subset=["SSI", "单产距平"]).reset_index(drop=True)
    print(f"配对样本: n = {len(mg)} (市-年, 2016-2024)")

    x = mg["SSI"].to_numpy(float)
    y = mg["单产距平"].to_numpy(float)

    # ============ 2. Pearson + Spearman ============
    r_p, p_p = stats.pearsonr(x, y)
    rho_s, p_s = stats.spearmanr(x, y)
    slope, intercept = np.polyfit(x, y, 1)
    print(f"\nPearson  r   = {r_p:+.4f}   (p = {p_p:.3e})")
    print(f"Spearman rho = {rho_s:+.4f}   (p = {p_s:.3e})")
    print(f"线性拟合: 单产距平 = {slope:.4f} × SSI + {intercept:.4f}")

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
    ax.set_ylabel("单产距平 (吨/公顷, 负=减产)", fontsize=13, color=C_AXIS)
    ax.set_title("浅层根区 SSI x 冬小麦单产距平 散点诊断 (2016-2024, 市-年)",
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
