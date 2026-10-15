# -*- coding: utf-8 -*-
"""
五层 SSI x 冬小麦产量相关系数对比柱状图 (深色主题, 适配深色网站与答辩 PPT)
数据源: 产量关联分析\\层级产量关联对比表.csv (yield_ssi_analysis_layers.py 生成)
输出: output\\五层SSI与产量相关系数对比.png (深色透明背景, 覆盖生成)
风格: 显著砖红 rgb(200,70,70) / 不显著深灰 rgb(100,100,100); 文字浅灰白; 仅保留左/下轴线
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(r"D:\大二\干旱预测")
# 注册随仓库分发的中文字体, 避免中文渲染为方框
_FONT = ROOT / "assets" / "NotoSansCJKsc-Regular.otf"
if _FONT.exists():
    fm.fontManager.addfont(str(_FONT))
    matplotlib.rcParams["font.sans-serif"] = [fm.FontProperties(fname=str(_FONT)).get_name(),
                                              "Microsoft YaHei", "SimHei", "DejaVu Sans"]
else:
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
CSV = ROOT / "产量关联分析" / "层级产量关联对比表.csv"
OUT_DIR = ROOT / "output"
OUT_PNG = OUT_DIR / "五层SSI与产量相关系数对比.png"

# 深色主题配色
C_SIG = (200 / 255, 70 / 255, 70 / 255)     # 显著: 砖红
C_NS = (100 / 255, 100 / 255, 100 / 255)    # 不显著: 深灰
C_TXT = "#e6e6e6"                            # 主文字: 浅灰白
C_AXIS = "#b3b3b3"                           # 轴线/刻度
BG = "#0e1117"                               # 深底色 (与 Streamlit 深色主题一致)


def sig_stars(p):
    return "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))


def main():
    df = pd.read_csv(CSV, encoding="utf-8-sig")
    df = df.sort_values("相关系数r", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10.5, 6.4), dpi=150)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    colors = [C_SIG if p < 0.05 else C_NS for p in df["p值"]]
    bars = ax.bar(df["层级"], df["相关系数r"], color=colors, width=0.62, zorder=3)

    # 柱顶 r 值 + 星号 (负值柱标在下方, 正值柱标在上方)
    for bar, r, p in zip(bars, df["相关系数r"], df["p值"]):
        y = bar.get_height()
        va = "top" if y < 0 else "bottom"
        off = -0.012 if y < 0 else 0.012
        ax.text(bar.get_x() + bar.get_width() / 2, y + off, f"{r:.2f}{sig_stars(p)}",
                ha="center", va=va, fontsize=16, color=C_TXT, fontweight="bold")

    # 极简: 仅保留左/下轴线, 去网格与多余边框
    ax.axhline(0, color="#888888", lw=1.0, zorder=2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(C_AXIS)
    ax.set_ylim(-0.42, 0.2)
    ax.set_ylabel("Pearson r", fontsize=15, color=C_AXIS)
    ax.tick_params(axis="y", colors=C_AXIS, labelsize=12)
    ax.tick_params(axis="x", colors=C_TXT, labelsize=14)
    plt.setp(ax.get_xticklabels(), rotation=-18, ha="left", rotation_mode="anchor")

    # 结论文字 (图下方)
    fig.text(0.5, 0.012,
             "表层（0-7cm）与浅层根区（0-28cm）同产量距平显著负相关，深层及全根区加权不显著——0-28cm 对土壤水分异常波动最敏感。\n"
             "减产由旱涝双端中的偏湿端驱动：重度偏湿年平均减产 2.5%~2.9%，中度干旱年减产率≈0；\n"
             "剔除 2018（湿害）、2023（烂场雨）两个偏湿灾年后 r 归零（+0.02）——样本期负相关主要由这两个年份贡献，偏旱端效应待更长序列验证。",
             ha="center", va="bottom", fontsize=12, color=C_TXT, wrap=True)

    fig.tight_layout(rect=(0, 0.12, 1, 1))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.25)
    print(f"已保存: {OUT_PNG}")


if __name__ == "__main__":
    main()
