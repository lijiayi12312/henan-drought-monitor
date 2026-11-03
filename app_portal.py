# -*- coding: utf-8 -*-
"""
中原农情旱涝双端风险研判与空间态势感知系统 —— 网格卡片式数据门户 (浅色主题)
第 4 阶段: 精细化打磨
  Tab1 - 地图下方 2行3列 共 6 张数据产品卡片 (height=200 统一高度), 点击切换主地图:
         现状旱涝状态 / 历史偏干频率 / 历史偏湿频率 / MK趋势变干 / MK趋势变湿 / Hurst持续性
         高风险图层旱涝分离: Sen<0&Hurst>0.5=深红(未来干旱风险), Sen>0&Hurst>0.5=深蓝(未来渍害风险)
         图例悬浮于地图右上角; 点击城市 -> 综合研判框 (左数据 + 右动态研判文字)
  Tab2 - 2行3列 共 6 张核心数据集卡片 (统一高度), HTML 淡蓝表头表格 + 翻页
运行:  python -m streamlit run app_portal.py --global.developmentMode false --server.port 8502
"""
import base64
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

_pylibs = Path(r"C:\pylibs")
if _pylibs.exists():
    sys.path.append(str(_pylibs))

import folium
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

import matplotlib
matplotlib.use("Agg")
from matplotlib import cm as mpl_cm
from matplotlib import pyplot as plt
from matplotlib import colors as mpl_colors
from matplotlib import patches as mpl_patches
import matplotlib.font_manager as fm

_ROOT = Path(__file__).resolve().parent
_FONT_FILE = _ROOT / "assets" / "NotoSansCJKsc-Regular.otf"
if _FONT_FILE.exists():
    fm.fontManager.addfont(str(_FONT_FILE))
    _CJK = fm.FontProperties(fname=str(_FONT_FILE)).get_name()
    matplotlib.rcParams["font.sans-serif"] = [_CJK, "Microsoft YaHei", "SimHei", "DejaVu Sans"]
else:
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

# ---------------- 数据路径 ----------------
ROOT = _ROOT
CITY_SHP = ROOT / "data" / "河南省市shp" / "河南省.shp"
MONTHS = [f"{y}-{m:02d}" for y in range(2016, 2026) for m in (4, 5)]  # 20 个波段
VERSIONS = {
    "浅层根区（0-28cm）": {
        "ssi": ROOT / "data" / "4-5月河南SSI" / "4-5月河南SSI_shallow_2016-2025.tif",
        "sen": ROOT / "output" / "Mann-Kendall 趋势检验" / "河南浅层SSI_4-5月_MK_Sen斜率_2016-2025.tif",
        "hurst": ROOT / "output" / "hurst指数" / "河南浅层SSI_4-5月_Hurst指数_2016-2025.tif",
        "yield_series": ROOT / "output" / "产量关联分析" / "city_yield_ssi_timeseries_shallow.csv",
    },
    "表层（0-7cm）": {
        "ssi": ROOT / "data" / "4-5月河南SSI" / "4-5月河南SSI_swvl1_2016-2025.tif",
        "sen": ROOT / "output" / "Mann-Kendall 趋势检验" / "河南SSI_4-5月_MK_Sen斜率_2016-2025.tif",
        "hurst": ROOT / "output" / "hurst指数" / "河南SSI_4-5月_Hurst指数_2016-2025.tif",
        "yield_series": ROOT / "output" / "产量关联分析" / "city_yield_ssi_timeseries.csv",
    },
}

# 地图数据产品阵列 (2行3列, 点击切换主地图)
PRODUCTS = [
    {"key": "ssi", "icon": "🗺️", "name": "现状旱涝状态", "sub": "当月 SSI 距平分级"},
    {"key": "freq_dry", "icon": "🌵", "name": "历史偏干频率", "sub": "SSI<0 月份占比"},
    {"key": "freq_wet", "icon": "🌧️", "name": "历史偏湿频率", "sub": "SSI>0 月份占比"},
    {"key": "trend_dry", "icon": "🔴", "name": "未来干旱高风险区", "sub": "Sen<-0.15 且 Hurst>0.5 · 仅展示高风险区域"},
    {"key": "trend_wet", "icon": "🔵", "name": "未来渍害高风险区", "sub": "Sen>0.02 且 Hurst>0.5 · 仅展示高风险区域"},
    {"key": "hurst", "icon": "🔁", "name": "未来风险演变趋势", "sub": "趋势持续性指数"},
]

# Tab2 核心数据集卡片 (2行3列)
DATASETS = [
    {"icon": "🌾", "title": "冬小麦产量面板（浅层根区）",
     "desc": "18 市 × 10 年 4-5 月浅层根区 SSI × 实际产量 / 趋势产量 / 距平 / 减产率，主分析数据集。",
     "path": ROOT / "output" / "产量关联分析" / "city_yield_ssi_timeseries_shallow.csv"},
    {"icon": "🌱", "title": "冬小麦产量面板（表层）",
     "desc": "同上面板结构的表层 swvl1 (0-7cm) 版本，供层位间对比验证。",
     "path": ROOT / "output" / "产量关联分析" / "city_yield_ssi_timeseries.csv"},
    {"icon": "🌀", "title": "旱涝事件回溯库（游程理论）",
     "desc": "2016-2025 年 4-5 月逐像元干旱事件档案：起止月份、持续月数、累积强度。",
     "path": ROOT / "output" / "游程理论" / "河南浅层SSI_4-5月_干旱事件_按真实时间_2016-2025.csv"},
    {"icon": "📊", "title": "各市 SSI-产量关系统计（浅层）",
     "desc": "各市 Pearson r / p 值与轻中重度旱涝分级减产率汇总（浅层根区）。",
     "path": ROOT / "output" / "产量关联分析" / "city_yield_analysis_shallow.csv"},
    {"icon": "📈", "title": "各市 SSI-产量关系统计（表层）",
     "desc": "各市 Pearson r / p 值与分级减产率汇总（表层 swvl1）。",
     "path": ROOT / "output" / "产量关联分析" / "city_yield_analysis.csv"},
    {"icon": "🗂️", "title": "五层 SSI × 产量相关对比表",
     "desc": "swvl1/swvl2/浅层根区/swvl3/全根区加权 五层相关系数与旱涝双端分级减产率。",
     "path": ROOT / "output" / "产量关联分析" / "层级产量关联对比表.csv"},
]


# ---------------- 数据加载 (缓存) ----------------
@st.cache_data
def load_ssi(version):
    with rasterio.open(VERSIONS[version]["ssi"]) as src:
        cube = src.read().astype(np.float64)
        cube[cube <= src.nodata] = np.nan
        transform = src.transform
    bounds = [transform.c, transform.f + cube.shape[1] * transform.e,
              transform.c + cube.shape[2] * transform.a, transform.f]
    return cube, bounds, transform


@st.cache_data
def load_array(path_str):
    """单波段分析图层 -> (数组, bounds); nodata -> NaN"""
    with rasterio.open(path_str) as src:
        arr = src.read(1).astype(np.float64)
        arr[arr <= src.nodata / 2] = np.nan
        t = src.transform
    bounds = [t.c, t.f + arr.shape[0] * t.e, t.c + arr.shape[1] * t.a, t.f]
    return arr, bounds


@st.cache_data
def load_yield_series(version):
    return pd.read_csv(VERSIONS[version]["yield_series"], encoding="utf-8-sig")


@st.cache_data
def load_city_gdf():
    import geopandas as gpd

    gdf = gpd.read_file(CITY_SHP, encoding="utf-8")
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    return gdf[["name", "geometry"]]


@st.cache_data
def load_city_geojson():
    import json as _json
    return _json.loads(load_city_gdf().to_json())


@st.cache_data
def load_city_labels():
    pts = load_city_gdf().geometry.representative_point()
    return [(str(n), (p.y, p.x)) for n, p in zip(load_city_gdf()["name"], pts)]


@st.cache_data
def load_city_bounds():
    w, s, e, n = load_city_gdf().total_bounds
    return float(s), float(w), float(n), float(e)


def norm_city(name):
    n = str(name).strip()
    for suf in ("示范区", "市"):
        if n.endswith(suf) and len(n) > len(suf):
            return n[: -len(suf)]
    return n


def find_city_name(lon, lat):
    from shapely.geometry import Point

    gdf = load_city_gdf()
    hit = gdf[gdf.geometry.contains(Point(lon, lat))]
    return str(hit.iloc[0]["name"]) if len(hit) else None


def city_mean_ssi(city, cube, transform, band_idx):
    """该市边界内当前波段 SSI 均值 (all_touched 掩膜)"""
    from rasterio.features import geometry_mask

    gdf = load_city_gdf()
    row = gdf[gdf["name"] == city]
    if row.empty:
        return np.nan
    h, w = cube.shape[1], cube.shape[2]
    mask = geometry_mask(list(row.geometry), out_shape=(h, w), transform=transform,
                         invert=True, all_touched=True)
    vals = cube[band_idx][mask]
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if vals.size else np.nan


@st.cache_data
def city_layer_mean(version, layer_key, city):
    """该市边界内单波段分析图层 (Sen 斜率 / Hurst 指数) 的均值 (all_touched 掩膜)"""
    from rasterio.features import geometry_mask

    arr, _ = load_array(str(VERSIONS[version][layer_key]))
    gdf = load_city_gdf()
    row = gdf[gdf["name"] == city]
    if row.empty:
        return np.nan
    with rasterio.open(VERSIONS[version][layer_key]) as src:
        transform = src.transform
    mask = geometry_mask(list(row.geometry), out_shape=arr.shape, transform=transform,
                         invert=True, all_touched=True)
    vals = arr[mask]
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if vals.size else np.nan


@st.cache_data
def city_ssi_all(version, band_idx):
    """全部 18 市在当前波段的 SSI 均值 {城市名: 值}, 供极值城市 Top3 图"""
    from rasterio.features import geometry_mask

    cube, _, transform = load_ssi(version)
    gdf = load_city_gdf()
    h, w = cube.shape[1], cube.shape[2]
    out = {}
    for _, r in gdf.iterrows():
        mask = geometry_mask([r.geometry], out_shape=(h, w), transform=transform,
                             invert=True, all_touched=True)
        vals = cube[band_idx][mask]
        vals = vals[np.isfinite(vals)]
        out[str(r["name"])] = float(vals.mean()) if vals.size else np.nan
    return out


def get_diagnosis(ssi):
    """SSI 统一诊断入口 (五档科学阈值) —— 顶部等级徽章 / 中间预警框 / 底部治理决策
    三处共用此函数, 保证等级文本、预警颜色、治理建议永远一致。
    阈值: 极强偏湿>=1.5 | 轻度偏湿[0.5,1.5) | 正常(-0.5,0.5) | 轻度偏干(-1.5,-0.5] | 重度偏干<=-1.5
    返回 dict: level(等级文本) / alert(error|warning|success|info) / color(徽章色) /
              head(预警框标题) / diag(诊断一句话) / advice(治理建议)"""
    if not np.isfinite(ssi):
        return {"level": "无数据", "alert": "info", "color": "#9aa3af",
                "head": "⬜ 暂无数据", "diag": "该区域当前月份无有效 SSI 像元",
                "advice": "该区域当前月份无有效 SSI 像元，暂无法生成治理决策建议。"}
    if ssi >= 1.5:
        return {"level": "极强偏湿（涝灾）", "alert": "error", "color": "#08306b",
                "head": "⚠️ 涝灾高风险预警", "diag": "当前处于极强偏湿风险区（涝灾高风险）",
                "advice": ("治理路径建议：\n"
                           "1. 工程改造：建议在“十五五”高标准农田建设中，重点排查并疏通该区末级排水沟渠（毛沟、斗沟），提升排涝标准。\n"
                           "2. 应急响应：5月灌浆成熟期若遇连阴雨，需提前跨区调配履带式收割机，并启用应急排涝泵站。\n"
                           "3. 理念转变：扭转“重灌轻排”倾向，排涝与保墒并重。")}
    if ssi >= 0.5:
        return {"level": "轻度偏湿", "alert": "warning", "color": "#6baed6",
                "head": "🌧️ 偏湿预警", "diag": "当前呈轻度偏湿态势",
                "advice": "建议关注灌浆—收获期降水过程，提前疏通排涝设施，防范根系渍害与田间积水。"}
    if ssi > -0.5:
        return {"level": "正常波动", "alert": "success", "color": "#41ab5d",
                "head": "✅ 水分正常", "diag": "当前水分条件正常，属于常规稳产区",
                "advice": "维持现有农田水利管护机制，做好常态化监测。"}
    if ssi > -1.5:
        return {"level": "轻度偏干", "alert": "warning", "color": "#fb6a4a",
                "head": "🌵 偏干预警", "diag": "当前呈轻度偏干态势",
                "advice": "建议做好灌溉补水调度，优先保障拔节—抽穗期关键水分供应。"}
    return {"level": "重度偏干（旱灾）", "alert": "error", "color": "#cb181d",
            "head": "⚠️ 旱灾高风险预警", "diag": "当前处于重度偏干风险区（旱灾高风险）",
            "advice": "依托现有高比例有效灌溉设施，适时进行补灌保墒，保障灌溉用水调度。"}


def build_report_text(city, yr, month, version, ssi_v, lvl, diag_txt, advice_txt):
    """生成单市耕地水分风险评估与治理建议报告 (纯函数)
    每次调用从空列表重新构造, 无任何全局/累积状态, 标题与数据只出现一次。
    """
    sep = "=" * 52
    lines = [
        "河南冬小麦主产区耕地水分风险研判与数字化治理决策系统",
        "耕地水分风险评估与治理建议报告",
        sep,
        f"城市：{city}",
        f"时间：{yr} 年 {int(month[5:])} 月（4-5 月关键期）",
        f"数据层级：{version}",
        f"全市平均 SSI：{ssi_v:+.2f}",
        f"水分胁迫等级：{lvl}",
        "-" * 52,
        f"诊断结果：{diag_txt}",
        "",
        advice_txt,
        sep,
        "注：本报告由系统基于当期 SSI 诊断自动生成，供决策参考。",
    ]
    return "\n".join(lines)


SSI_STATE_BINS = [(-99, -1.5, "极端偏干"), (-1.5, -1.0, "重度偏干"), (-1.0, -0.5, "轻度偏干"),
                  (-0.5, 0.5, "正常"), (0.5, 1.0, "轻度偏湿"), (1.0, 1.5, "中度偏湿"),
                  (1.5, 99, "极端偏湿")]


def ssi_state_bin(s):
    """SSI -> 离散状态名 (七档, 供状态转移预测的转移样本分组)"""
    for lo, hi, name in SSI_STATE_BINS:
        if lo <= s < hi:
            return name
    return "正常"


def transition_probs(series_df, cur_ssi):
    """历史经验状态转移: 当前 SSI 状态 -> 下一年 极端偏湿(>1.5)/极端偏干(<-1.5) 的频率
    汇总全部 18 市 2016-2024 的相邻年转移样本 (纯函数, 每次全新计算)。
    返回 (当前状态, 样本数n, p_极湿, p_极干); 样本不足时返回 n=0。"""
    cur_state = ssi_state_bin(cur_ssi)
    df = series_df.dropna(subset=["SSI"]).sort_values(["key", "年份"])
    df = df.assign(next_ssi=df.groupby("key")["SSI"].shift(-1)).dropna(subset=["next_ssi"])
    same = df[df["SSI"].map(ssi_state_bin) == cur_state]
    n = len(same)
    if n == 0:
        return cur_state, 0, np.nan, np.nan
    return cur_state, n, float((same["next_ssi"] > 1.5).mean()), float((same["next_ssi"] < -1.5).mean())


# ---------------- 栅格渲染 (缓存) ----------------
def _rgba_of(arr, vmin, vmax, cmap_name, flip=False):
    t = np.clip((arr - vmin) / (vmax - vmin), 0, 1)
    rgba = plt.get_cmap(cmap_name)(1.0 - t if flip else t)
    rgba[~np.isfinite(arr)] = 0.0
    return rgba


def _fig_to_uri(rgba, cell_px=10, dpi=100):
    h, w = rgba.shape[0], rgba.shape[1]
    fig = plt.figure(figsize=(w / cell_px, h / cell_px), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.imshow(rgba, interpolation="bilinear")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@st.cache_data
def render_product_png(version, product_key, band_idx):
    """主地图图层 -> (PNG data URI, bounds)
    trend_dry: 仅展示高风险区 Sen<-0.15&Hurst>0.5=深红(未来干旱风险), 其余像元极浅灰
    trend_wet: 仅展示高风险区 Sen>0.02&Hurst>0.5=深蓝(未来渍害风险), 其余像元极浅灰
    freq_dry/freq_wet: 由 SSI 立方体统计 10 年 20 期中 SSI<0 / >0 的占比"""
    if product_key == "ssi":
        cube, bounds, _ = load_ssi(version)
        rgba = _rgba_of(cube[band_idx], -2.0, 2.0, "RdYlBu")  # 红=偏干, 蓝=偏湿
    elif product_key in ("freq_dry", "freq_wet"):
        cube, bounds, _ = load_ssi(version)
        fin = np.isfinite(cube)
        n_valid = fin.sum(axis=0).astype(np.float64)
        n_valid[n_valid == 0] = np.nan
        if product_key == "freq_dry":
            freq = ((cube < 0) & fin).sum(axis=0) / n_valid
            rgba = _rgba_of(freq, 0.0, 1.0, "YlOrRd")
        else:
            freq = ((cube > 0) & fin).sum(axis=0) / n_valid
            rgba = _rgba_of(freq, 0.0, 1.0, "YlGnBu")
    elif product_key in ("trend_dry", "trend_wet"):
        sen, bounds = load_array(str(VERSIONS[version]["sen"]))
        hur, _ = load_array(str(VERSIONS[version]["hurst"]))
        valid = np.isfinite(sen) & np.isfinite(hur)
        persist = valid & (hur > 0.5)
        rgba = np.zeros((*sen.shape, 4))
        rgba[valid] = (0.55, 0.60, 0.66, 0.15)  # 背景像元: 极浅灰
        if product_key == "trend_dry":
            rgba[persist & (sen < -0.15)] = (0.70, 0.09, 0.17, 0.95)      # 深红=未来干旱高风险
        else:
            rgba[persist & (sen > 0.02)] = (0.13, 0.40, 0.67, 0.95)       # 深蓝=未来渍害高风险
    else:  # hurst
        arr, bounds = load_array(str(VERSIONS[version]["hurst"]))
        rgba = _rgba_of(arr, 0.0, 1.0, "PuOr", flip=True)
    return _fig_to_uri(rgba), bounds


@st.cache_data
def render_thumb_uri(version, product_key, band_idx):
    return render_product_png(version, product_key, band_idx)[0]


@st.cache_data
def render_legend_png(version, product_key):
    """地图右上角悬浮图例 (紧凑版); 返回 PNG bytes"""
    swatches = {
        "trend_dry": [((0.70, 0.09, 0.17), "未来干旱风险 (Sen<-0.15 且 Hurst>0.5)"),
                      ((0.55, 0.60, 0.66), "其他区域 · 仅展示高风险区")],
        "trend_wet": [((0.13, 0.40, 0.67), "未来渍害风险 (Sen>0.02 且 Hurst>0.5)"),
                      ((0.55, 0.60, 0.66), "其他区域 · 仅展示高风险区")],
    }
    bars = {
        "ssi": (-2.0, 2.0, "RdYlBu", False, (-2, 0, 2), "红=偏干 · 蓝=偏湿"),
        "freq_dry": (0.0, 1.0, "YlOrRd", False, (0, 0.5, 1), "偏干频率 (SSI<0 占比)"),
        "freq_wet": (0.0, 1.0, "YlGnBu", False, (0, 0.5, 1), "偏湿频率 (SSI>0 占比)"),
        "hurst": (0.0, 1.0, "PuOr", True, (0, 0.5, 1), "Hurst 指数 (>0.5 趋势持续)"),
    }
    if product_key in swatches:
        # 图幅加宽加高, 防止标签被裁切 (配合 savefig tight)
        fig = plt.figure(figsize=(3.4, 1.05), dpi=150)
        fig.patch.set_alpha(0.0)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        for i, (color, label) in enumerate(swatches[product_key]):
            y = 0.58 - i * 0.46
            ax.add_patch(mpl_patches.Rectangle((0.02, y), 0.06, 0.30, color=color))
            ax.text(0.10, y + 0.15, label, fontsize=8.5, va="center")
    else:
        vmin, vmax, cmap_name, flip, ticks, label = bars[product_key]
        # 加高图幅, 给 colorbar 下方的 xlabel (如 红=偏干 · 蓝=偏湿) 留足空间
        fig = plt.figure(figsize=(2.6, 1.0), dpi=150)
        fig.patch.set_alpha(0.0)
        ax = fig.add_axes([0.04, 0.42, 0.92, 0.28])
        ax.patch.set_alpha(0.0)
        norm = mpl_colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap(cmap_name).reversed() if flip else plt.get_cmap(cmap_name)
        fig.colorbar(mpl_cm.ScalarMappable(norm=norm, cmap=cmap),
                     cax=ax, orientation="horizontal", ticks=ticks)
        ax.set_xlabel(label, fontsize=8.5)
        ax.tick_params(labelsize=8, length=2)
    buf = io.BytesIO()
    # tight: 自动扩展画布容纳全部文字, 杜绝任何方向上的裁切
    fig.savefig(buf, format="png", transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return buf.getvalue()


# ---------------- 页面 ----------------
st.set_page_config(page_title="河南冬小麦主产区耕地水分风险研判与数字化治理决策系统", page_icon="🌾", layout="wide")

CSS = """
<style>
:root { --portal-blue:#1f6feb; --portal-blue-dark:#0b4da2; }
html, body, [class*="css"] { font-family:"PingFang SC","Microsoft YaHei",sans-serif !important; }
.stApp { background:#F5F7FA; }
header[data-testid="stHeader"] { display:none; }
#MainMenu, footer { visibility:hidden; }
.block-container { padding-top:0rem; max-width:1440px; }

/* Hero 视觉大图 (实景背景 + 深色遮罩 + 垂直居中) */
.hero-banner { position:relative; height:380px; border-radius:0; overflow:hidden;
  background-size:cover; background-position:center;
  display:flex; align-items:center; justify-content:center; }
.hero-overlay { position:absolute; inset:0; background:rgba(0,0,0,.5); }
.hero-inner { position:relative; z-index:2; text-align:center; max-width:920px;
  padding:0 28px; width:100%; }
.hero-inner h1 { color:#fff; font-size:3em; font-weight:700; margin:0 0 12px;
  letter-spacing:1px; text-shadow:0 2px 12px rgba(0,0,0,.55); }
.hero-inner .sub { color:#E0E0E0; font-size:15px; margin:0 0 28px;
  text-shadow:0 1px 6px rgba(0,0,0,.45); }
/* 全局搜索栏 (悬浮于 Hero 底部, 真实检索) */
div[data-testid="stTextInput"] { position:relative; z-index:6; margin-top:-50px; }
div[data-testid="stTextInput"] input {
  height:54px !important; border-radius:999px !important;
  border:1px solid #e6ebf2 !important; background:#fff !important;
  box-shadow:0 8px 22px rgba(16,42,80,.22) !important;
  font-size:15px !important; padding:0 26px !important;
}
div[data-testid="stTextInput"] input::placeholder { color:#8a94a6; opacity:1; }
.search-hint { background:#fff; border:1px solid #dbe7f5; color:#12325e; font-size:12.5px;
  padding:7px 14px; border-radius:10px; box-shadow:0 2px 8px rgba(16,42,80,.08);
  text-align:center; margin-top:2px; }

.stat-card {
  background:#FFFFFF; border-radius:12px; padding:20px 22px; height:118px;
  border:1px solid #eef1f6; box-shadow:0 10px 26px rgba(16,42,80,.20);
  margin-top:22px;
  transition:box-shadow .2s ease, transform .2s ease;
}
.stat-card:hover { box-shadow:0 14px 30px rgba(16,42,80,.26); transform:translateY(-2px); }
.stat-icon { font-size:22px; }
.stat-num { font-size:30px; font-weight:700; color:var(--portal-blue); line-height:1.15; margin-top:6px; }
.stat-num small { font-size:14px; font-weight:500; color:#5b6572; margin-left:2px; }
.stat-label { font-size:13px; color:#5b6572; margin-top:2px; }

.stTabs [data-baseweb="tab-list"] { gap:8px; background:transparent; margin-top:6px; }
.stTabs [data-baseweb="tab"] {
  background:#FFFFFF; border-radius:10px 10px 0 0; padding:10px 24px;
  border:1px solid #eef1f6; border-bottom:none; font-weight:600; color:#3a4552;
}
.stTabs [aria-selected="true"] { color:var(--portal-blue); border-top:3px solid var(--portal-blue); }

[data-testid="stVerticalBlockBorderWrapper"] {
  background:#FFFFFF; border-radius:12px !important; border:1px solid #eef1f6 !important;
  box-shadow:0 4px 6px rgba(0,0,0,.05); padding:10px 14px;
}
.card-title { font-size:16px; font-weight:700; color:#1a2433; margin:2px 0 10px; }
.card-title .tag { font-size:12px; font-weight:500; color:#8a94a6; margin-left:8px; }

/* 产品卡片 (统一高度 200px 容器内部) */
.prod-title { font-size:14px; font-weight:700; color:#1a2433; line-height:1.3; }
.prod-sub { font-size:11.5px; color:#8a94a6; margin:1px 0 4px; }
.prod-thumb { width:100%; height:82px; object-fit:cover; border-radius:6px;
              border:1px solid #eef1f6; display:block; }

/* 数据集卡片 */
.ds-title { font-size:15px; font-weight:700; color:#1a2433; margin-bottom:6px; line-height:1.35; }
.ds-desc { font-size:12.5px; color:#5b6572; line-height:1.5; min-height:74px; }

.stButton>button, .stDownloadButton>button {
  border-radius:8px; background:var(--portal-blue); color:#fff; border:none;
  box-shadow:0 2px 6px rgba(31,111,235,.25); font-weight:600; padding:0 20px;
}
.stButton>button:hover, .stDownloadButton>button:hover { background:var(--portal-blue-dark); color:#fff; }
.stButton>button:disabled { background:#9fbdf2; color:#fff; }

/* HTML 数据表 (淡蓝吸顶表头, 细线框, 滚动) */
.tbl-wrap { max-height:520px; overflow:auto; border:1px solid #eef1f6; border-radius:10px; }
table.portal-tbl { width:100%; border-collapse:collapse; font-size:13px; }
.portal-tbl thead th {
  position:sticky; top:0; z-index:2; background:#dbe7f5; color:#12325e;
  font-weight:700; padding:9px 12px; text-align:left; white-space:nowrap;
}
.portal-tbl tbody td { padding:8px 12px; border-top:1px solid #f0f3f8; color:#333; white-space:nowrap; }
.portal-tbl tbody tr:hover { background:#f5f9ff; }

/* 综合研判框 */
.judge-box { border-radius:10px; padding:14px 18px; font-size:13.5px; line-height:1.75;
             border:1px solid; height:100%; }
.judge-box .jb-head { font-weight:700; font-size:14px; margin-bottom:4px; }

.lvl-badge {
  display:inline-block; border-radius:999px; padding:4px 16px; color:#fff;
  font-size:13.5px; font-weight:700; letter-spacing:1px;
}

/* ---- Hover 公式提示 (灰色问号 + LaTeX 气泡) ---- */
.hf-wrap { display:inline-block; position:relative; vertical-align:middle; margin-left:6px; }
.hf-icon {
  display:inline-flex; align-items:center; justify-content:center;
  width:18px; height:18px; border-radius:50%;
  background:#c6ccd6; color:#fff; font-size:12px; font-weight:600;
  cursor:help; user-select:none; line-height:1;
  transition:background .15s ease;
}
.hf-icon:hover { background:#9aa3af; }
.hf-tip {
  position:absolute; bottom:calc(100% + 10px); left:50%; transform:translateX(-50%);
  background:#fff; color:#1a2433; border:1px solid #e8edf3; border-radius:12px;
  padding:12px 16px; box-shadow:0 10px 32px rgba(16,42,80,.20);
  white-space:nowrap; line-height:1.5;
  opacity:0; visibility:hidden; transition:opacity .18s ease, visibility .18s ease;
  z-index:9999; pointer-events:none;
}
.hf-tip::after {
  content:""; position:absolute; top:100%; left:50%; transform:translateX(-50%);
  border:6px solid transparent; border-top-color:#fff;
}
.hf-wrap:hover .hf-tip { opacity:1; visibility:visible; }
.hf-label { font-size:11.5px; font-weight:700; color:#8a94a6; letter-spacing:.5px; margin-bottom:6px; }
.hf-math { font-size:15px; }
.hf-math .katex { font-size:15px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# KaTeX CDN (渲染 hover 公式气泡里的 LaTeX)
st.markdown(
    """
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
    <script src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
    <script>
    (function(){
      if (!window.__hfInit) {
        window.__hfInit = true;
        window.__hfRender = function(){
          if (typeof katex === 'undefined') return;
          document.querySelectorAll('.hf-math:not([data-hf-done])').forEach(function(el){
            var txt = el.textContent.trim();
            if (txt.slice(0,2) === '$$') txt = txt.slice(2);
            if (txt.slice(-2) === '$$') txt = txt.slice(0, -2);
            try { katex.render(txt, el, {displayMode:false, throwOnError:false}); }
            catch(e){ el.textContent = txt; }
            el.setAttribute('data-hf-done','1');
          });
        };
        new MutationObserver(window.__hfRender).observe(document.body, {childList:true, subtree:true});
        window.addEventListener('load', window.__hfRender);
      }
      if (typeof katex !== 'undefined') { window.__hfRender(); }
      else { document.addEventListener('DOMContentLoaded', window.__hfRender); }
    })();
    </script>
    """,
    unsafe_allow_html=True,
)


def hover_formula(text, formula):
    """灰色问号悬浮图标: 悬停弹出带 LaTeX 公式的白色气泡。
    text 为气泡顶部标签 (如 "SSI公式"), formula 为 LaTeX 源码 (内部用 $$ 包裹由 KaTeX 渲染)。"""
    st.markdown(
        f'<span class="hf-wrap" title="{text}">'
        f'<span class="hf-icon">?</span>'
        f'<span class="hf-tip">'
        f'<span class="hf-label">{text}</span>'
        f'<span class="hf-math">$${formula}$$</span>'
        f'</span></span>',
        unsafe_allow_html=True,
    )

# ---- Hero 视觉大图 Banner (实景背景 + 深色遮罩) ----
@st.cache_data
def load_banner_b64():
    p = ROOT / "assets" / "wheat_banner.jpg"
    return base64.b64encode(p.read_bytes()).decode() if p.exists() else ""


_b64 = load_banner_b64()
_bg = f"url('data:image/jpeg;base64,{_b64}')" if _b64 else "none"
st.markdown(
    f"""
    <div class="hero-banner" style="background-image:{_bg};">
      <div class="hero-overlay"></div>
      <div class="hero-inner">
        <h1>河南冬小麦主产区耕地水分风险研判与数字化治理决策系统</h1>
        <p class="sub">Henan Winter Wheat Cropland Moisture Risk Assessment &amp; Digital Governance Decision Support System · 覆盖 2016-2025 年多深度土壤水分数据集</p>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---- 全局搜索栏 (真实检索, 悬浮于 Banner 底部) ----
if st.session_state.get("_pending_clear"):
    st.session_state["global_search"] = ""
    st.session_state["_pending_clear"] = False

_sL, _sC, _sR = st.columns([1, 2.4, 1])
with _sC:
    search_query = st.text_input(
        "全局检索", "", key="global_search", label_visibility="collapsed",
        placeholder="🔍 输入城市（如：商丘）、年份（如：2023）或数据集名称进行检索...")
    q = (search_query or "").strip()

    # 解析关键词: 年份 -> 时间滑块; 城市 -> 综合研判定位; 数据集名 -> 自动打开
    year_hit = next((str(y) for y in range(2016, 2026) if str(y) in q), None) if q else None
    city_hit = None
    if q:
        for _n in load_city_gdf()["name"].astype(str):
            if norm_city(_n) and norm_city(_n) in q:
                city_hit = _n
                break
        if city_hit is None:
            city_hit = next((_n for _n in load_city_gdf()["name"].astype(str)
                             if len(q) >= 2 and q in _n), None)
    ds_hit = None
    if q and not city_hit and not year_hit:
        ds_hit = next((d for d in DATASETS if q in d["title"]), None)

    # 联动注入 (仅当关键词变化时执行一次, 不覆盖用户后续手动操作)
    if year_hit:
        if st.session_state.get("_sy_applied") != year_hit:
            if f"{year_hit}-04" in MONTHS:
                st.session_state["month"] = f"{year_hit}-04"   # 时间滑块跳至该年 4 月
            st.session_state["_sy_applied"] = year_hit
    else:
        st.session_state["_sy_applied"] = ""

    if city_hit:
        if st.session_state.get("_sc_applied") != city_hit:
            _g = load_city_gdf()
            _row = _g[_g["name"] == city_hit]
            if len(_row):
                _pt = _row.geometry.representative_point().iloc[0]  # 保证点在市界内
                st.session_state["click"] = {"lon": float(_pt.x), "lat": float(_pt.y)}
            st.session_state["_sc_applied"] = city_hit
    else:
        st.session_state["_sc_applied"] = ""

    if ds_hit:
        if st.session_state.get("_sd_applied") != ds_hit["title"]:
            st.session_state["ds_active"] = ds_hit["title"]
            st.session_state[f"page_{ds_hit['title']}"] = 0
            st.session_state["_sd_applied"] = ds_hit["title"]
    else:
        st.session_state["_sd_applied"] = ""

    # 搜索状态提示
    if q:
        _parts = []
        if city_hit:
            _parts.append(f"📍 已定位 <b>{city_hit}</b> · 地图下方已生成综合研判卡片")
        if year_hit:
            _parts.append(f"⏳ 时间滑块已跳转至 <b>{year_hit}-04</b>")
        if ds_hit:
            _parts.append(f"🗂️ 已自动打开数据集 <b>{ds_hit['title']}</b>")
        if _parts:
            st.markdown("<div class='search-hint'>🔍 " + " ｜ ".join(_parts) + "</div>",
                        unsafe_allow_html=True)
        else:
            st.markdown(
                "<div class='search-hint' style='border-color:#f5c6cb;color:#8a1c1c'>"
                f"🔍 未识别到对应的城市 / 年份 / 数据集，请检查关键词（如：商丘、2023、产量面板）</div>",
                unsafe_allow_html=True)
        if st.button("✖ 清空搜索", key="btn_clear_search"):
            st.session_state["_pending_clear"] = True
            st.rerun()

# ---- 统计卡片 ----
STAT_CARDS = [
    ("📍", "18", "个", "地市覆盖"),
    ("🗄️", "162", "条", "面板数据 (市-年)"),
    ("⏳", "10", "年", "长时序记录 (2016–2025)"),
    ("🗺️", "6", "类", "空间数据产品"),
]
for col, (icon, num, unit, label) in zip(st.columns(4), STAT_CARDS):
    col.markdown(
        f'<div class="stat-card"><div class="stat-icon">{icon}</div>'
        f'<div class="stat-num">{num}<small>{unit}</small></div>'
        f'<div class="stat-label">{label}</div></div>',
        unsafe_allow_html=True,
    )
st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

tab_map, tab_db = st.tabs(["空间态势感知", "旱涝属性数据库"])

# ================= Tab 1: 空间态势感知 =================
with tab_map:
    map_col, ctrl_col = st.columns([3.5, 1.5], gap="medium")

    active_product = st.session_state.get("map_layer", "ssi")
    prod = next(p for p in PRODUCTS if p["key"] == active_product)

    # ---- 右侧: 控制面板 + 全省旱涝面积概览 + 极值城市 Top3 ----
    with ctrl_col:
        with st.container(border=True):
            st.markdown('<div class="card-title">控制面板</div>', unsafe_allow_html=True)
            version = st.selectbox("数据层级", list(VERSIONS.keys()), index=0, key="ver",
                                   help="浅层根区为 0-28cm 两层平均, 与冬小麦产量相关性最强; 表层为 0-7cm swvl1")
            month = st.select_slider("月份", options=MONTHS, value=MONTHS[-1], key="month",
                                     help="每年 4、5 月, 共 20 期; 仅作用于「现状旱涝状态」图层与综合研判框")
            st.caption("当前: " + version + " · " + month)

        band_idx = MONTHS.index(month)
        cube, _, transform = load_ssi(version)

        with st.container(border=True):
            st.markdown(f'<div class="card-title">🍩 全省旱涝面积概览<span class="tag">{month}</span></div>',
                        unsafe_allow_html=True)
            band = cube[band_idx]
            fin = np.isfinite(band)
            n_dry = int(((band < -1) & fin).sum())
            n_wet = int(((band > 1) & fin).sum())
            n_norm = int(fin.sum()) - n_dry - n_wet
            fig_pie = go.Figure(go.Pie(
                labels=["偏干 (SSI<-1)", "正常", "偏湿 (SSI>1)"],
                values=[n_dry, n_norm, n_wet], hole=0.58,
                marker=dict(colors=["#D32F2F", "#c9d1d9", "#1976D2"]),  # 红=偏干, 蓝=偏湿
                textinfo="percent", textposition="inside", textfont_size=12, sort=False))
            fig_pie.update_layout(height=300, margin=dict(l=10, r=10, t=16, b=42),
                                  showlegend=True,
                                  legend=dict(orientation="h", yanchor="top", y=-0.05,
                                              xanchor="center", x=0.5, font_size=11),
                                  paper_bgcolor="rgba(0,0,0,0)",
                                  annotations=[dict(text=month, x=0.5, y=0.5, font_size=13,
                                                    showarrow=False, font_color="#5b6572")])
            st.plotly_chart(fig_pie, use_container_width=True)

        with st.container(border=True):
            st.markdown(f'<div class="card-title">🏙️ 重点治理靶区 Top3<span class="tag">SSI 最低 / 最高</span></div>',
                        unsafe_allow_html=True)
            srt = sorted(((k, v) for k, v in city_ssi_all(version, band_idx).items()
                          if np.isfinite(v)), key=lambda kv: kv[1])
            low3, high3 = srt[:3], srt[-3:][::-1]
            names = [k for k, _ in high3] + [k for k, _ in low3[::-1]]
            vals = [v for _, v in high3] + [v for _, v in low3[::-1]]
            fig_top = go.Figure(go.Bar(
                y=names, x=vals, orientation="h",
                # 红=偏干(SSI<0), 蓝=偏湿(SSI>0)
                marker_color=["#1976D2" if v > 0 else "#D32F2F" for v in vals],
                text=[f"{v:+.2f}" for v in vals], textposition="outside", textfont_size=11))
            xr = max(abs(min(vals)), abs(max(vals))) + 0.6
            fig_top.update_layout(height=250, margin=dict(l=6, r=34, t=6, b=6),
                                  yaxis=dict(tickfont=dict(size=11.5)),
                                  xaxis=dict(zeroline=True, zerolinecolor="#999",
                                             range=[-xr, xr], showticklabels=False),
                                  paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_top, use_container_width=True)

    # ---- 左侧主地图 (图例悬浮左下角) ----
    with map_col:
        with st.container(border=True):
            tag = f"{month} · {version}" if active_product == "ssi" else version
            st.markdown(f'<div class="card-title">{prod["icon"]} {prod["name"]}<span class="tag">{tag}</span></div>',
                        unsafe_allow_html=True)
            img_uri, bounds = render_product_png(version, active_product, band_idx)
            west, south, east, north = bounds
            m = folium.Map(location=[33.8, 113.5], tiles=None, zoom_control=True)
            # 纯色底图 (.leaflet-container 背景色, 见 MAP_CSS): 不请求任何外部瓦片,
            # 避免 Esri/OSM 被墙导致的 ERR_BLOCKED_BY_ORB / ERR_ABORTED
            folium.raster_layers.ImageOverlay(
                image=img_uri, bounds=[[south, west], [north, east]], opacity=0.9,
            ).add_to(m)
            folium.GeoJson(
                load_city_geojson(),
                style_function=lambda f: {"color": "#c8c8c8", "weight": 0.8, "opacity": 1.0, "fill": False},
            ).add_to(m)
            MAP_CSS = """
            <style>
              .leaflet-container { background: #eef1f5 !important; }
              .leaflet-tooltip.city-label, .leaflet-tooltip.city-label::before {
                background: transparent !important; border: none !important;
                box-shadow: none !important; padding: 0 !important;
              }
              .leaflet-control-attribution { display: none !important; }
            </style>
            """
            m.get_root().html.add_child(folium.Element(MAP_CSS))
            LABEL_STYLE = "color:#555555;font-size:10px;font-weight:500;padding:0;"
            NO_ICON = folium.DivIcon(html="", icon_size=(0, 0))
            for name, (clat, clon) in load_city_labels():
                folium.Marker(
                    location=[clat, clon],
                    tooltip=folium.Tooltip(name, permanent=True, direction="center",
                                           offset=(0, 0), style=LABEL_STYLE, class_name="city-label"),
                    icon=NO_ICON,
                ).add_to(m)
            # 图例悬浮: 地图左下角 (半透明小卡片)
            lg_b64 = base64.b64encode(render_legend_png(version, active_product)).decode()
            m.get_root().html.add_child(folium.Element(
                f'<div style="position:absolute; bottom:18px; left:18px; z-index:9999; '
                f'background:rgba(255,255,255,.9); padding:10px 12px 8px; border-radius:9px; '
                f'box-shadow:0 2px 8px rgba(0,0,0,.18); border:1px solid #e6ebf2; '
                f'min-width:240px;">'
                f'<img src="data:image/png;base64,{lg_b64}" style="width:232px; display:block;"></div>'
            ))
            city_s, city_w, city_n, city_e = load_city_bounds()
            m.fit_bounds([[city_s, city_w], [city_n, city_e]], padding=(4, 4))
            if st.session_state.get("click"):
                _cl = st.session_state.click
                folium.CircleMarker(
                    location=[_cl["lat"], _cl["lon"]], radius=7,
                    color="#000000", weight=2, fill=True,
                    fill_color="#ffd700", fill_opacity=0.35,
                ).add_to(m)
            res = st_folium(m, height=700, use_container_width=True,
                            returned_objects=["last_clicked"], key="map")

        # ---- 数据产品卡片阵列 (2行3列, 统一高度 200px) ----
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        for row_i in range(2):
            cols = st.columns(3)
            for col, p in zip(cols, PRODUCTS[row_i * 3: row_i * 3 + 3]):
                with col:
                    with st.container(height=200, border=True):
                        st.markdown(f'<div class="prod-title">{p["icon"]} {p["name"]}</div>'
                                    f'<div class="prod-sub">{p["sub"]}</div>',
                                    unsafe_allow_html=True)
                        st.markdown(f'<img class="prod-thumb" src="{render_thumb_uri(version, p["key"], band_idx)}">',
                                    unsafe_allow_html=True)
                        if p["key"] == active_product:
                            st.button("✓ 当前图层", key=f"prod_btn_{p['key']}", disabled=True,
                                      use_container_width=True)
                        elif st.button("切换到主地图", key=f"prod_btn_{p['key']}",
                                       use_container_width=True):
                            st.session_state["map_layer"] = p["key"]
                            st.rerun()

    # ---- 点击 -> 综合研判框 ----
    last = (res or {}).get("last_clicked") if isinstance(res, dict) else None
    if last and isinstance(last, dict) and "lat" in last and "lng" in last:
        st.session_state.click = {"lon": float(last["lng"]), "lat": float(last["lat"])}

    with st.container(border=True):
        if st.session_state.get("click"):
            cl = st.session_state.click
            city = find_city_name(cl["lon"], cl["lat"])
            if city is None:
                st.markdown('<div class="card-title">综合研判</div>', unsafe_allow_html=True)
                st.info("点击位置不在任何地级市界内, 请点击河南省内区域")
            else:
                ssi_v = city_mean_ssi(city, cube, transform, band_idx)
                diag = get_diagnosis(ssi_v)   # 统一诊断入口: 徽章/预警框/治理决策三处共用
                lvl, alert_type, lvl_color = diag["level"], diag["alert"], diag["color"]
                yr = int(month[:4])
                key = norm_city(city)
                ser = load_yield_series(version)
                row = ser[(ser["key"] == key) & (ser["年份"] == yr)]
                loss = float(row["减产率%"].iloc[0]) if len(row) and np.isfinite(row["减产率%"].iloc[0]) else np.nan
                anom = float(row["距平"].iloc[0]) if len(row) and np.isfinite(row["距平"].iloc[0]) else np.nan

                st.markdown(
                    f'<div class="card-title">📍 {city} · {yr} 年 {int(month[5:])} 月 综合研判'
                    f'<span class="tag">数据层级: {version}</span></div>',
                    unsafe_allow_html=True)
                d_col, j_col, c_col = st.columns([1, 1.25, 1.35], gap="medium")
                with d_col:
                    r1c1, r1c2 = st.columns(2)
                    r1c1.metric("全市平均 SSI", f"{ssi_v:+.2f}" if np.isfinite(ssi_v) else "—")
                    hover_formula("SSI 计算公式", r"SSI_{it} = \frac{X_{it} - \bar{X}_i}{\sigma_i}")
                    with r1c2:
                        st.markdown(
                            f"<div style='font-size:13px;color:#5b6572'>水分胁迫等级</div>"
                            f"<div style='margin-top:6px'><span class='lvl-badge' style='background:{lvl_color}'>{lvl}</span></div>",
                            unsafe_allow_html=True)
                    r2c1, r2c2 = st.columns(2)
                    loss_txt = f"{loss:+.2f}%" if np.isfinite(loss) else "—"
                    anom_txt = f"{anom:+.2f} 万吨" if np.isfinite(anom) else "—"
                    r2c1.markdown(
                        f"<div style='font-size:13px;color:#5b6572'>当年减产率</div>"
                        f"<div style='font-size:20px;font-weight:700;color:#1f2d3d;white-space:nowrap'>{loss_txt}</div>"
                        f"<div style='font-size:11px;color:#9aa3af'>(实际−趋势)/趋势×100，负=减产；2025 未公布</div>",
                        unsafe_allow_html=True)
                    r2c2.markdown(
                        f"<div style='font-size:13px;color:#5b6572'>当年产量距平</div>"
                        f"<div style='font-size:20px;font-weight:700;color:#1f2d3d;white-space:nowrap'>{anom_txt}</div>"
                        f"<div style='font-size:11px;color:#9aa3af'>实际产量 − 线性趋势产量</div>",
                        unsafe_allow_html=True)
                with j_col:
                    # 动态研判文字 (等级/标题/颜色统一来自 get_diagnosis, 与顶部徽章、底部决策永远一致)
                    if alert_type == "info":
                        head, body, bg, bd, fc = (diag["head"], diag["advice"],
                                                  "#f2f4f8", "#d9dee6", "#5b6572")
                    else:
                        loss_txt = f"当年减产率 {loss:+.2f}%" if np.isfinite(loss) \
                            else "当年产量数据尚未公布"
                        if alert_type == "error":
                            head = diag["head"]
                            bg, bd, fc = (("#fdecea", "#f5c6cb", "#8a1c1c") if ssi_v < 0
                                          else ("#e8f1fd", "#bcd6f5", "#0b4da2"))
                        elif alert_type == "warning":
                            head = diag["head"]
                            bg, bd, fc = (("#fdf1ea", "#f5d3bd", "#8a4c1c") if ssi_v < 0
                                          else ("#eef6fd", "#c7e0f7", "#1d5a8c"))
                        else:
                            head = diag["head"]
                            bg, bd, fc = "#e9f7ef", "#b7e4c7", "#1e6b3a"
                        body = (f"该市当期 SSI {ssi_v:+.2f}，判定为【{lvl}】（{loss_txt}）。"
                                f"{city}{diag['diag']}。{diag['advice']}")
                    st.markdown(
                        f'<div class="judge-box" style="background:{bg};border-color:{bd};color:{fc}">'
                        f'<div class="jb-head">{head}</div>{body}</div>',
                        unsafe_allow_html=True)
                with c_col:
                    # 该市 10 年 产量距平(柱, 绿增红减) × SSI(线, 右轴) 微型图
                    cs = ser[ser["key"] == key].sort_values("年份")
                    yrs = cs["年份"].tolist()
                    ssi_list = cs["SSI"].tolist()
                    anom_list = [float(a) if np.isfinite(a) else None for a in cs["距平"]]
                    bar_colors = ["#41ab5d" if (a is not None and a >= 0) else "#cb181d"
                                  for a in anom_list]
                    ssi_pt_colors = ["#9aa3af" if not np.isfinite(v)
                                     else ("#D32F2F" if v < 0 else "#1976D2")
                                     for v in ssi_list]  # 红=偏干, 蓝=偏湿
                    fig_mini = go.Figure()
                    fig_mini.add_bar(x=yrs, y=anom_list, marker_color=bar_colors,
                                     name="产量距平 (万吨)")
                    fig_mini.add_scatter(x=yrs, y=ssi_list, mode="lines+markers", yaxis="y2",
                                         line=dict(color="#6b7280", width=2),
                                         marker=dict(size=6, color=ssi_pt_colors), name="SSI")
                    fig_mini.add_hline(y=0, line=dict(color="#999", width=1))
                    fig_mini.update_layout(
                        height=250, margin=dict(l=38, r=38, t=52, b=8),
                        title=dict(text=f"{city} 2016–2025 产量距平 × SSI", font_size=12,
                                   y=0.98, yanchor="top"),
                        yaxis=dict(title=dict(text="距平 (万吨)", font=dict(size=10)),
                                   zeroline=False, tickfont_size=10),
                        yaxis2=dict(title=dict(text="SSI", font=dict(size=10)),
                                    overlaying="y", side="right", showgrid=False,
                                    tickfont_size=10),
                        legend=dict(orientation="h", y=1.02, yanchor="bottom",
                                    x=0.5, xanchor="center", font_size=10),
                        paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_mini, use_container_width=True)
                    # ---- Hurst × Sen 趋势研判 (图表下方) ----
                    _hurst = city_layer_mean(version, "hurst", city)
                    _sen = city_layer_mean(version, "sen", city)
                    if np.isfinite(_hurst) and np.isfinite(_sen):
                        if _hurst > 0.5 and _sen > 0:
                            _conclu = "未来水分持续偏湿（涝害风险加剧）"
                            _advice = "建议将排涝工程纳入未来三年储备库。"
                        elif _hurst > 0.5 and _sen < 0:
                            _conclu = "未来水分持续偏干（旱灾风险加剧）"
                            _advice = "建议升级灌溉设施。"
                        else:
                            _conclu = "未来趋势可能反转"
                            _advice = "维持现状，加强监测。"
                        st.info(f"📊 **趋势研判**（Hurst {_hurst:.2f} · Sen 斜率 {_sen:+.3f}）："
                                f"预测结论：**{_conclu}**。治理决策：{_advice}")
                    else:
                        st.info("📊 **趋势研判**：该市 Hurst / Sen 图层无有效像元, 暂无法生成趋势预测。")
                    hover_formula("Hurst 持续性公式", r"E\left[\frac{R(n)}{S(n)}\right] \sim C n^H")
                    hover_formula("Sen 斜率公式", r"\beta = \text{Median}\left\{ \frac{x_j - x_i}{j - i} \right\}")
                if not np.isfinite(loss):
                    st.caption(f"{yr} 年产量数据尚未公布, 研判仅基于 SSI 与等级")

                # ---- 系统自动生成的靶向治理决策 ----
                st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
                st.markdown('<div class="card-title">🩺 系统自动生成的靶向治理决策'
                            '<span class="tag">基于当期 SSI 诊断</span></div>',
                            unsafe_allow_html=True)
                if not np.isfinite(ssi_v):
                    st.info("该区域当前月份无有效 SSI 像元，暂无法生成治理决策建议。")
                else:
                    # 诊断与建议全部来自 get_diagnosis, 与顶部徽章、中间预警框永远一致
                    diag_txt = f"{city}{diag['diag']}。"
                    advice_txt = diag["advice"]
                    show = {"error": st.error, "warning": st.warning,
                            "success": st.success}[alert_type]
                    icon = "✅" if alert_type == "success" else "⚠️"
                    show(f"{icon} 诊断结果：{diag_txt}")
                    st.info(f"💡 **治理路径建议**：{advice_txt}")
                    # 一键导出报告 (.txt) —— 纯函数生成, 每次调用全新构造, 无累积状态
                    report = build_report_text(city, yr, month, version,
                                               ssi_v, lvl, diag_txt, advice_txt)
                    st.download_button(
                        "📥 一键导出该市耕地水分风险评估与治理建议报告",
                        data=report.encode("utf-8-sig"),
                        file_name=f"{city}_{month}_水分风险评估与治理建议.txt",
                        mime="text/plain", use_container_width=True)
        else:
            st.markdown('<div class="card-title">综合研判</div>', unsafe_allow_html=True)
            st.info("在地图上点击任意城市, 此处将生成该市「数据 + 研判文字」的综合研判框")

# ================= 侧边栏: 智能决策推演 · 情景推演 =================
with st.sidebar:
    st.markdown("### 🔮 智能决策推演 · 情景推演")
    f_ssi = st.slider("未来5月 SSI 预测值", -2.5, 2.5, 0.0, 0.1,
                      help="负值代表干旱, 正值代表洪涝 (标准化土壤水分指数 SSI)")
    eng_pct = st.slider("工程干预力度(%)", 0, 100, 0,
                        help="排涝或灌溉设施的提升比例")
    # 面板数据回归实证系数 (双端非对称): 涝端 beta1 / 旱端 beta2
    BETA1, BETA2 = -0.1889, 0.0211
    pred_anom = (BETA1 if f_ssi > 0 else BETA2) * f_ssi
    eng_anom = pred_anom * (1 - eng_pct / 100 * 0.8)   # 工程干预最多挽回 80% 损失
    m1, m2 = st.columns(2)
    m1.metric("预测单产波动 (吨/公顷)", f"{pred_anom:+.2f}")
    m2.metric("干预后单产波动 (吨/公顷)", f"{eng_anom:+.2f}",
              delta=f"{eng_anom - pred_anom:+.2f} 工程挽回")
    hover_formula("双端非对称模型", r"Y_{it} = \beta_1 SSI_{it}^{+} + \beta_2 SSI_{it}^{-} + \epsilon_{it}")
    st.caption("基于 2016-2024 面板回归的双端非对称实证系数: 涝端 β₁=-0.1889***, "
               "旱端 β₂=+0.0211 (波动为负=减产)")
    if f_ssi > 0.5:
        st.error("⚠️ **涝灾预警**：基于实证模型，涝端减产效应极强。"
                 "建议提升高标准农田排涝标准，疏通沟渠。")
    elif f_ssi < -0.5:
        st.warning("⚠️ **旱灾预警**：基于实证模型，干旱会导致轻微减产（虽不显著但需防范）。"
                   "建议完善灌溉设施保墒。")
    else:
        st.success("✅ 未来水分处于正常区间, 按常规农情管理, 保持常态化监测即可。")

    # ---- 基于历史数据的状态转移预测 ----
    st.markdown("---")
    st.markdown("### 📈 历史状态转移预测")
    _cl = st.session_state.get("click")
    _city = find_city_name(_cl["lon"], _cl["lat"]) if _cl else None
    if _city is None:
        st.info("请在主地图上点击城市后, 基于该市当前 SSI 进行状态转移预测。")
    else:
        _ssi = city_mean_ssi(_city, cube, transform, band_idx)
        _ser = load_yield_series(version)
        _row = _ser[(_ser["key"] == norm_city(_city)) & (_ser["年份"] == int(month[:4]))]
        # 当年状态: 优先用该市当年 4-5 月均值 SSI (与时序表口径一致), 缺则退回当月均值
        _cur = float(_row["SSI"].iloc[0]) if len(_row) and np.isfinite(_row["SSI"].iloc[0]) else _ssi
        if np.isfinite(_cur):
            _state, _n, _pw, _pd = transition_probs(_ser, _cur)
            if _n == 0:
                st.warning(f"当前状态「{_state}」在 2016-2024 历史序列中无转移样本, 暂无法预测。")
            else:
                st.markdown(f"当前状态：**{_state}**（{_city} {month[:4]} 年 SSI {_cur:+.2f}）"
                            f"· 历史同状态转移样本 {_n} 条")
                st.markdown(f"模型预测：该市未来转为**极端涝害**（SSI>1.5）的概率为 **{_pw*100:.0f}%**")
                st.progress(_pw, text=f"→ 极端偏湿 (涝害)")
                st.markdown(f"模型预测：该市未来转为**重度干旱**（SSI<-1.5）的概率为 **{_pd*100:.0f}%**")
                st.progress(_pd, text=f"→ 极端偏干 (旱害)")
                if _pw >= 0.25:
                    st.error("🚰 **转移概率治理决策**：涝害转移概率偏高，建议**修建/改造排水沟渠**——"
                             "按高标准农田排涝规范疏浚毛沟、斗沟, 并在低洼麦田增设暗管排水。")
                elif _pw >= 0.10:
                    st.warning("🚰 **转移概率治理决策**：存在一定涝害转移风险, 建议开展排涝沟渠"
                               "汛前排查与清淤, 预置移动排涝泵站。")
                if _pd >= 0.25:
                    st.error("💧 **转移概率治理决策**：旱害转移概率偏高，建议**加强灌溉保障**——"
                             "检修机井与渠系配套, 制定分片区轮灌方案, 保障拔节—灌浆期用水。")
                elif _pd >= 0.10:
                    st.warning("💧 **转移概率治理决策**：存在一定旱害转移风险, 建议提前调度"
                               "灌溉水源, 检修提灌设施。")
                if _pw < 0.10 and _pd < 0.10:
                    st.success("✅ **转移概率治理决策**：旱涝两端转移概率均较低, "
                               "维持常态化监测与现有水利管护即可。")
        else:
            st.info("该市当年无有效 SSI 数据, 暂无法进行状态转移预测。")

# ================= Tab 2: 旱涝属性数据库 =================
with tab_db:
    if q and (city_hit or year_hit):
        # ---- 搜索联动: 主面板按 城市 / 年份 过滤 ----
        st.markdown(f'<div class="card-title" style="font-size:18px">🔍 检索结果'
                    f'<span class="tag">关键词: {q}</span></div>',
                    unsafe_allow_html=True)
        with st.container(border=True):
            _panel = load_yield_series("浅层根区（0-28cm）")
            _mask = (_panel["城市"].astype(str).str.contains(q, regex=False, na=False)
                     | _panel["key"].astype(str).str.contains(q, regex=False, na=False)
                     | _panel["年份"].astype(str).str.contains(q, regex=False, na=False))
            _sub = _panel[_mask]
            if len(_sub):
                st.caption(f"来源: 冬小麦产量面板（浅层根区）· 城市 / 年份 含「{q}」的 "
                           f"{len(_sub)} 条记录（18 市 × 10 年面板）")
                _disp = _sub.round({"SSI": 2, "实际产量": 2, "趋势产量": 2, "距平": 2, "减产率%": 2})
                st.dataframe(_disp, use_container_width=True, hide_index=True,
                             height=min(38 + 35 * len(_sub), 560))
                st.download_button("下载检索结果 (CSV)",
                                   _sub.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
                                   file_name=f"检索结果_{q}.csv", mime="text/csv")
            else:
                st.warning("未找到匹配的数据，请检查城市名或年份。")
    elif q and not ds_hit:
        st.markdown('<div class="card-title" style="font-size:18px">🔍 检索结果</div>',
                    unsafe_allow_html=True)
        st.warning("未找到匹配的数据，请检查城市名或年份。")
    else:
        st.markdown('<div class="card-title" style="font-size:18px">核心数据集<span class="tag">点击卡片按钮查看对应数据表</span></div>',
                    unsafe_allow_html=True)
        for row_i in range(2):  # 2 行 x 3 列
            cols = st.columns(3)
            for col, ds in zip(cols, DATASETS[row_i * 3: row_i * 3 + 3]):
                with col:
                    with st.container(height=230, border=True):
                        st.markdown(
                            f'<div class="ds-title">{ds["icon"]} {ds["title"]}</div>'
                            f'<div class="ds-desc">{ds["desc"]}</div>',
                            unsafe_allow_html=True)
                        if st.button("查看数据", key=f"ds_btn_{ds['title']}", use_container_width=True):
                            st.session_state["ds_active"] = ds["title"]
                            st.session_state[f"page_{ds['title']}"] = 0

        active = st.session_state.get("ds_active")
        if active:
            ds = next(d for d in DATASETS if d["title"] == active)
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown(f'<div class="card-title">{ds["icon"]} {ds["title"]}<span class="tag">{ds["path"].name}</span></div>',
                            unsafe_allow_html=True)
                try:
                    df = pd.read_csv(ds["path"], encoding="utf-8-sig")
                except Exception as exc:
                    st.error(f"数据文件读取失败: {exc}")
                else:
                    page_size = 50
                    n_rows = len(df)
                    n_pages = max(1, (n_rows + page_size - 1) // page_size)
                    page_key = f"page_{ds['title']}"
                    page = min(st.session_state.get(page_key, 0), n_pages - 1)

                    p1, p2, p3 = st.columns([1, 1, 4])
                    if p1.button("← 上一页", key=f"prev_{ds['title']}", disabled=(page <= 0)):
                        st.session_state[page_key] = page - 1
                        st.rerun()
                    if p2.button("下一页 →", key=f"next_{ds['title']}", disabled=(page >= n_pages - 1)):
                        st.session_state[page_key] = page + 1
                        st.rerun()
                    p3.markdown(
                        f"<div style='padding-top:8px;color:#5b6572;font-size:13px'>"
                        f"第 <b>{page + 1}</b> / {n_pages} 页 · 共 {n_rows} 行 · 每页 {page_size} 行 · 表格区域内可上下滚动</div>",
                        unsafe_allow_html=True)

                    df_page = df.iloc[page * page_size: (page + 1) * page_size]
                    tbl_html = df_page.to_html(index=False, classes="portal-tbl", border=0,
                                               float_format=lambda v: f"{v:.3f}")
                    st.markdown(f'<div class="tbl-wrap">{tbl_html}</div>', unsafe_allow_html=True)
                    st.download_button("下载完整数据 (CSV)",
                                       df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
                                       file_name=ds["path"].name, mime="text/csv")
        else:
            st.info("👆 点击上方任意数据集卡片的「查看数据」, 此处将展示对应数据表 (每页 50 行, 可翻页)")

# ---- 页脚 ----
st.markdown(
    "<div style='text-align:center;color:#9aa3af;font-size:12px;margin:26px 0 10px'>"
    "河南冬小麦主产区耕地水分风险研判与数字化治理决策系统 · 数据门户 (2016–2025, 4–5 月 SSI)</div>",
    unsafe_allow_html=True,
)
