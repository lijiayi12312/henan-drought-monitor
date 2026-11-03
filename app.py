# -*- coding: utf-8 -*-
"""
中原农情旱涝双端风险研判与空间态势感知系统 —— 网格卡片式数据门户 (浅色主题)
第 5 阶段: 三模块主导航重构 (st.tabs)
  Tab1 空间态势感知 - 全局横向控制栏(城市/年份/图层) + 全宽主地图 + 数据产品卡片 + 全省概览/靶区 Top3
  Tab2 智能研判与决策推演 - 左列[综合研判与靶向处方] 右列[未来气候情景沙盘](推演滑块/双端非对称/状态转移预测)
  Tab3 旱涝属性数据库 - 核心数据集卡片 + HTML 淡蓝表头表格 + 翻页 + 下载
运行:  $env:PYTHONPATH="C:\pylibs"; python -m streamlit run app.py
(原 app_portal.py 已合并为 app.py; 旧版单页平台备份在 app_old_backup.py)
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
import streamlit.components.v1 as components

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

# Tab3 核心数据集卡片 (2行3列)
DATASETS = [
    {"icon": "🌾", "title": "冬小麦产量面板（浅层根区）",
     "desc": "18 市 × 10 年 4-5 月浅层根区 SSI × 实际单产 / 趋势单产 / 单产距平 / 减产率，主分析数据集。",
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

# Tab3 三层结构: Tabs分类 + 卡片式入口 + 动态表格
_TAB3_BASE = ROOT / "data" / "河南冬小麦全育期土壤水分干旱窗口与产量关联的统计检验" / "Henan_Wheat_Drought_Project"

# 主分析面板: 产量 / SSI 面板与产量关系统计
TAB3_MAIN = [
    {"key": "yield_shallow", "icon": "🌾", "title": "冬小麦产量面板（浅层）",
     "desc": "年份, 城市, SSI, 实际产量, 趋势产量, 距平, 减产率%",
     "path": ROOT / "output" / "产量关联分析" / "city_yield_ssi_timeseries_shallow.csv"},
    {"key": "yield_surface", "icon": "🌱", "title": "冬小麦产量面板（表层）",
     "desc": "年份, 城市, SSI, 实际产量, 趋势产量, 距平, 减产率%",
     "path": ROOT / "output" / "产量关联分析" / "city_yield_ssi_timeseries.csv"},
    {"key": "full_ssi", "icon": "📊", "title": "全育期SSI面板",
     "desc": "年份, 城市, SSI, 实际产量, 趋势产量, 距平",
     "path": _TAB3_BASE / "output" / "final_panel_wheat_2016_2024.csv"},
    {"key": "raw_yield", "icon": "🌾", "title": "全育期产量原始数据",
     "desc": "城市, 年份, 产量, 距平",
     "path": _TAB3_BASE / "data" / "henan_wheat_yield_clean.csv"},
    {"key": "corr_stats", "icon": "📊", "title": "各市SSI-产量关系统计",
     "desc": "城市, Pearson r, p值, 减产率汇总",
     "path": ROOT / "output" / "产量关联分析" / "city_yield_analysis_shallow.csv"},
    {"key": "layer_compare", "icon": "📈", "title": "五层SSI × 产量相关对比表",
     "desc": "层级, 相关系数, 显著性",
     "path": ROOT / "output" / "产量关联分析" / "层级产量关联对比表.csv"},
]

# 气象与遥感特征: 旱涝事件档案 / 植被指数 / 辐射
TAB3_REMOTE = [
    {"key": "run_theory", "icon": "🌀", "title": "旱涝事件回溯库（游程理论）",
     "desc": "起止月份, 持续月数, 累积强度",
     "path": ROOT / "output" / "游程理论" / "河南浅层SSI_4-5月_干旱事件_按真实时间_2016-2025.csv"},
    {"key": "ndvi", "icon": "🛰️", "title": "NDVI 植被指数",
     "desc": "城市, 年份, 月份, NDVI均值（遥感特征）",
     "path": _TAB3_BASE / "output" / "city_ndvi_4_5_2016_2024.csv"},
    {"key": "evi", "icon": "🌿", "title": "EVI 增强植被指数",
     "desc": "城市, 年份, 月份, EVI均值（遥感特征）",
     "path": _TAB3_BASE / "output" / "重要数据集" / "原始数据" / "city_evi_4_5_2016_2024.csv"},
    {"key": "solar", "icon": "☀️", "title": "太阳辐射对比",
     "desc": "城市, 2021/2023太阳辐射, 变化率",
     "path": _TAB3_BASE / "output" / "city_solar_comparison.csv"},
]

# 工程与设施数据: 灌溉工程 / 播种面积
TAB3_ENG = [
    {"key": "irrigation", "icon": "💧", "title": "灌溉与工程防御边界",
     "desc": "城市, 年份, 灌溉面积, 有效灌溉面积, 有效灌溉占比",
     "path": _TAB3_BASE / "data" / "河南省分市数据(2016-2024)" / "河南省各市灌溉面积，有效（耕地）灌溉面积与占比.xlsx"},
    {"key": "sowing_area", "icon": "🚜", "title": "各市冬小麦播种面积",
     "desc": "城市, 2016-2024 年播种面积（千公顷）",
     "path": _TAB3_BASE / "data" / "河南省分市数据(2016-2024)" / "河南省各市冬小麦播种面积.xlsx"},
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
def load_irrigation(path_str):
    """清洗各市灌溉面积 xlsx: 每年 3 列(灌溉面积/有效灌溉面积/百分比), 转长表"""
    raw = pd.read_excel(path_str, header=None)
    years = []
    for c in range(1, raw.shape[1], 3):
        v = raw.iat[1, c]
        if pd.isna(v):
            break
        years.append(str(v).replace("年", ""))
    records = []
    for r in range(5, raw.shape[0]):
        city = raw.iat[r, 0]
        if pd.isna(city):
            continue
        city = pd.Series([city]).astype(str).str.replace(r"\s+", "", regex=True).iloc[0]
        for i, y in enumerate(years):
            base = 1 + i * 3
            pct = pd.to_numeric(raw.iat[r, base + 2], errors="coerce")
            records.append({
                "城市": city, "年份": int(y),
                "灌溉面积(千公顷)": pd.to_numeric(raw.iat[r, base], errors="coerce"),
                "有效灌溉面积(千公顷)": pd.to_numeric(raw.iat[r, base + 1], errors="coerce"),
                "有效灌溉占比(%)": round(pct * 100, 2) if pd.notna(pct) else np.nan,
            })
    return pd.DataFrame(records)


@st.cache_data
def load_sowing_area(path_str):
    """清洗各市冬小麦播种面积宽表: 行=城市, 列=2016-2024 年份"""
    raw = pd.read_excel(path_str, header=None)
    years = [str(v).replace("年", "") for v in raw.iloc[1, 1:].tolist()]
    df = raw.iloc[2:].copy()
    df.columns = ["城市"] + years
    df = df.dropna(subset=["城市"])
    df["城市"] = df["城市"].astype(str).str.replace(r"\s+", "", regex=True)
    df = df[~df["城市"].isin(["全省", "各市之和"])]
    return df.reset_index(drop=True)


@st.cache_data
def load_tab3_table(key):
    """按 key 加载 Tab3 数据表, 返回 DataFrame 或 None"""
    for ds in TAB3_MAIN + TAB3_REMOTE + TAB3_ENG:
        if ds["key"] == key:
            if ds["key"] == "sowing_area":
                return load_sowing_area(str(ds["path"]))
            if ds["key"] == "irrigation":
                return load_irrigation(str(ds["path"]))
            return pd.read_csv(ds["path"], encoding="utf-8-sig")
    return None


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


SSI_STATE_BINS = [(-99, -1.5, "重度偏干（旱灾）"), (-1.5, -0.5, "轻度偏干"),
                  (-0.5, 0.5, "正常波动"), (0.5, 1.5, "轻度偏湿"),
                  (1.5, 99, "极强偏湿（涝灾）")]


def ssi_state_bin(s):
    """SSI -> 离散状态名 (五档, 与 get_diagnosis 阈值一致, 供状态转移预测的转移样本分组)"""
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


def _fig_to_uri(rgba, cell_px=30, dpi=100):
    from scipy.ndimage import gaussian_filter
    # 云图晕染: 预乘 alpha 高斯模糊, 柔化像元硬边界、消除马赛克感 (类似气象云图)。
    # 预乘 alpha 可避免透明区(nodata/省界外)的黑色在模糊时污染边缘颜色, 边缘自然过渡。
    sigma = 1.5
    alpha = rgba[..., 3:4]
    premult = rgba[..., :3] * alpha
    rgb_blur = np.stack([gaussian_filter(premult[..., c], sigma) for c in range(3)], axis=-1)
    a_blur = gaussian_filter(alpha[..., 0], sigma)[..., None]
    rgb_blur = np.clip(rgb_blur / np.maximum(a_blur, 1e-9), 0, 1)
    rgba = np.concatenate([rgb_blur, a_blur], axis=-1)
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

.stTabs [data-baseweb="tab-list"] { gap:36px; background:transparent; margin-top:6px; }
.stTabs [data-baseweb="tab"] {
  background:#FFFFFF; border-radius:10px 10px 0 0; padding:10px 30px;
  border:1px solid #eef1f6; border-bottom:none; font-weight:600; color:#3a4552;
}
.stTabs [aria-selected="true"] { color:var(--portal-blue); border-top:3px solid var(--portal-blue); }

[data-testid="stVerticalBlockBorderWrapper"] {
  background:#FFFFFF; border-radius:12px !important; border:1px solid #eef1f6 !important;
  box-shadow:0 4px 6px rgba(0,0,0,.05); padding:10px 14px;
}
.card-title { font-size:16px; font-weight:700; color:#1a2433; margin:2px 0 10px; }
.card-title .tag { font-size:12px; font-weight:500; color:#8a94a6; margin-left:8px; }

/* 产品卡片 (自适应高度容器内部) */
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

/* 横向控制栏 */
.ctrl-bar { background:#FFFFFF; border:1px solid #eef1f6; border-radius:12px;
  padding:10px 16px 6px; box-shadow:0 4px 10px rgba(16,42,80,.06); margin-bottom:12px; }
.ctrl-bar .cb-label { font-size:12.5px; color:#5b6572; font-weight:600; margin-bottom:2px; }
.ctrl-bar .cb-val { font-size:14px; color:#1a2433; font-weight:700; }

</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---- 全局城市/年份/图层 初始化 (在标签页之前定义, 供 tab2/tab3 直接引用) ----
city_names = sorted(load_city_gdf()["name"].astype(str).tolist())
YEARS = list(range(2016, 2026))
if "current_city" not in st.session_state or st.session_state["current_city"] not in city_names:
    st.session_state["current_city"] = city_names[0]
if "current_year" not in st.session_state:
    st.session_state["current_year"] = 2025
if "current_month" not in st.session_state:
    st.session_state["current_month"] = 4
if "map_layer" not in st.session_state:
    st.session_state["map_layer"] = "ssi"
if "current_table_name" not in st.session_state:
    st.session_state["current_table_name"] = None
# 主分析数据层级固定为浅层根区 (0-28cm, 与冬小麦产量相关性最强)
version = "浅层根区（0-28cm）"

# ---- 处理图层卡片点击回传的延迟写入 (在控件实例化前写入, 规避 StreamlitWidgetAlreadyInstantiatedError) ----
# Streamlit 规定: 控件一旦实例化(带 key), 之后不能再直接写该 key。故图层卡片点击
# 先写入独立的"请求键", 再 st.rerun(); 本轮在控件创建前把它回填到 map_layer。
_request_layer = st.session_state.pop("_map_layer_request", None)
if _request_layer and _request_layer in [p["key"] for p in PRODUCTS]:
    st.session_state["map_layer"] = _request_layer


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

# ---- 全局搜索栏 (真实检索, 悬浮于 Banner 底部; 城市/年份联动注入到全局选择) ----
if st.session_state.get("_pending_clear"):
    st.session_state["global_search"] = ""
    st.session_state["_pending_clear"] = False

_sL, _sC, _sR = st.columns([1, 2.4, 1])
with _sC:
    search_query = st.text_input(
        "全局检索", "", key="global_search", label_visibility="collapsed",
        placeholder="🔍 输入城市（如：商丘）、年份（如：2023）或数据集名称进行检索...")
    q = (search_query or "").strip()

    # 解析关键词: 年份 -> 全局年份; 城市 -> 全局城市; 数据集名 -> 自动打开
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
            st.session_state["current_year"] = int(year_hit)   # 全局年份跳至该年
            st.session_state["_sy_applied"] = year_hit
    else:
        st.session_state["_sy_applied"] = ""

    if city_hit:
        if st.session_state.get("_sc_applied") != city_hit:
            st.session_state["current_city"] = city_hit          # 全局城市定位
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
            _parts.append(f"📍 已选中 <b>{city_hit}</b> · 请在「智能研判与决策推演」标签查看研判")
        if year_hit:
            _parts.append(f"⏳ 已切换至 <b>{year_hit} 年</b>")
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

# ---- 统计卡片 (数字由真实数据动态计算) ----
_panel_main = load_yield_series(version)
STAT_CARDS = [
    ("📍", str(len(city_names)), "个", "地市覆盖"),
    ("🗄️", str(len(_panel_main)), "条", "面板数据 (市-年)"),
    ("⏳", str(len(YEARS)), "年", "长时序记录 (2016–2025)"),
    ("🗺️", str(len(PRODUCTS)), "类", "空间数据产品"),
]
for col, (icon, num, unit, label) in zip(st.columns(4), STAT_CARDS):
    col.markdown(
        f'<div class="stat-card"><div class="stat-icon">{icon}</div>'
        f'<div class="stat-num">{num}<small>{unit}</small></div>'
        f'<div class="stat-label">{label}</div></div>',
        unsafe_allow_html=True,
    )
st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

# ================= 全局控制栏 (主界面 Tab 列表上方) =================
with st.container(border=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        selected_city = st.selectbox("📍 选择城市", city_names, key="current_city")
    with c2:
        selected_year = st.slider("⏳ 选择年份", 2016, 2025, key="current_year",
                                  help="拖动选择 2016–2025 年")
    with c3:
        selected_month = st.radio("🗓️ 选择月份", [4, 5], key="current_month",
                                  format_func=lambda m: f"{m}月", horizontal=True)

# tab 内部既有代码延用 current_* 命名, 此处以别名桥接; 图层切换仍由 Tab1 卡片按钮驱动
current_city, current_year = selected_city, selected_year
current_month = selected_month
layer_key = st.session_state["map_layer"]

# ================= 主导航: 三个大模块 =================
tab1, tab2, tab3 = st.tabs(["📍 空间态势感知", "🧠 智能研判与决策推演", "🗄️ 旱涝属性数据库"])

# ================= Tab 1: 空间态势感知 =================
with tab1:
    # ---- 派生变量 (侧边栏已接管城市/年份/月份/图层) ----
    month = f"{current_year}-{current_month:02d}"
    band_idx = MONTHS.index(month)
    cube, _, transform = load_ssi(version)
    active_product = layer_key
    prod = next(p for p in PRODUCTS if p["key"] == active_product)
    current_ssi = city_mean_ssi(current_city, cube, transform, band_idx)
    diag = get_diagnosis(current_ssi)

    # ---- 主屏幕左右分栏: 左=主地图(宽 1.8), 右=概览饼图 + 靶区 Top3(窄 1) ----
    col_left, col_right = st.columns([1.8, 1], gap="medium")

    # ---------- 左列: 河南省主地图 ----------
    with col_left:
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
            # 图例悬浮: 地图左下角 (半透明小卡片) -- 保留色带图例
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
            # 静态渲染 folium 地图: 走 Streamlit 内置组件通道 (components.html),
            # 不加载 streamlit-folium 的前端 bundle, 彻底规避 Cloud 上的间歇性组件加载失败
            components.html(m.get_root().render(), height=720, scrolling=False)

    # ---------- 右列: 全省旱涝面积概览 + 重点治理靶区 Top3 ----------
    with col_right:
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
                values=[n_dry, n_norm, n_wet], hole=0.62,
                marker=dict(colors=["#D32F2F", "#c9d1d9", "#1976D2"]),  # 红=偏干, 蓝=偏湿
                textinfo="percent", textposition="inside", textfont_size=11, sort=False))
            fig_pie.update_layout(height=250, margin=dict(l=6, r=6, t=10, b=6),
                                  showlegend=True,
                                  legend=dict(orientation="h", yanchor="bottom", y=-0.10,
                                              xanchor="center", x=0.5, font_size=10),
                                  paper_bgcolor="rgba(0,0,0,0)",
                                  annotations=[dict(text=month, x=0.5, y=0.5, font_size=12,
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
            fig_top.update_layout(height=230, margin=dict(l=6, r=40, t=4, b=4),
                                  yaxis=dict(tickfont=dict(size=11), automargin=True),
                                  xaxis=dict(zeroline=True, zerolinecolor="#999",
                                             range=[-xr, xr], showticklabels=False),
                                  paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_top, use_container_width=True)

    # ---- 下半部分: 6 个图层卡片 (2行3列, 缩略图 + 切换按钮, 点击切换主地图图层) ----
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    for row_i in range(2):
        cols = st.columns(3)
        for col, p in zip(cols, PRODUCTS[row_i * 3: row_i * 3 + 3]):
            with col:
                with st.container(border=True):
                    _help = ""
                    if p["key"] == "hurst":
                        _help = "Hurst指数公式: E[R(n)/S(n)] ∝ C · n^H (n→∞)\n当 H > 0.5 时趋势持续，H < 0.5 时趋势可能反转。"
                    elif p["key"] in ("trend_dry", "trend_wet"):
                        _help = "Sen斜率公式: β = Median [ (xⱼ - xᵢ) / (j - i) ]  (1 ≤ i < j ≤ n)\n用于量化水分随时间变化的速率。"
                    st.markdown(f'<div class="prod-title">{p["icon"]} {p["name"]}</div>'
                                f'<div class="prod-sub">{p["sub"]}</div>',
                                unsafe_allow_html=True, help=_help or None)
                    st.markdown(f'<img class="prod-thumb" src="{render_thumb_uri(version, p["key"], band_idx)}">',
                                unsafe_allow_html=True)
                    if p["key"] == active_product:
                        st.button("✓ 当前图层", key=f"prod_btn_{p['key']}", disabled=True,
                                  use_container_width=True)
                    elif st.button("切换到主地图", key=f"prod_btn_{p['key']}",
                                   use_container_width=True):
                        st.session_state["_map_layer_request"] = p["key"]
                        st.rerun()

# ================= Tab 2: 智能研判与决策推演 =================
with tab2:
    # 复用侧边栏 + Tab1 定义的外部变量: current_city / current_year / current_month / current_ssi / diag / cube / transform / band_idx / version
    key = norm_city(current_city)
    ser = load_yield_series(version)
    row = ser[(ser["key"] == key) & (ser["年份"] == current_year)]
    loss = float(row["减产率%"].iloc[0]) if len(row) and np.isfinite(row["减产率%"].iloc[0]) else np.nan
    lvl, alert_type = diag["level"], diag["alert"]
    yr = current_year

    L, _gap, R = st.columns([1.2, 0.06, 1], gap="medium")

    # ========== 左列【综合研判与靶向处方】 ==========
    with L:
        st.markdown('<div class="card-title">🧠 综合研判与靶向处方'
                    f'<span class="tag">{current_city} · {yr} 年 · {version}</span></div>',
                    unsafe_allow_html=True)

        # 顶层：核心指标概览（三列横排，SSI 公式用 help 显示）
        t1, t2, t3 = st.columns(3)
        with t1:
            st.metric("全市平均 SSI", f"{current_ssi:+.2f}" if np.isfinite(current_ssi) else "—",
                      help="SSI指数公式: SSI = (X_it - X̄_i) / σ_i\n其中 X_it 为当月土壤水分，X̄_i 为基准期均值，σ_i 为标准差。")
        with t2:
            st.markdown(
                f"<div style='font-size:13px;color:#5b6572'>水分胁迫等级</div>"
                f"<div style='margin-top:6px'><span class='lvl-badge' style='background:{diag['color']}'>{lvl}</span></div>",
                unsafe_allow_html=True)
        t3.metric("当年减产率", f"{loss:+.2f}%" if np.isfinite(loss) else "待公布")

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # 双轴折线图: 该市 10 年 单产距平(柱) × SSI(线, 右轴)
        cs = ser[ser["key"] == key].sort_values("年份")
        yrs = cs["年份"].tolist()
        ssi_list = cs["SSI"].tolist()
        anom_list = [float(a) if np.isfinite(a) else None for a in cs["单产距平"]]
        bar_colors = ["#41ab5d" if (a is not None and a >= 0) else "#cb181d"
                      for a in anom_list]
        ssi_pt_colors = ["#9aa3af" if not np.isfinite(v)
                         else ("#D32F2F" if v < 0 else "#1976D2")
                         for v in ssi_list]  # 红=偏干, 蓝=偏湿
        fig_mini = go.Figure()
        fig_mini.add_bar(x=yrs, y=anom_list, marker_color=bar_colors,
                         name="单产距平 (吨/公顷)")
        fig_mini.add_scatter(x=yrs, y=ssi_list, mode="lines+markers", yaxis="y2",
                             line=dict(color="#6b7280", width=2),
                             marker=dict(size=6, color=ssi_pt_colors), name="SSI")
        fig_mini.add_hline(y=0, line=dict(color="#999", width=1))
        fig_mini.update_layout(
            height=280, margin=dict(l=38, r=38, t=52, b=8),
            title=dict(text=f"{current_city} 2016–2025 单产距平 × SSI", font_size=12,
                       y=0.98, yanchor="top"),
            yaxis=dict(title=dict(text="距平 (吨/公顷)", font=dict(size=10)),
                       zeroline=False, tickfont_size=10),
            yaxis2=dict(title=dict(text="SSI", font=dict(size=10)),
                        overlaying="y", side="right", showgrid=False,
                        tickfont_size=10),
            legend=dict(orientation="h", y=1.02, yanchor="bottom",
                        x=0.5, xanchor="center", font_size=10),
            paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_mini, use_container_width=True)

        # 偏干/湿预警框（等级/标题/颜色统一来自 get_diagnosis）
        _loss_txt = f"当年减产率 {loss:+.2f}%" if np.isfinite(loss) else "当年产量数据尚未公布"
        if alert_type == "info":
            st.info(f"{diag['head']}：该区域当期 SSI 无有效像元。")
        elif alert_type == "error":
            st.error(f"{diag['head']}：{current_city}当期 SSI {current_ssi:+.2f}，"
                     f"{diag['diag']}（{_loss_txt}）。")
        elif alert_type == "success":
            st.success(f"{diag['head']}：{current_city}当期 SSI {current_ssi:+.2f}，"
                       f"{diag['diag']}（{_loss_txt}）。")
        else:
            st.warning(f"{diag['head']}：{current_city}当期 SSI {current_ssi:+.2f}，"
                       f"{diag['diag']}（{_loss_txt}）。")

        # Hurst × Sen 趋势研判（公式问号已移至地图图层控制区，此处不重复显示）
        _hurst = city_layer_mean(version, "hurst", current_city)
        _sen = city_layer_mean(version, "sen", current_city)
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
            st.info(f"📈 **趋势研判**（Hurst {_hurst:.2f} · Sen 斜率 {_sen:+.3f}）："
                    f"预测结论：**{_conclu}**。治理决策：{_advice}")
        else:
            st.info("📈 **趋势研判**：该市 Hurst / Sen 图层无有效像元, 暂无法生成趋势预测。")

        # 靶向治理决策卡片（诊断与建议全部来自 get_diagnosis）
        st.markdown('<div class="card-title">🎯 系统自动生成的靶向治理决策'
                    '<span class="tag">基于当期 SSI 诊断</span></div>',
                    unsafe_allow_html=True)
        if not np.isfinite(current_ssi):
            st.info("该区域当前月份无有效 SSI 像元，暂无法生成治理决策建议。")
        else:
            diag_txt = f"{current_city}{diag['diag']}。"
            advice_txt = diag["advice"]
            show = {"error": st.error, "warning": st.warning,
                    "success": st.success}[alert_type]
            icon = "✅" if alert_type == "success" else "⚠️"
            show(f"{icon} 诊断结果：{diag_txt}")
            st.info(f"💡 **治理路径建议**：{advice_txt}")
            # 一键导出报告 (.txt) —— 纯函数生成, 每次调用全新构造, 无累积状态
            report = build_report_text(current_city, yr, month, version,
                                       current_ssi, lvl, diag_txt, advice_txt)
            st.download_button(
                "📥 一键导出该市耕地水分风险评估与治理建议报告",
                data=report.encode("utf-8-sig"),
                file_name=f"{current_city}_{month}_水分风险评估与治理建议.txt",
                mime="text/plain", use_container_width=True)

    # ========== 右列【未来气候情景沙盘】 ==========
    with R:
        st.markdown('<div class="card-title">🔮 未来气候情景沙盘'
                    '<span class="tag">独立推演 · 不影响实地现状</span></div>',
                    unsafe_allow_html=True)
        st.caption("⚠️ 独立推演工具：以下滑块仅用于情景模拟，不影响上方主地图的实地现状与研判结果。")

        with st.container(border=True):
            st.markdown("**情景推演滑块**")
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

            st.markdown("**双端非对称模型**",
                        help="双端回归模型公式: Y_it = β₁ · SSI_it⁺ + β₂ · SSI_it⁻ + ε_it\n涝端 β₁<0、旱端 β₂>0 的非对称实证设定。")
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

        # ---- 基于历史数据的状态转移预测 (使用当前所选城市 + 当前 SSI) ----
        st.markdown("---")
        st.markdown("### 📈 历史状态转移预测")
        if not np.isfinite(current_ssi):
            st.info("该市当年无有效 SSI 数据, 暂无法进行状态转移预测。")
        else:
            _state, _n, _pw, _pd = transition_probs(ser, current_ssi)
            if _n == 0:
                st.warning(f"当前状态「{_state}」在 2016-2024 历史序列中无转移样本, 暂无法预测。")
            else:
                st.markdown(f"当前状态：**{_state}**（{current_city} {current_year} 年 SSI {current_ssi:+.2f}）"
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

# ================= Tab 3: 旱涝属性数据库 (三层结构) =================
with tab3:
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
                _disp = _sub.round({"SSI": 2, "实际单产": 2, "趋势单产": 2, "单产距平": 2, "减产率%": 2})
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
        # 第一层: 三大数据维度分区 (横向分隔符隔开, 全部同时展现)
        # 第二层: 卡片式入口; 第三层: 动态表格 (选中哪个分类的卡片, 表格显示在该分类区块下方)
        def _render_cards(datasets):
            """渲染卡片区域 (第二层)"""
            for i in range(0, len(datasets), 3):
                cols = st.columns(3)
                for col, ds in zip(cols, datasets[i:i+3]):
                    with col:
                        with st.container(border=True):
                            st.markdown(
                                f'<div class="ds-title">{ds["icon"]} {ds["title"]}</div>'
                                f'<div class="ds-desc">{ds["desc"]}</div>',
                                unsafe_allow_html=True)
                            if st.button("查看数据", key=f"tab3_btn_{ds['key']}", use_container_width=True):
                                st.session_state["current_table_name"] = ds["key"]
                                st.rerun()

        def _render_table_area(datasets):
            """渲染表格展示区域 (第三层): 仅当选中的表属于本分类时显示"""
            current = st.session_state.get("current_table_name")
            if current not in [ds["key"] for ds in datasets]:
                return
            df = load_tab3_table(current)
            if df is None:
                st.error("数据加载失败")
                return
            ds = next(d for d in datasets if d["key"] == current)
            with st.container(border=True):
                fname = ds["path"].name
                st.markdown(f'<div class="card-title">{ds["icon"]} {ds["title"]}<span class="tag">{fname}</span></div>',
                            unsafe_allow_html=True)
                st.dataframe(df, use_container_width=True, hide_index=True, height=400)
                st.download_button("下载完整数据 (CSV)",
                                   df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
                                   file_name=f"{current}.csv",
                                   mime="text/csv", key=f"tab3_dl_{current}")

        if not st.session_state.get("current_table_name"):
            st.info("👆 点击下方任意数据集卡片的「查看数据」, 对应数据表将展示在该分类区块下方")

        # 分区 1: 主分析面板
        st.markdown('<div class="card-title" style="font-size:18px">📋 主分析面板</div>',
                    unsafe_allow_html=True)
        _render_cards(TAB3_MAIN)
        _render_table_area(TAB3_MAIN)
        st.divider()

        # 分区 2: 气象与遥感特征
        st.markdown('<div class="card-title" style="font-size:18px">🛰️ 气象与遥感特征</div>',
                    unsafe_allow_html=True)
        _render_cards(TAB3_REMOTE)
        _render_table_area(TAB3_REMOTE)
        st.divider()

        # 分区 3: 工程与设施数据
        st.markdown('<div class="card-title" style="font-size:18px">🚜 工程与设施数据</div>',
                    unsafe_allow_html=True)
        _render_cards(TAB3_ENG)
        _render_table_area(TAB3_ENG)

# ---- 页脚 ----
st.markdown(
    "<div style='text-align:center;color:#9aa3af;font-size:12px;margin:26px 0 10px'>"
    "河南冬小麦主产区耕地水分风险研判与数字化治理决策系统 · 数据门户 (2016–2025, 4–5 月 SSI)</div>",
    unsafe_allow_html=True,
)

