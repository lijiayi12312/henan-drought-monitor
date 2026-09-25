# -*- coding: utf-8 -*-
"""
河南土壤水分干旱监测平台 (Streamlit)
数据: 河南省 4-5 月 SSI (2016-04 ~ 2025-05, 每年 4/5 月共 20 个波段)
功能:
  侧边栏  - 月份选择 (20 个可用月份) / 图层选择 (SSI, 干旱频率, MK Z值, Sen斜率, Hurst指数, 未来高风险预警)
  主区域  - folium 地图 (streamlit-folium, 点击交互), 当前月份干旱面积占比
  下方    - 全省平均 SSI 时间序列折线图; 点击地图显示该像元 20 个月 SSI 折线 (plotly)
运行:  python -m streamlit run app.py  (本地需设 PYTHONPATH=C:\\pylibs; 云端由 requirements.txt 提供依赖)
依赖:  见 requirements.txt
"""
import base64
import io
import sys
from pathlib import Path

import numpy as np
import rasterio

# 本机手动装在 C:\pylibs 的第三方库 (Windows); 云端/Linux 由 requirements.txt 提供, 自动跳过
_pylibs = Path(r"C:\pylibs")
if _pylibs.exists():
    sys.path.append(str(_pylibs))

import folium
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

import matplotlib
matplotlib.use("Agg")
from matplotlib import cm as mpl_cm
from matplotlib import pyplot as plt
from matplotlib import colors as mpl_colors

# matplotlib 默认 DejaVu Sans 无中文字形 (图例中文会渲染成方框)
# 优先注册随仓库分发的 Noto Sans CJK (Windows/Linux/云端通用), 保证部署后中文一致显示
import matplotlib.font_manager as fm

_FONT_FILE = Path(__file__).resolve().parent / "assets" / "NotoSansCJKsc-Regular.otf"
if _FONT_FILE.exists():
    fm.fontManager.addfont(str(_FONT_FILE))
    _CJK = fm.FontProperties(fname=str(_FONT_FILE)).get_name()
    matplotlib.rcParams["font.sans-serif"] = [_CJK, "Microsoft YaHei", "SimHei", "DejaVu Sans"]
else:
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

# ---------------- 数据路径 ----------------
# app.py 所在目录, 本地 (Windows) 与 Streamlit Cloud (Linux) 通用
ROOT = Path(__file__).resolve().parent
SSI_PATH = ROOT / "4-5月河南SSI" / "4-5月河南SSI_2016-2025.tif"
LAYER_PATHS = {
    "干旱频率": ROOT / "干旱频率" / "河南SSI_4-5月_干旱频率_2016-2025.tif",
    "MK Z值": ROOT / "Mann-Kendall 趋势检验" / "河南SSI_4-5月_MK_Z值_2016-2025.tif",
    "Sen斜率": ROOT / "Mann-Kendall 趋势检验" / "河南SSI_4-5月_MK_Sen斜率_2016-2025.tif",
    "Hurst指数": ROOT / "hurst指数" / "河南SSI_4-5月_Hurst指数_2016-2025.tif",
}
MONTHS = [f"{y}-{m:02d}" for y in range(2016, 2026) for m in (4, 5)]  # 20 个波段
CITY_SHP = ROOT / "河南省市shp" / "河南省.shp"
EVENTS_CSV = ROOT / "游程理论" / "河南SSI_4-5月_干旱事件_按真实时间_2016-2025.csv"


# ---------------- 数据加载 (缓存) ----------------
@st.cache_data
def load_ssi():
    with rasterio.open(SSI_PATH) as src:
        cube = src.read().astype(np.float64)
        nodata = src.nodata
        cube[cube <= nodata] = np.nan
        transform = src.transform
        h, w = cube.shape[1], cube.shape[2]
    bounds = [transform.c, transform.f + h * transform.e,
              transform.c + w * transform.a, transform.f]  # [w, s, e, n]
    return cube, bounds, transform


@st.cache_data
def load_layer(path_str):
    with rasterio.open(path_str) as src:
        arr = src.read(1).astype(np.float64)
        nodata = src.nodata
        arr[arr <= nodata / 2] = np.nan
    return arr


@st.cache_data
def load_high_risk():
    """未来高风险预警像元:
    判据: Sen斜率 <= 全省最负 20% 分位 (变干最快的前两成像元) 且 Hurst > 0.5 (趋势持续)
    选型说明: 更严的三条件判据 (干旱频率>0.6 & Sen<-0.05 & Hurst>0.6) 仅命中 16 像元
    (6.2%), 低于预警图 10%~25% 的合理占比; 本判据命中 46 像元 (17.8%), 在目标区间内。
    返回 (风险掩膜, 有效掩膜, 判据说明文字)"""
    s = load_layer(str(LAYER_PATHS["Sen斜率"]))
    h = load_layer(str(LAYER_PATHS["Hurst指数"]))
    valid = np.isfinite(s) & np.isfinite(h)
    q20 = float(np.quantile(s[valid], 0.2))  # Sen 斜率最负的前 20% 分位
    risk = (s <= q20) & (h > 0.5)
    return risk, valid, f"Sen斜率 ≤ 最负20%分位 ({q20:.3f}) 且 Hurst > 0.5"


@st.cache_data
def load_high_risk_geojson():
    """高风险掩膜 -> 矢量多边形 GeoJSON:
    rasterio.features.shapes 矢量化 -> buffer(±0.6像元) 合并近邻 ->
    simplify 去重 -> Chaikin 圆角细分使边界曲线化 -> 过滤面积 < 3 像元的碎斑。
    返回 GeoJSON dict (EPSG:4326)"""
    import json as _json

    import geopandas as gpd
    from rasterio.features import shapes
    from shapely.geometry import MultiPolygon as _MP
    from shapely.geometry import Polygon as _Poly
    from shapely.geometry import shape as _shape

    risk_mask, _, _ = load_high_risk()
    with rasterio.open(SSI_PATH) as _src:
        transform = _src.transform  # 与 SSI 同网格, 用于像元->经纬度
    pw = abs(transform.a)  # 像元边长 (度)
    geoms = [_shape(g) for g, val in shapes(risk_mask.astype(np.uint8), transform=transform)
             if val == 1]

    def chaikin_ring(coords, n=3):
        """Chaikin 切角细分: 每轮把每段在 1/4 与 3/4 处取两点, 使折线变为平滑曲线"""
        pts = np.asarray(coords)
        for _ in range(n):
            new = [pts[0]]
            for i in range(len(pts) - 1):
                p, q = pts[i], pts[i + 1]
                new.append(0.75 * p + 0.25 * q)
                new.append(0.25 * p + 0.75 * q)
            new.append(pts[-1])
            pts = np.asarray(new)
        return pts

    def chaikin_poly(poly, n=3):
        ext = chaikin_ring(poly.exterior.coords, n)
        holes = [chaikin_ring(r.coords, n) for r in poly.interiors]
        return _Poly(ext, holes)

    def smooth_geom(geom, n=3):
        if geom.geom_type == "Polygon":
            return chaikin_poly(geom, n)
        if geom.geom_type == "MultiPolygon":
            return _MP([chaikin_poly(g, n) for g in geom.geoms])
        return geom

    buf = 0.6 * pw  # buffer 幅度 (像元): 合并近邻、削平粗锯齿
    try:  # join_style=2 (mitre); 老版本 shapely 无此参数则回退默认
        base = [g.buffer(buf, join_style=2).buffer(-buf, join_style=2).simplify(pw * 0.1)
                for g in geoms]
    except TypeError:
        base = [g.buffer(buf).buffer(-buf).simplify(pw * 0.1) for g in geoms]
    polys = [smooth_geom(g, 3) for g in base if not g.is_empty and g.area >= 3 * pw * pw]
    gdf = gpd.GeoDataFrame(geometry=polys, crs=4326)
    return _json.loads(gdf.to_json())


@st.cache_data
def load_city_geojson():
    """地级市边界 -> GeoJSON dict (EPSG:4326)"""
    import geopandas as gpd
    import json as _json

    gdf = gpd.read_file(CITY_SHP, encoding="utf-8")
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    return _json.loads(gdf[["name", "geometry"]].to_json())


@st.cache_data
def load_events():
    """干旱事件表 (游程理论, 按真实时间版); 行/列 与 SSI 栅格同网格, 可精确匹配"""
    df = pd.read_csv(EVENTS_CSV, encoding="utf-8-sig")
    df["行"] = df["行"].astype(int)
    df["列"] = df["列"].astype(int)
    return df


@st.cache_data
def load_city_labels():
    """地级市名称及其代表点坐标 (EPSG:4326), 用于地图上的河南省内城市标注"""
    import geopandas as gpd

    gdf = gpd.read_file(CITY_SHP, encoding="utf-8")
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    pts = gdf.geometry.representative_point()
    return [(str(n), (p.y, p.x)) for n, p in zip(gdf["name"], pts)]


@st.cache_data
def load_city_bounds():
    """河南省城市边界的外包框 (EPSG:4326), 用于地图 fit_bounds"""
    import geopandas as gpd

    gdf = gpd.read_file(CITY_SHP, encoding="utf-8")
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    w, s, e, n = gdf.total_bounds
    return float(s), float(w), float(n), float(e)


# ---------------- 图层样式配置 (专业制图配色) ----------------
# cmap 为 matplotlib 色带名; flip=True 时地图取 cmap(1-t) (使低值=色带高端颜色)
# Sen斜率的 vmin/vmax/ticks 为 None -> 运行时取数据实际最小/最大值
LAYER_STYLE = {
    "干旱频率": {"vmin": 0.0, "vmax": 1.0, "cmap": "YlOrRd", "flip": False,
               "ticks": [0, 0.25, 0.5, 0.75, 1],
               "label": "浅黄=低频, 深红=高频干旱区"},
    "MK Z值": {"vmin": -2.0, "vmax": 2.0, "cmap": "RdBu", "flip": False,
              "ticks": [-2, -1, 0, 1, 2],
              "label": "红=变干趋势 (Z<0), 蓝=变湿趋势 (Z>0)"},
    "Sen斜率": {"vmin": None, "vmax": None, "cmap": "RdYlGn", "flip": False,
              "ticks": None,
              "label": "红=负斜率 (变干), 绿=正斜率 (变湿)"},
    "Hurst指数": {"vmin": 0.0, "vmax": 1.0, "cmap": "PuOr", "flip": True,
               "ticks": [0, 0.25, 0.5, 0.75, 1],
               "label": "紫=趋势反转 (H<0.5), 橙=趋势持续 (H>0.5)"},
}


# ---------------- 渲染函数 (缓存) ----------------
@st.cache_data
def render_layer_png(layer_key, band_idx=0):
    """图层 -> matplotlib PNG (data URI) + 数据范围。
    layer_key: "SSI" / 图层名 / 高风险预警名; 结果缓存, 同图层只渲染一次,
    切换月份/图层或点击地图的重跑直接复用, 免费版弱 CPU 下提速明显。"""
    if layer_key == "SSI":
        arr = load_ssi()[0][band_idx]
        vmin, vmax, flip, cmap_name, interp = -2.0, 2.0, True, "RdYlBu_r", "bilinear"
    elif layer_key == HIGH_RISK:
        risk_mask, risk_valid, _ = load_high_risk()
        vmin = vmax = None
        interp = "nearest"
    else:
        arr = load_layer(str(LAYER_PATHS[layer_key]))
        cfg = LAYER_STYLE[layer_key]
        vmin, vmax = cfg["vmin"], cfg["vmax"]
        if vmin is None:  # Sen斜率: 用数据实际范围
            vmin, vmax = float(np.nanmin(arr)), float(np.nanmax(arr))
        flip, cmap_name, interp = cfg["flip"], cfg["cmap"], "bilinear"

    # 生成 RGBA: 风险图层只渲染非风险淡灰轮廓 (高风险区改用矢量多边形), 其余为连续渐变
    if layer_key == HIGH_RISK:
        rgba = np.zeros((*risk_mask.shape, 4))
        rgba[risk_valid] = (0.93, 0.93, 0.93, 0.95)  # 非风险有效区: 极淡灰显示省域轮廓
        grid = risk_mask.shape
        print(f"=== 图层调试 [{HIGH_RISK}] 高风险像元={int(risk_mask.sum())} / {int(risk_valid.sum())} (矢量多边形渲染) ===")
    else:
        # 归一化到 [0,1], 按图层色带取色 (flip 时用 1-t 使低值=色带高端)
        cmap = plt.get_cmap(cmap_name)
        t = np.clip((arr - vmin) / (vmax - vmin), 0, 1)
        rgba = cmap(1.0 - t if flip else t)  # (rows, cols, 4) float 0~1
        rgba[~np.isfinite(arr)] = 0.0  # NoData 完全透明
        grid = arr.shape
        print(f"=== 图层调试 [{layer_key}] cmap={cmap_name} vmin={vmin:.3g} vmax={vmax:.3g} flip={flip} 有效像元={int(np.isfinite(arr).sum())} ===")

    # matplotlib 渲染为 PNG (bilinear 插值形成连续面), 转 base64 data URI
    fig = plt.figure(figsize=(grid[1] / 10, grid[0] / 10), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.imshow(rgba, interpolation=interp)  # 行序北在上, origin 默认 upper 即正确
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True)
    plt.close(fig)
    img_uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    return img_uri, vmin, vmax


@st.cache_data
def render_legend_png(is_risk, vmin, vmax, flip, cmap_name, ticks, label):
    """图例 -> PNG bytes (缓存); ticks 需传 tuple (可哈希)。"""
    from matplotlib.patches import FancyBboxPatch, Patch

    fig = plt.figure(figsize=(6, 0.75), dpi=130)
    fig.patch.set_alpha(0.0)
    if is_risk:
        ax = fig.add_axes([0.06, 0.30, 0.88, 0.40])
        ax.patch.set_alpha(0.0)
        ax.axis("off")
        handles = [Patch(facecolor=(210 / 255, 50 / 255, 50 / 255, 0.75), edgecolor="#961414",
                         linewidth=1.2, label="高风险区（红色）"),
                   Patch(facecolor="#ededed", edgecolor="#c8c8c8", linewidth=0.6, label="非风险区（灰色）")]
        ax.legend(handles=handles, loc="center", ncol=2, frameon=False, fontsize=9)
        ax.set_title(label, fontsize=9, color="#444444", pad=2)
    else:
        ax = fig.add_axes([0.06, 0.40, 0.88, 0.30])  # 内边距: 四周留白
        ax.patch.set_alpha(0.0)
        norm = mpl_colors.Normalize(vmin=vmin, vmax=vmax)
        # colorbar 用 cmap(t) 取值; 地图 flip 时用 cmap(1-t), 因此图例色带需取反
        bar_cmap = plt.get_cmap(cmap_name).reversed() if flip else plt.get_cmap(cmap_name)
        fig.colorbar(mpl_cm.ScalarMappable(norm=norm, cmap=bar_cmap),
                     cax=ax, orientation="horizontal", ticks=list(ticks))
        ax.set_xlabel(label, fontsize=9)
        ax.tick_params(labelsize=9, length=2)
    # 圆角矩形底板
    card = FancyBboxPatch((0.015, 0.06), 0.97, 0.88,
                          boxstyle="round,pad=0.012,rounding_size=0.06",
                          transform=fig.transFigure, facecolor="white",
                          edgecolor="#e3e3e3", linewidth=0.8, zorder=-10)
    fig.add_artist(card)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True)
    plt.close(fig)
    return buf.getvalue()


# ---------------- 页面 ----------------
st.set_page_config(page_title="河南土壤水分干旱监测平台", layout="wide")
st.title("河南土壤水分干旱监测平台 (2016–2025, 4–5 月 SSI)")

cube, bounds, transform = load_ssi()
west, south, east, north = bounds

# ---- 侧边栏 ----
st.sidebar.header("图层控制")
month = st.sidebar.select_slider(
    "月份 (数据含每年 4、5 月, 共 20 个月)",
    options=MONTHS, value=MONTHS[-1],
)
HIGH_RISK = "未来高风险预警"
layer_choice = st.sidebar.selectbox("显示图层", ["SSI"] + list(LAYER_PATHS.keys()) + [HIGH_RISK])
LAYER_DESC = {
    "SSI": "红色代表土壤水分亏缺（干旱），蓝色代表水分充足（湿润），黄色为正常。",
    "干旱频率": "颜色越深，代表 2016-2025 年 4-5 月发生干旱的频率越高。",
    "MK Z值": "红色代表变干趋势（Z<0），蓝色代表变湿趋势（Z>0），颜色越深趋势越显著。",
    "Sen斜率": "红色代表下降速率快（变干快），绿色代表上升速率快（变湿快）。",
    "Hurst指数": "橙色代表趋势具有持续性（如变干趋势会继续），紫色代表趋势可能反转。",
    HIGH_RISK: "深红色区域 = 过去显著变干且未来可能持续的高风险区。",
}
st.sidebar.caption(LAYER_DESC[layer_choice])

# ---- 地图图层 (所有图层统一: matplotlib PNG -> ImageOverlay 连续面) ----
band_idx = MONTHS.index(month)
is_risk = layer_choice == HIGH_RISK
if layer_choice == "SSI":
    img_uri, vmin, vmax = render_layer_png("SSI", band_idx)  # flip: SSI 低=红(干旱), 高=蓝(湿润)
    flip, cmap_name = True, "RdYlBu_r"
    label = "红=干旱 (SSI≤-1), 黄=正常, 蓝=湿润 (SSI>0)"
    ticks = [-2, -1, 0, 1, 2]
elif is_risk:
    risk_mask, risk_valid, risk_criteria = load_high_risk()
    img_uri, _, _ = render_layer_png(HIGH_RISK)
    vmin = vmax = flip = cmap_name = None  # 风险图层无连续色带, 图例走分类分支
    ticks = ()
    label = "红色区域 = 未来持续变干高风险预警区 (静态图层, 不随月份变化)"
else:
    img_uri, vmin, vmax = render_layer_png(layer_choice)
    st_cfg = LAYER_STYLE[layer_choice]
    if st_cfg["vmin"] is not None:  # Sen斜率 (None) 已在渲染函数内取数据实际范围
        vmin, vmax = st_cfg["vmin"], st_cfg["vmax"]
    flip = st_cfg["flip"]
    cmap_name = st_cfg["cmap"]
    label = st_cfg["label"] + " (静态图层, 不随月份变化)"
    ticks = st_cfg["ticks"] or np.linspace(vmin, vmax, 5).round(3).tolist()

# ---- folium 地图: matplotlib 渐变 PNG -> ImageOverlay 贴图 (保持连续面效果) ----
# 底图: Esri.WorldGrayCanvas (免费无需 API Key; Carto 已强制 Key, 会出现水印)
city_s, city_w, city_n, city_e = load_city_bounds()  # 河南省外包框, 供 fit_bounds 用
ESRI_ATTR = "Esri"
m = folium.Map(location=[33.8, 113.5], tiles=None)
folium.TileLayer(  # 浅灰纯色底图, 不带任何标签
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
    attr=ESRI_ATTR, name="Esri 灰底图", control=False,
).add_to(m)
folium.raster_layers.ImageOverlay(
    image=img_uri,  # data URI (matplotlib 渲染的 PNG, NoData 已透明)
    bounds=[[south, west], [north, east]],
    opacity=0.9,
).add_to(m)
if is_risk:  # 高风险连片区域: 矢量多边形 (栅格矢量化+平滑), 警示红填充 + 深红边界
    folium.GeoJson(
        load_high_risk_geojson(),
        style_function=lambda f: {
            "fillColor": "#d23232",   # RGB [210, 50, 50] 警示红
            "fillOpacity": 0.75,
            "color": "#961414",       # RGB [150, 20, 20] 深红边界
            "weight": 1.5,
            "fill": True,
        },
    ).add_to(m)
# 地级市边界线: 极浅的细灰线, 仅作参考不抢视觉焦点
folium.GeoJson(
    load_city_geojson(),
    style_function=lambda f: {"color": "#c8c8c8", "weight": 0.8, "opacity": 1.0, "fill": False},
).add_to(m)
# 去掉地名 Tooltip 的白色背景框与 Leaflet 右下角版权水印, 通过注入 CSS 覆盖默认样式
MAP_CSS = """
<style>
  .leaflet-tooltip.city-label, .leaflet-tooltip.city-label::before {
    background: transparent !important; border: none !important;
    box-shadow: none !important; padding: 0 !important;
  }
  .leaflet-control-attribution { display: none !important; }
</style>
"""
m.get_root().html.add_child(folium.Element(MAP_CSS))
# 仅标注河南省内地级市名: Marker + 永久 Tooltip (居中, 深灰小字, 无背景框, 无图标)
LABEL_STYLE = "color:#555555;font-size:10px;font-weight:500;padding:0;"
NO_ICON = folium.DivIcon(html="", icon_size=(0, 0))  # 隐藏默认蓝色图钉
for name, (clat, clon) in load_city_labels():
    folium.Marker(
        location=[clat, clon],
        tooltip=folium.Tooltip(name, permanent=True, direction="center",
                               offset=(0, 0), style=LABEL_STYLE, class_name="city-label"),
        icon=NO_ICON,
    ).add_to(m)
# 视野自动缩放到刚好包裹河南省 (城市边界 shp 的外包框)
m.fit_bounds([[city_s, city_w], [city_n, city_e]], padding=(15, 15))
# 已点击像元标记
if st.session_state.get("click"):
    cl = st.session_state.click
    folium.CircleMarker(
        location=[cl["lat"], cl["lon"]], radius=7,
        color="#000000", weight=2, fill=True,
        fill_color="#ffd700", fill_opacity=0.35,
    ).add_to(m)

st.subheader(f"{month} · {layer_choice}" if layer_choice == "SSI" else layer_choice)
res = st_folium(
    m, height=620, use_container_width=True,
    returned_objects=["last_clicked"], key="map",
)
# ---- 图例 (圆角矩形卡片; 风险图层为两色分类图例, 其余为渐变条; 渲染结果缓存) ----
legend_png = render_legend_png(is_risk, vmin, vmax, flip, cmap_name, tuple(ticks), label)
st.image(legend_png, use_container_width=True)

# ---- 统计面板 ----
def _nearest_cities(mask):
    """高风险像元中心点 -> 最近地级市名, 按出现次数降序"""
    from collections import Counter

    rows, cols = np.where(mask)
    if len(rows) == 0:
        return []
    lons = transform.c + (cols + 0.5) * transform.a
    lats = transform.f + (rows + 0.5) * transform.e
    labels = load_city_labels()
    names = []
    for lon, lat in zip(lons, lats):
        k = np.cos(np.radians(lat))  # 经度距离按纬度缩放
        dists = [(lat - clat) ** 2 + (k * (lon - clon)) ** 2 for _, (clat, clon) in labels]
        names.append(labels[int(np.argmin(dists))][0])
    return [n for n, _ in Counter(names).most_common()]


if is_risk:
    risk_mask, risk_valid, risk_criteria = load_high_risk()
    high_n = int(risk_mask.sum())
    risk_ratio = high_n / int(risk_valid.sum()) if risk_valid.any() else 0.0
    cities = _nearest_cities(risk_mask)
    c1, c2, c3 = st.columns(3)
    c1.metric("高风险像元数", high_n)
    c2.metric("高风险面积占比", f"{risk_ratio:.1%}")
    if cities:
        shown = "、".join(cities[:4]) + ("等" if len(cities) > 4 else "")
        c3.metric("主要涉及城市", shown)
    else:
        c3.metric("主要涉及城市", "—")
    st.caption(f"判定标准: {risk_criteria}")
    if high_n == 0:
        st.info("当前判据下无高风险像元。")
else:
    band = cube[band_idx]
    valid_n = int(np.isfinite(band).sum())
    dry_n = int((band <= -1).sum())
    ratio = dry_n / valid_n if valid_n else 0.0
    c1, c2, c3 = st.columns(3)
    c1.metric("有效像元数", valid_n)
    c2.metric("干旱像元数 (SSI ≤ -1)", dry_n)
    c3.metric(f"{month} 干旱面积占比", f"{ratio:.1%}")
    st.progress(min(max(ratio, 0.0), 1.0))

# ---- 点击处理: st_folium 返回的 last_clicked 坐标 -> 像元 ----
last = (res or {}).get("last_clicked") if isinstance(res, dict) else None
if last and isinstance(last, dict) and "lat" in last and "lng" in last:
    lon, lat = float(last["lng"]), float(last["lat"])
    col = int((lon - transform.c) / transform.a)
    row = int((lat - transform.f) / transform.e)
    if 0 <= row < cube.shape[1] and 0 <= col < cube.shape[2] and np.isfinite(cube[0, row, col]):
        st.session_state.click = {"row": row, "col": col, "lon": lon, "lat": lat}

# ---- 折线图: 全省平均 SSI ----
st.subheader("全省平均 SSI 时间变化 (2016–2025, 4–5 月)")
mean_ssi = np.nanmean(cube, axis=(1, 2))
df_mean = pd.DataFrame({"月份": MONTHS, "全省平均SSI": mean_ssi}).set_index("月份")
st.line_chart(df_mean, x_label="月份", y_label="SSI")

# ---- 折线图: 点击像元的 20 个月 SSI (plotly) ----
st.subheader("点击位置 SSI 序列 (20 个月)")
if st.session_state.get("click"):
    cl = st.session_state.click
    series = cube[:, cl["row"], cl["col"]]
    st.markdown(f"**选中像元**: 行 {cl['row']}, 列 {cl['col']} "
                f"(约 {cl['lon']:.3f}°E, {cl['lat']:.3f}°N)")
    fig_px = go.Figure(go.Scatter(
        x=MONTHS, y=series, mode="lines+markers",
        line=dict(color="#d62728", width=2), marker=dict(size=6),
    ))
    fig_px.add_hline(y=0, line_dash="dash", line_color="gray",
                     annotation_text="SSI = 0")
    fig_px.update_layout(
        height=320, margin=dict(l=10, r=10, t=10, b=10),
        yaxis_title="SSI", xaxis_title="月份",
    )
    st.plotly_chart(fig_px, use_container_width=True)

    # ---- 干旱事件列表 (游程理论, 按 行/列 精确匹配点击像元) ----
    ev_px = load_events()
    ev_px = ev_px[(ev_px["行"] == cl["row"]) & (ev_px["列"] == cl["col"])]
    ev_px = ev_px.sort_values("起始月份")[["起始月份", "结束月份", "持续月数", "累积强度"]]
    st.markdown(f"**该位置干旱事件档案** (SSI ≤ -1.0, 共 {len(ev_px)} 次)")
    if len(ev_px):
        st.dataframe(ev_px, use_container_width=True, hide_index=True)
    else:
        st.info("该位置 2016-2025 年无干旱事件")
else:
    st.info("在地图上点击任意像元, 此处将显示该位置 2016-04 ~ 2025-05 的 SSI 折线图")
