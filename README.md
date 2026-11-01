# 中原农情旱涝双端风险研判与空间态势感知系统

![Streamlit](https://img.shields.io/badge/Streamlit-数据门户-FF4B4B?logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)
![数据周期](https://img.shields.io/badge/数据周期-2016--2025-2E7D32)

> **一句话简介**：基于河南省 18 市 2016—2025 年 4—5 月标准化土壤水分指数（SSI），融合 Mann-Kendall 趋势检验、Hurst 持续性分析与冬小麦产量面板，实现旱涝双端风险研判与空间态势感知的一站式科学数据门户。

---

## 核心功能

| 功能模块 | 能力说明 |
|:---:|:---|
| **旱涝监测** | 逐期 SSI 空间制图与旱涝分级渲染、历史偏干 / 偏湿频率统计、Hurst 持续性指数、Sen 斜率趋势分析，支持浅层根区（0—28cm）与表层（0—7cm）双层级切换 |
| **智能研判** | 点击城市即时生成综合研判卡片（SSI 值、旱涝等级、减产率仪表盘）；基于「Sen 斜率 + Hurst 指数」识别未来干旱 / 渍害双端高风险区，并输出动态研判结论 |
| **数据检索** | 全局搜索栏支持城市、年份、数据集名称一键检索并跨模块联动；旱涝属性数据库提供 6 大核心数据集的在线浏览与排序 |

## 本地运行

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动应用
streamlit run app.py
```

启动后浏览器自动打开（默认 `http://localhost:8501`），即可访问数据门户。

## 目录结构

```text
henan-drought-monitor/
├── app.py                  # 应用入口（Streamlit 数据门户）
├── requirements.txt        # Python 依赖清单
├── data/                   # 原始输入（SSI 栅格、市界 shp、产量年鉴）
│   ├── 4-5月河南SSI/       # 多波段 SSI 栅格（2016—2025，4—5 月）
│   └── 河南省市shp/        # 河南省市界矢量边界
├── output/                 # 分析成果
│   ├── Mann-Kendall 趋势检验/  # MK Z 值与 Sen 斜率栅格
│   ├── hurst指数/          # Hurst 持续性指数栅格
│   ├── 干旱频率/           # 干旱频率栅格
│   ├── 游程理论/           # 干旱事件回溯库
│   └── 产量关联分析/        # 城市 × 年份产量面板与关系统计
├── assets/                 # 静态资源（字体、横幅图）
├── scripts/                # 数据处理与分析脚本
└── docs/                   # 项目文档
```

## 技术栈

- **应用框架**：Streamlit + streamlit-folium（交互式地图）
- **空间分析**：rasterio / geopandas / shapely
- **统计方法**：标准化土壤水分指数（SSI）、Mann-Kendall 趋势检验、Sen 斜率、Hurst 持续性、游程理论干旱事件识别
- **可视化**：Plotly / Matplotlib / Folium
