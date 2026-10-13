# -*- coding: utf-8 -*-
"""
提取 河南省SSI 下多个 SSI 数据集 中每年 4 月、5 月数据, 各堆叠为一个 20 波段 tif
输入: D:\\大二\\干旱预测\\河南省SSI\\{swvl2,swvl3,swvl_root_ssi}\\河南SSI_YYYYMM.tif (201601-202512)
输出: D:\\大二\\干旱预测\\4-5月河南SSI\\4-5月河南SSI_{数据集}_2016-2025.tif
      20 波段, 顺序 2016-04, 2016-05, ..., 2025-05; 格式与 swvl1 版 4-5月河南SSI_2016-2025.tif 一致
      (float32, nodata=-3.4e38, deflate, EPSG:4326, 波段描述=年月), 网格取各源数据自身的网格
依赖: rasterio
"""
from pathlib import Path

import numpy as np
import rasterio

SRC_ROOT = Path(r"D:\大二\干旱预测\河南省SSI")
OUT_DIR = Path(r"D:\大二\干旱预测\4-5月河南SSI")

DATASETS = {
    "shallow": "swvl_shallow_ssi",  # 浅层根区 (0-28cm 简单平均) SSI
    # "swvl2": "swvl2",             # 已生成, 需要重跑时取消注释
    # "swvl3": "swvl3",
    # "root": "swvl_root_ssi",
}

YEARS = range(2016, 2026)
MONTHS = (4, 5)


def stack_45(src_dir: Path, out_path: Path) -> None:
    """从 src_dir 收集 4/5 月 tif, 校验网格一致后堆叠写出多波段 tif"""
    items = []
    for year in YEARS:
        for month in MONTHS:
            path = src_dir / f"河南SSI_{year}{month:02d}.tif"
            if not path.exists():
                raise FileNotFoundError(f"缺少文件: {path}")
            items.append((f"{year}-{month:02d}", path))
    print(f"  收集 {len(items)} 个栅格: {items[0][0]} ~ {items[-1][0]}")

    with rasterio.open(items[0][1]) as ref:
        ref_meta = (ref.width, ref.height, ref.transform, ref.crs, ref.nodata)
        profile = ref.profile.copy()
    for label, path in items[1:]:
        with rasterio.open(path) as src:
            if (src.width, src.height, src.transform, src.crs, src.nodata) != ref_meta:
                raise ValueError(f"网格不一致: {path.name}")

    profile.update(count=len(items), dtype="float32", nodata=ref_meta[4], compress="deflate")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        for idx, (label, path) in enumerate(items, start=1):
            with rasterio.open(path) as src:
                dst.write(src.read(1), idx)
            dst.set_band_description(idx, label)
    print(f"  -> 输出: {out_path}  ({len(items)} 波段, 网格 {ref.width}x{ref.height})")


def main():
    for out_tag, src_sub in DATASETS.items():
        src_dir = SRC_ROOT / src_sub
        out_path = OUT_DIR / f"4-5月河南SSI_{out_tag}_2016-2025.tif"
        print(f"===== {src_sub} =====")
        try:
            stack_45(src_dir, out_path)
        except Exception as exc:
            print(f"  [警告] {src_sub} 处理失败: {exc}")
    print("\n全部完成")


if __name__ == "__main__":
    main()
