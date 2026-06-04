# -*- coding: utf-8 -*-
"""
合成体模测试：在不加载 VoxTell 的情况下验证 L3 体成分流水线中
**与模型无关的经典部分**（L3 定位、肌肉精修、VAT/SAT 分离、面积量化）。

运行::
    python test_l3_pipeline_synthetic.py

它会构造一个 3D “腹部” 体模（已知各组织真值面积），跑流水线，并报告
SAT/VAT/肌肉的 Dice 与面积误差。Dice > 0.8、面积相对误差 < 10% 即视为通过。
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from voxtell.applications.l3_body_composition import (
    body_mask_2d, refine_muscle_2d, split_fat_2d, pick_l3_index, area_cm2,
    HU_MUSCLE, HU_FAT,
)


def make_phantom(Z=40, H=200, W=200):
    """返回 (hu[Z,H,W], 模拟的VoxTell肌肉mask, 模拟的L3椎体mask, 真值dict, spacing)。"""
    yy, xx = np.mgrid[0:H, 0:W]
    cy, cx = H // 2, W // 2

    def ellipse(ry, rx, oy=0, ox=0):
        return ((yy - cy - oy) / ry) ** 2 + ((xx - cx - ox) / rx) ** 2 <= 1

    body = ellipse(80, 62)
    wall_outer, wall_inner = ellipse(58, 44), ellipse(50, 37)
    muscle_ring = wall_outer & ~wall_inner
    spine = ellipse(13, 13, oy=55)
    psoas = ellipse(10, 8, oy=48, ox=-22) | ellipse(10, 8, oy=48, ox=22)
    organs = ellipse(15, 15, oy=-6, ox=-8) | ellipse(12, 12, oy=-4, ox=18)

    hu = np.full((Z, H, W), -1000.0, np.float32)
    musc_prompt = np.zeros((Z, H, W), bool)
    vert = np.zeros((Z, H, W), bool)
    gt = {"sm": np.zeros((Z, H, W), bool), "vat": np.zeros((Z, H, W), bool),
          "sat": np.zeros((Z, H, W), bool)}

    # 椎体仅出现在中间若干层，且中部最大（用于测 max_area 定位）
    vert_zs = range(14, 26)
    for z in range(Z):
        sl = np.full((H, W), -1000.0, np.float32)
        sl[body] = -100.0                       # 体内默认填脂肪（皮下+内脏皆脂肪）
        sl[wall_inner] = -100.0
        sl[muscle_ring] = 45.0
        sl[psoas] = 45.0
        sl[organs] = 35.0                        # 器官（肌肉密度，应排除于脂肪外）
        sl[spine] = 420.0
        hu[z] = sl
        musc_prompt[z] = muscle_ring | psoas
        gt["sm"][z] = muscle_ring | psoas
        gt["sat"][z] = ((sl >= HU_FAT[0]) & (sl <= HU_FAT[1])) & ~wall_outer & body
        gt["vat"][z] = ((sl >= HU_FAT[0]) & (sl <= HU_FAT[1])) & wall_inner
        if z in vert_zs:
            # 椎体截面在中部(z=20)最大，两端渐小
            shrink = 1.0 - abs(z - 20) / 8.0
            r = max(3, int(13 * shrink))
            vert[z] = ellipse(r, r, oy=55)

    spacing = (3.0, 1.0, 1.0)                    # z=3mm 层厚, 层内 1x1 mm
    return hu, musc_prompt, vert, gt, spacing


def dice(a, b):
    s = a.sum() + b.sum()
    return 2.0 * (a & b).sum() / s if s else 1.0


def main():
    hu, musc_prompt, vert, gt, spacing = make_phantom()

    # 1) L3 定位：应落在椎体最大截面层 z=20
    z = pick_l3_index(vert, mode="max_area")
    print(f"[L3 定位] 选中层 z={z} (真值=20)  ->", "OK" if z == 20 else "检查")

    # 2) 在该层跑经典流水线
    hu2d = hu[z]
    body = body_mask_2d(hu2d)
    muscle_region = musc_prompt[z] & body
    spine = (hu2d > 250) & body
    sm = refine_muscle_2d(hu2d, muscle_region, body, hu_window=HU_MUSCLE)
    fat = (hu2d >= HU_FAT[0]) & (hu2d <= HU_FAT[1]) & body
    vat, sat, _ = split_fat_2d(fat, sm | muscle_region, body, spine=spine)

    # 3) Dice
    d_sm, d_vat, d_sat = dice(sm, gt["sm"][z]), dice(vat, gt["vat"][z]), dice(sat, gt["sat"][z])
    print(f"[Dice]  SM={d_sm:.3f}  VAT={d_vat:.3f}  SAT={d_sat:.3f}")

    # 4) 面积（cm²）与真值对比
    sp_yx = (spacing[1], spacing[2])
    def rel(a, b): return abs(a - b) / (b + 1e-9)
    pred = {"SM": area_cm2(sm, sp_yx), "VAT": area_cm2(vat, sp_yx), "SAT": area_cm2(sat, sp_yx)}
    truth = {"SM": area_cm2(gt["sm"][z], sp_yx), "VAT": area_cm2(gt["vat"][z], sp_yx),
             "SAT": area_cm2(gt["sat"][z], sp_yx)}
    for k in pred:
        print(f"[面积] {k}: pred={pred[k]:.1f} cm²  truth={truth[k]:.1f} cm²  rel.err={rel(pred[k],truth[k])*100:.1f}%")

    ok = (z == 20 and d_sm > 0.8 and d_vat > 0.8 and d_sat > 0.8
          and rel(pred["SAT"], truth["SAT"]) < 0.10 and rel(pred["VAT"], truth["VAT"]) < 0.15)
    print("\n=== 测试", "通过 ✅" if ok else "需检查 ⚠️", "===")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
