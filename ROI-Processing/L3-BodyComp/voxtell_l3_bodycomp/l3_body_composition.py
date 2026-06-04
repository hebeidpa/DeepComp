# -*- coding: utf-8 -*-
"""
L3-level body composition analysis built on top of VoxTell.

为什么需要这个模块
------------------
VoxTell 是一个 *文本提示式* 的通用 3D 分割模型：给一句话/一个词，输出整卷
体数据上对应结构的 3D mask。它本身**做不了** L3 椎体层面的体成分量化，原因有三：

1. 它没有“在 L3 层面”的概念——prompt "skeletal muscle" 会分割全卷的骨骼肌，
   而不是 L3 这一张横断面。
2. 内脏脂肪(VAT)和皮下脂肪(SAT)在 CT 上**密度完全相同**，区分它们靠的是
   解剖位置（腹壁肌肉包络内 vs 外），而不是灰度。零样本文本提示无法稳定区分。
3. 体成分是一个**定量测量**任务（标准化 L3 单层的横截面积 cm²），不是单纯分割。

因此正确的工程做法不是去改 VoxTell 的网络权重（那需要重新训练），而是
**在 VoxTell 之上增加一个任务流水线**：用 VoxTell 做“定位 + 粗分割”，再用
经典图像处理（HU 窗 + 形态学包络）做“精修 + VAT/SAT 分离 + 量化”。本模块即此流水线。

流水线
------
  1. 定位 L3：用 VoxTell 分割 L3 椎体（多个同义 prompt 取并集），取椎体的
     代表性轴位层（最大截面层或质心层）作为 L3 层。
  2. 粗分割：一次性用 VoxTell 分割骨骼肌相关结构（腰大肌、竖脊肌、腹壁肌等）
     和（可选）腹腔，得到“肌肉所在区域”。
  3. 精修：在 L3 层（或薄层 slab）内，肌肉 = 肌肉区域 ∩ HU∈[-29,150]；
     脂肪 = 体区域 ∩ HU∈[-190,-30]。
  4. VAT/SAT 分离：用肌肉(+脊柱)构造腹壁包络（闭运算补缺口后填充），
     VAT = 脂肪 ∩ 包络内腔；SAT = 脂肪 ∩ 包络外（仍在体内）。
  5. 量化：按层内像素间距换算 cm²，输出 SM/VAT/SAT/IMAT 面积、SM 平均 HU、
     VAT/SAT 比；给定身高可算 SMI。保存 label NIfTI + 叠加 PNG + JSON/CSV。

坐标约定（来自 nnunetv2 NibabelIOWithReorient）
----------------------------------------------
  读图后数组形状 (1, Z, Y, X)，重排到 RAS：
    axis 0 = Z = 上下方向（轴位切片轴，index 越大越靠头侧）
    axis 1 = Y = 前后方向
    axis 2 = X = 左右方向
  spacing = [z_mm, y_mm, x_mm]，故层内像素面积 = spacing[1] * spacing[2] (mm²)。

注意 / 局限
----------
* HU 窗为常用经验值（肌肉 -29..150，脂肪 -190..-30），**增强期相、kVp、不同
  扫描仪都会影响**，临床使用前应在自己的数据上校验/微调。
* VAT/SAT 分离依赖腹壁肌肉包络的完整性；VoxTell 肌肉 mask 越完整越好。
* 这是“用基础模型 + 经典后处理”的稳健方案；若追求最高精度，可：
  (a) 用本流水线产出的 mask 作为伪标签微调 VoxTell（仓库 roadmap 计划开放微调）；
  (b) 与专用工具交叉验证（如 Stanford Comp2Comp、TotalSegmentator 的
      `tissue_types` / `body` 任务）。

作者：在 VoxTell (MIC-DKFZ, arXiv:2511.11450) 基础上扩展。许可遵循 VoxTell 的 Apache-2.0。
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage as ndi

# ----------------------------------------------------------------------------- #
#  可配置常量
# ----------------------------------------------------------------------------- #
HU_MUSCLE: Tuple[int, int] = (-29, 150)     # 骨骼肌 HU 窗（Mitsiopoulos/常规）
HU_FAT: Tuple[int, int] = (-190, -30)       # 脂肪组织 HU 窗
HU_BODY_THRESHOLD: int = -500               # 体区域阈值（> 此值视为组织）
HU_BONE_THRESHOLD: int = 250                # 骨/脊柱阈值，用于构造包络

# VoxTell 文本提示（可按需增删；同义词取并集以提高召回）
VERTEBRA_PROMPTS: List[str] = [
    "L3 vertebra", "third lumbar vertebra", "L3 vertebral body",
]
MUSCLE_PROMPTS: List[str] = [
    "skeletal muscle",
    "psoas major muscle",
    "erector spinae muscle",
    "quadratus lumborum muscle",
    "abdominal wall muscles",
    "rectus abdominis muscle",
]
CAVITY_PROMPTS: List[str] = [          # 可选：用于细化内脏腔
    "abdominal cavity", "peritoneal cavity",
]

LABELS = {"background": 0, "skeletal_muscle": 1, "visceral_fat": 2,
          "subcutaneous_fat": 3, "intermuscular_fat": 4}


# ----------------------------------------------------------------------------- #
#  纯经典 CV 部分（不依赖 VoxTell / torch，可独立单元测试）
# ----------------------------------------------------------------------------- #
def largest_component(mask: np.ndarray) -> np.ndarray:
    """保留最大连通域。"""
    if mask.sum() == 0:
        return mask
    lbl, n = ndi.label(mask)
    if n <= 1:
        return mask
    sizes = ndi.sum(np.ones_like(lbl), lbl, index=np.arange(1, n + 1))
    return lbl == (1 + int(np.argmax(sizes)))


def body_mask_2d(hu: np.ndarray, thr: int = HU_BODY_THRESHOLD) -> np.ndarray:
    """从一张轴位 HU 图得到病人体区域（去除空气与扫描床）。"""
    m = hu > thr
    # 形态学开运算断开与扫描床的细连接，再取最大连通域并填洞
    m = ndi.binary_opening(m, structure=np.ones((3, 3)), iterations=2)
    m = largest_component(m)
    m = ndi.binary_fill_holes(m)
    return m


def refine_muscle_2d(hu: np.ndarray, muscle_region: np.ndarray, body: np.ndarray,
                     hu_window: Tuple[int, int] = HU_MUSCLE,
                     dilate_px: int = 3) -> np.ndarray:
    """
    用 HU 窗在“肌肉所在区域”内精修出真正的肌肉体素。
    muscle_region 为 VoxTell 给出的肌肉粗 mask（定位用），HU 窗去除肌内脂肪/血管。
    若 muscle_region 为空（模型失败），退化为整张体区域内的 HU 窗。
    """
    win = (hu >= hu_window[0]) & (hu <= hu_window[1]) & body
    if muscle_region.sum() == 0:
        return win
    region = ndi.binary_dilation(muscle_region, structure=np.ones((3, 3)),
                                 iterations=max(1, dilate_px)) & body
    return win & region


def intermuscular_fat_2d(hu: np.ndarray, muscle_region: np.ndarray, body: np.ndarray,
                         hu_window: Tuple[int, int] = HU_FAT,
                         dilate_px: int = 3) -> np.ndarray:
    """肌内/肌间脂肪 (IMAT)：肌肉所在区域内的脂肪 HU 体素。"""
    if muscle_region.sum() == 0:
        return np.zeros_like(body)
    region = ndi.binary_dilation(muscle_region, structure=np.ones((3, 3)),
                                 iterations=max(1, dilate_px)) & body
    return (hu >= hu_window[0]) & (hu <= hu_window[1]) & region


def abdominal_envelope_2d(muscle: np.ndarray, spine: Optional[np.ndarray] = None,
                          close_px: int = 8) -> np.ndarray:
    """
    由腹壁肌肉(+脊柱)构造“腹壁包络”：闭运算桥接 linea alba 等细缺口后填充，
    得到包络内的实心区域 wall_filled（= 肌肉墙 + 其包围的内脏腔）。
    """
    wall = muscle.copy()
    if spine is not None:
        wall = wall | spine
    if close_px > 0:
        wall = ndi.binary_closing(wall, structure=np.ones((close_px, close_px)))
    wall_filled = ndi.binary_fill_holes(wall)
    return wall_filled


def split_fat_2d(fat: np.ndarray, muscle: np.ndarray, body: np.ndarray,
                 spine: Optional[np.ndarray] = None,
                 cavity_hint: Optional[np.ndarray] = None,
                 close_px: int = 8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    将脂肪分为内脏(VAT)与皮下(SAT)。
      VAT = 脂肪 ∩ 内脏腔（腹壁包络内、排除肌肉本身）
      SAT = 脂肪 ∩ 包络外（仍在体内）
    cavity_hint：可选的 VoxTell 腹腔 mask，用于与形态学腔取交集做细化。
    返回 (VAT, SAT, wall_filled)。
    """
    wall_filled = abdominal_envelope_2d(muscle, spine, close_px=close_px)
    visceral_cavity = wall_filled & ~muscle          # 用原始肌肉挖空，避免闭运算吃进 VAT
    if cavity_hint is not None and cavity_hint.sum() > 0:
        # 形态学腔与模型腔取并集再与包络相交，提升对复杂腹壁的鲁棒性
        visceral_cavity = (visceral_cavity | (cavity_hint & wall_filled)) & ~muscle
    vat = fat & visceral_cavity
    sat = fat & ~wall_filled & body
    return vat, sat, wall_filled


def pick_l3_index(vertebra_mask: np.ndarray, mode: str = "max_area") -> int:
    """
    从 3D L3 椎体 mask (Z,Y,X) 取代表性轴位层 index。
      max_area : 椎体横截面积最大的层（≈ 椎体中部，临床常用）
      centroid : 椎体体素质心所在层
    """
    if vertebra_mask.sum() == 0:
        raise ValueError("L3 椎体 mask 为空，无法定位 L3 层。请检查朝向(RAS)或更换 prompt。")
    per_slice = vertebra_mask.reshape(vertebra_mask.shape[0], -1).sum(1)
    if mode == "centroid":
        z = int(np.round(np.average(np.arange(len(per_slice)), weights=per_slice)))
    else:
        z = int(np.argmax(per_slice))
    return z


def area_cm2(mask2d: np.ndarray, spacing_yx: Tuple[float, float]) -> float:
    """层内面积 cm²；spacing_yx = (y_mm, x_mm)。"""
    return float(mask2d.sum()) * spacing_yx[0] * spacing_yx[1] / 100.0


# ----------------------------------------------------------------------------- #
#  结果容器
# ----------------------------------------------------------------------------- #
@dataclass
class L3Result:
    l3_slice_index: int
    n_slices_averaged: int
    spacing_zyx: Tuple[float, float, float]
    skeletal_muscle_area_cm2: float
    visceral_fat_area_cm2: float
    subcutaneous_fat_area_cm2: float
    intermuscular_fat_area_cm2: float
    total_fat_area_cm2: float
    vat_sat_ratio: float
    skeletal_muscle_mean_hu: float
    skeletal_muscle_index: Optional[float] = None      # SMI = SMA / height_m^2
    notes: List[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)


# ----------------------------------------------------------------------------- #
#  顶层分析器（依赖 VoxTell）
# ----------------------------------------------------------------------------- #
class L3BodyCompositionAnalyzer:
    """
    用法::

        analyzer = L3BodyCompositionAnalyzer(model_dir="/path/voxtell_v1.1", device="cuda")
        result, label_volume, props = analyzer.analyze("abd_ct.nii.gz", out_dir="out",
                                                        height_m=1.72)
        print(result.to_json())
    """

    def __init__(self, model_dir: str, device: str = "cuda",
                 vertebra_prompts: Optional[List[str]] = None,
                 muscle_prompts: Optional[List[str]] = None,
                 cavity_prompts: Optional[List[str]] = None,
                 use_cavity_prompt: bool = False):
        import torch
        from voxtell.inference.predictor import VoxTellPredictor
        self._torch = torch
        dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
        self.device = dev
        self.predictor = VoxTellPredictor(model_dir=model_dir, device=dev)
        self.vertebra_prompts = vertebra_prompts or VERTEBRA_PROMPTS
        self.muscle_prompts = muscle_prompts or MUSCLE_PROMPTS
        self.cavity_prompts = cavity_prompts or CAVITY_PROMPTS
        self.use_cavity_prompt = use_cavity_prompt

    # -- VoxTell 调用：一次滑窗拿到所有 prompt 的并集分组 mask ----------------- #
    def _segment(self, img: np.ndarray) -> Dict[str, np.ndarray]:
        prompts: List[str] = list(self.vertebra_prompts) + list(self.muscle_prompts)
        if self.use_cavity_prompt:
            prompts += list(self.cavity_prompts)
        seg = self.predictor.predict_single_image(img, prompts)   # (P, Z, Y, X) uint8
        seg = seg.astype(bool)
        idx = 0
        nv = len(self.vertebra_prompts)
        nm = len(self.muscle_prompts)
        vert = seg[idx:idx + nv].any(0); idx += nv
        musc = seg[idx:idx + nm].any(0); idx += nm
        out = {"vertebra": vert, "muscle": musc}
        if self.use_cavity_prompt:
            nc = len(self.cavity_prompts)
            out["cavity"] = seg[idx:idx + nc].any(0)
        return out

    def analyze(self, image_path: str, out_dir: Optional[str] = None,
                height_m: Optional[float] = None, slab_mm: float = 0.0,
                l3_mode: str = "max_area",
                hu_muscle: Tuple[int, int] = HU_MUSCLE,
                hu_fat: Tuple[int, int] = HU_FAT,
                save_overlay: bool = True):
        from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient

        reader = NibabelIOWithReorient()
        img, props = reader.read_images([image_path])      # img: (1,Z,Y,X) HU, RAS
        spacing = tuple(float(s) for s in props["spacing"])  # (z,y,x) mm
        hu = np.asarray(img)[0] if img.ndim == 4 else np.asarray(img)  # (Z,Y,X)

        seg = self._segment(img)
        notes: List[str] = []

        # --- 定位 L3 层 ---
        z = pick_l3_index(seg["vertebra"], mode=l3_mode)

        # --- 选层 / 薄层 slab ---
        if slab_mm and slab_mm > 0 and spacing[0] > 0:
            half = max(0, int(round((slab_mm / 2.0) / spacing[0])))
            z0, z1 = max(0, z - half), min(hu.shape[0] - 1, z + half)
        else:
            z0 = z1 = z
        z_list = list(range(z0, z1 + 1))

        # --- 逐层计算掩膜与面积，最后对 slab 取均值 ---
        sm_a, vat_a, sat_a, imat_a, sm_hu_vals = [], [], [], [], []
        per_slice_masks = {}
        for zi in z_list:
            hu2d = hu[zi]
            body = body_mask_2d(hu2d)
            muscle_region = seg["muscle"][zi] & body
            spine = (hu2d > HU_BONE_THRESHOLD) & body
            sm = refine_muscle_2d(hu2d, muscle_region, body, hu_window=hu_muscle)
            imat = intermuscular_fat_2d(hu2d, muscle_region, body, hu_window=hu_fat)
            fat = (hu2d >= hu_fat[0]) & (hu2d <= hu_fat[1]) & body
            fat = fat & ~imat                                  # 总腔/皮下脂肪不含肌内脂肪
            cav = seg.get("cavity", None)
            cav2d = cav[zi] if cav is not None else None
            vat, sat, _ = split_fat_2d(fat, sm | muscle_region, body, spine=spine,
                                       cavity_hint=cav2d)
            sp_yx = (spacing[1], spacing[2])
            sm_a.append(area_cm2(sm, sp_yx))
            vat_a.append(area_cm2(vat, sp_yx))
            sat_a.append(area_cm2(sat, sp_yx))
            imat_a.append(area_cm2(imat, sp_yx))
            if sm.sum() > 0:
                sm_hu_vals.append(float(hu2d[sm].mean()))
            if zi == z:
                per_slice_masks = {"sm": sm, "vat": vat, "sat": sat, "imat": imat}

        sma = float(np.mean(sm_a)); vata = float(np.mean(vat_a))
        sata = float(np.mean(sat_a)); imata = float(np.mean(imat_a))
        sm_hu = float(np.mean(sm_hu_vals)) if sm_hu_vals else float("nan")
        smi = (sma / (height_m ** 2)) if (height_m and height_m > 0) else None

        if seg["vertebra"].sum() < 50:
            notes.append("L3 椎体 mask 体素很少，定位可能不准——请确认输入为 RAS 朝向。")
        if not np.isfinite(sm_hu):
            notes.append("该层未检出骨骼肌，请检查 HU 窗或肌肉 prompt。")

        result = L3Result(
            l3_slice_index=z, n_slices_averaged=len(z_list), spacing_zyx=spacing,
            skeletal_muscle_area_cm2=round(sma, 2),
            visceral_fat_area_cm2=round(vata, 2),
            subcutaneous_fat_area_cm2=round(sata, 2),
            intermuscular_fat_area_cm2=round(imata, 2),
            total_fat_area_cm2=round(vata + sata + imata, 2),
            vat_sat_ratio=round(vata / sata, 3) if sata > 0 else float("nan"),
            skeletal_muscle_mean_hu=round(sm_hu, 1),
            skeletal_muscle_index=round(smi, 2) if smi is not None else None,
            notes=notes,
        )

        # --- 构造 label 体（仅在 L3 层填标签），并按原始几何写盘 ---
        label_vol = np.zeros(hu.shape, dtype=np.uint8)
        m = per_slice_masks
        label_vol[z][m["imat"]] = LABELS["intermuscular_fat"]
        label_vol[z][m["sat"]] = LABELS["subcutaneous_fat"]
        label_vol[z][m["vat"]] = LABELS["visceral_fat"]
        label_vol[z][m["sm"]] = LABELS["skeletal_muscle"]

        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            base = os.path.basename(image_path).replace(".nii.gz", "").replace(".nii", "")
            reader.write_seg(label_vol, os.path.join(out_dir, f"{base}_L3_bodycomp.nii.gz"), props)
            with open(os.path.join(out_dir, f"{base}_L3_metrics.json"), "w", encoding="utf-8") as f:
                f.write(result.to_json())
            _write_csv(os.path.join(out_dir, f"{base}_L3_metrics.csv"), result)
            if save_overlay:
                try:
                    _save_overlay(hu[z], m, os.path.join(out_dir, f"{base}_L3_overlay.png"), result)
                except Exception as e:                       # 可视化失败不影响主结果
                    notes.append(f"overlay 渲染失败: {e}")

        return result, label_vol, props


# ----------------------------------------------------------------------------- #
#  输出辅助
# ----------------------------------------------------------------------------- #
def _write_csv(path: str, r: L3Result) -> None:
    import csv
    d = asdict(r); d.pop("notes", None)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(d.keys()); w.writerow(d.values())


def _save_overlay(hu2d: np.ndarray, masks: Dict[str, np.ndarray], path: str,
                  r: L3Result) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ov = np.zeros((*hu2d.shape, 3), dtype=float)
    ov[masks["sm"]] = [0.90, 0.10, 0.10]     # 肌肉 红
    ov[masks["vat"]] = [0.10, 0.85, 0.20]    # VAT 绿
    ov[masks["sat"]] = [0.15, 0.45, 1.00]    # SAT 蓝
    ov[masks["imat"]] = [1.00, 0.85, 0.10]   # IMAT 黄
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(hu2d, cmap="gray", vmin=-200, vmax=300); ax[0].set_title(f"L3 (slice {r.l3_slice_index})")
    ax[1].imshow(hu2d, cmap="gray", vmin=-200, vmax=300); ax[1].imshow(ov, alpha=0.45)
    ax[1].set_title("R=SM  G=VAT  B=SAT  Y=IMAT")
    cap = (f"SM {r.skeletal_muscle_area_cm2} cm²  VAT {r.visceral_fat_area_cm2}  "
           f"SAT {r.subcutaneous_fat_area_cm2}  SM-HU {r.skeletal_muscle_mean_hu}")
    fig.suptitle(cap, fontsize=10)
    for a in ax:
        a.axis("off")
    plt.tight_layout(); plt.savefig(path, dpi=110, bbox_inches="tight"); plt.close(fig)


# ----------------------------------------------------------------------------- #
#  CLI
# ----------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(
        description="基于 VoxTell 的 L3 椎体层面体成分自动分析（骨骼肌 / 内脏脂肪 / 皮下脂肪）")
    p.add_argument("-i", "--input", required=True, help="输入 CT NIfTI (.nii/.nii.gz)")
    p.add_argument("-m", "--model", required=True, help="VoxTell 模型目录，如 .../voxtell_v1.1")
    p.add_argument("-o", "--output", required=True, help="输出目录")
    p.add_argument("--device", default="cuda", help="cuda 或 cpu")
    p.add_argument("--height_m", type=float, default=None, help="病人身高(米)，用于计算 SMI")
    p.add_argument("--slab_mm", type=float, default=0.0, help="L3 薄层平均厚度(mm)，0=单层")
    p.add_argument("--l3_mode", default="max_area", choices=["max_area", "centroid"])
    p.add_argument("--use_cavity_prompt", action="store_true",
                   help="额外用 VoxTell 腹腔 prompt 细化 VAT/SAT 分界")
    p.add_argument("--hu_muscle", type=int, nargs=2, default=list(HU_MUSCLE))
    p.add_argument("--hu_fat", type=int, nargs=2, default=list(HU_FAT))
    args = p.parse_args()

    analyzer = L3BodyCompositionAnalyzer(
        model_dir=args.model, device=args.device,
        use_cavity_prompt=args.use_cavity_prompt)
    result, _, _ = analyzer.analyze(
        args.input, out_dir=args.output, height_m=args.height_m,
        slab_mm=args.slab_mm, l3_mode=args.l3_mode,
        hu_muscle=tuple(args.hu_muscle), hu_fat=tuple(args.hu_fat))
    print(result.to_json())


if __name__ == "__main__":
    main()
