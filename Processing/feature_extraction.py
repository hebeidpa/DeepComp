import os
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
from PIL import Image

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

# 本地模型路径
LOCAL_MODEL_PATH = "./Processing/weights/"


def window_ct(slice_hu: np.ndarray, wl: float, ww: float) -> np.ndarray:
    """HU -> [0,1] after windowing"""
    low = wl - ww / 2.0
    high = wl + ww / 2.0
    x = np.clip(slice_hu, low, high)
    x = (x - low) / (high - low + 1e-6)
    return x


def slice_to_rgb(slice_hu: np.ndarray, mode: str = "triple") -> Image.Image:
    """Convert a single CT slice (HU) into an RGB PIL image."""
    if mode == "gray":
        wl, ww = -600, 1500
        x = (window_ct(slice_hu, wl, ww) * 255.0).astype(np.uint8)
        rgb = np.stack([x, x, x], axis=-1)
        return Image.fromarray(rgb, mode="RGB")

    if mode == "triple":
        lung = (window_ct(slice_hu, -600, 1500) * 255.0).astype(np.uint8)
        medi = (window_ct(slice_hu, 40, 400) * 255.0).astype(np.uint8)
        bone = (window_ct(slice_hu, 300, 1500) * 255.0).astype(np.uint8)
        rgb = np.stack([lung, medi, bone], axis=-1)
        return Image.fromarray(rgb, mode="RGB")

    raise ValueError(f"Unknown mode={mode}. Use gray or triple.")


def ensure_hwd(vol: np.ndarray, slice_axis: int) -> np.ndarray:
    """Ensure volume is (H, W, D) where D is slice axis."""
    if slice_axis == 2:
        return vol
    if slice_axis == 0:
        return np.transpose(vol, (1, 2, 0))
    if slice_axis == 1:
        return np.transpose(vol, (0, 2, 1))
    raise ValueError("slice_axis must be 0, 1, or 2")


def get_vision_model(model):
    """获取视觉模型，适配不同模型结构"""
    possible_names = ['vision_model', 'vision_tower', 'vision_encoder']
    for name in possible_names:
        if hasattr(model, name):
            return getattr(model, name)
        elif hasattr(model, 'model') and hasattr(model.model, name):
            return getattr(model.model, name)
    raise RuntimeError("无法找到视觉编码器")


def pick_uniform_indices(indices: np.ndarray, k: int) -> np.ndarray:
    """From a sorted list of indices, pick k uniformly (with repetition if needed)."""
    indices = np.array(indices, dtype=int)
    if len(indices) == 0:
        return indices
    if k <= 0 or k >= len(indices):
        return indices
    pos = np.linspace(0, len(indices) - 1, k).round().astype(int)
    return indices[pos]


def bbox_from_mask(mask2d: np.ndarray):
    """Return bbox (ymin, ymax, xmin, xmax) for nonzero mask2d. If empty, return None."""
    ys, xs = np.where(mask2d > 0)
    if len(xs) == 0:
        return None
    ymin, ymax = ys.min(), ys.max()
    xmin, xmax = xs.min(), xs.max()
    return int(ymin), int(ymax), int(xmin), int(xmax)


def crop_with_pad(arr2d: np.ndarray, bbox, pad: int):
    """Crop 2D array by bbox with padding, clipping to valid range."""
    H, W = arr2d.shape
    ymin, ymax, xmin, xmax = bbox
    ymin = max(0, ymin - pad)
    xmin = max(0, xmin - pad)
    ymax = min(H - 1, ymax + pad)
    xmax = min(W - 1, xmax + pad)
    return arr2d[ymin:ymax + 1, xmin:xmax + 1], (ymin, ymax, xmin, xmax)


def extract_feats_from_pil_images(pil_images, image_processor, vision, device, dtype):
    """Given list of PIL images, run vision encoder and return slice_feats [K, D]."""
    all_pixel_values = []
    for img in pil_images:
        inputs = image_processor(images=img, return_tensors="pt")
        pv = inputs["pixel_values"]
        all_pixel_values.append(pv)

    pixel_values = torch.cat(all_pixel_values, dim=0).to(device=device, dtype=dtype)

    with torch.no_grad():
        vout = vision(pixel_values=pixel_values, return_dict=True)
        patch_feats = vout.last_hidden_state          # [K, N, D]
        slice_feats = patch_feats.mean(dim=1)         # [K, D]
        slice_feats = torch.nn.functional.normalize(slice_feats, dim=-1)
    return slice_feats


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--nii", required=True, help="Path to CT .nii or .nii.gz")
    ap.add_argument("--roi", required=True, help="Path to ROI mask .nii or .nii.gz")
    ap.add_argument("--out_dir", required=True, help="Output directory")

    ap.add_argument("--model", default=LOCAL_MODEL_PATH, help="HF repo id or local folder")
    ap.add_argument("--slice_axis", type=int, default=2, help="0/1/2, which axis is depth")
    ap.add_argument("--rgb_mode", choices=["gray", "triple"], default="triple")

    # ROI options
    ap.add_argument("--roi_label", type=int, default=None,
                    help="If set, only extract this label. Otherwise: extract all labels >0")
    ap.add_argument("--k", type=int, default=-1,
                    help="Max number of ROI-containing slices to use per ROI (uniform sampling). "
                         "If k<=0, use all ROI slices.")
    ap.add_argument("--pad", type=int, default=0, help="Padding (pixels) around ROI bbox crop")

    ap.add_argument("--save_roi_slice_feats", action="store_true",
                    help="Save per-slice ROI features for each ROI label")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- device/dtype ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"使用设备: {device}, 数据类型: {dtype}")

    # ---- load model & processor ----
    print(f"加载模型: {args.model}")
    processor = AutoProcessor.from_pretrained(args.model)
    image_processor = processor.image_processor if hasattr(processor, "image_processor") else processor

    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    ).eval()

    vision = get_vision_model(model)
    print(f"视觉编码器: {type(vision)}")

    # ---- load CT ----
    print(f"加载CT: {args.nii}")
    ct_nii = nib.load(args.nii)
    ct = ct_nii.get_fdata().astype(np.float32)
    if ct.ndim == 4:
        ct = ct[..., 0]
    if ct.ndim != 3:
        raise ValueError(f"CT expected 3D, got {ct.shape}")

    # ---- load ROI mask ----
    print(f"加载ROI: {args.roi}")
    roi_nii = nib.load(args.roi)
    roi = roi_nii.get_fdata()
    if roi.ndim == 4:
        roi = roi[..., 0]
    if roi.ndim != 3:
        raise ValueError(f"ROI expected 3D, got {roi.shape}")

    # ---- check alignment (strict) ----
    if ct.shape != roi.shape:
        raise ValueError(f"CT shape {ct.shape} != ROI shape {roi.shape}. "
                         "需要先把 ROI 重采样到 CT 网格。")
    # affine check (允许小误差)
    if not np.allclose(ct_nii.affine, roi_nii.affine, atol=1e-4):
        raise ValueError("CT affine 与 ROI affine 不一致。需要先对齐/重采样 ROI。")

    # ---- ensure (H,W,D) ----
    ct = ensure_hwd(ct, args.slice_axis)
    roi = ensure_hwd(roi, args.slice_axis)
    H, W, D = ct.shape
    print(f"CT/ROI 形状 (H,W,D): {ct.shape}")

    # ---- determine ROI labels ----
    roi_int = np.rint(roi).astype(np.int32)  # mask/label
    if args.roi_label is not None:
        labels = [int(args.roi_label)]
    else:
        labels = sorted([int(x) for x in np.unique(roi_int) if x > 0])

    if len(labels) == 0:
        raise RuntimeError("ROI mask 中没有任何 >0 的体素（空mask）。")

    print(f"将提取 ROI labels: {labels}")

    # ---- run per ROI label ----
    base = os.path.basename(args.nii)
    base = base.replace(".nii.gz", "").replace(".nii", "")

    roi_rows = []
    for lab in labels:
        mask_lab = (roi_int == lab).astype(np.uint8)

        # slices where ROI exists
        slice_has = np.where(mask_lab.sum(axis=(0, 1)) > 0)[0]
        if len(slice_has) == 0:
            print(f"[label={lab}] 无任何 ROI 层，跳过")
            continue

        # sample slices (uniform)
        if args.k is None or args.k <= 0:
            use_slices = slice_has
        else:
            use_slices = pick_uniform_indices(slice_has, args.k)

        print(f"[label={lab}] ROI层数={len(slice_has)}, 使用层数={len(use_slices)} -> {use_slices}")

        # build cropped ROI slice images
        pil_images = []
        used_slice_indices = []
        crop_boxes = []

        for z in use_slices:
            m2d = mask_lab[..., z]
            bbox = bbox_from_mask(m2d)
            if bbox is None:
                continue

            ct2d = ct[..., z]
            ct_crop, bbox_pad = crop_with_pad(ct2d, bbox, args.pad)

            pil = slice_to_rgb(ct_crop, mode=args.rgb_mode)
            pil_images.append(pil)
            used_slice_indices.append(int(z))
            crop_boxes.append(bbox_pad)

        if len(pil_images) == 0:
            print(f"[label={lab}] ROI裁剪后无有效切片，跳过")
            continue

        # extract slice feats
        slice_feats = extract_feats_from_pil_images(
            pil_images=pil_images,
            image_processor=image_processor,
            vision=vision,
            device=model.device,
            dtype=dtype
        )  # [K, D]

        # aggregate to ROI-level embedding
        with torch.no_grad():
            roi_feat = slice_feats.mean(dim=0)
            roi_feat = torch.nn.functional.normalize(roi_feat, dim=-1)

        # save ROI-level outputs
        roi_np = roi_feat.detach().cpu().numpy()
        roi_id = f"{base}_ROI{lab}"

        out_npy = os.path.join(args.out_dir, f"{roi_id}_feat.npy")
        np.save(out_npy, roi_np)

        out_csv = os.path.join(args.out_dir, f"{roi_id}_feat.csv")
        df_out = pd.DataFrame([roi_np], columns=[f"f{i}" for i in range(roi_np.shape[0])])
        df_out.insert(0, "roi_label", lab)
        df_out.insert(0, "nii", os.path.basename(args.nii))
        df_out.insert(0, "roi_mask", os.path.basename(args.roi))
        df_out.to_csv(out_csv, index=False)

        print(f"[label={lab}] 保存 ROI 特征: {out_npy} | {out_csv}")

        roi_rows.append({
            "nii": os.path.basename(args.nii),
            "roi_mask": os.path.basename(args.roi),
            "roi_label": lab,
            "n_roi_slices_total": int(len(slice_has)),
            "n_roi_slices_used": int(len(used_slice_indices)),
            "roi_feat_npy": os.path.basename(out_npy),
            "roi_feat_csv": os.path.basename(out_csv),
        })

        # optional per-slice save
        if args.save_roi_slice_feats:
            slice_np = slice_feats.detach().cpu().numpy()
            slice_csv = os.path.join(args.out_dir, f"{roi_id}_slice_feats.csv")
            slice_df = pd.DataFrame(slice_np, columns=[f"f{i}" for i in range(slice_np.shape[1])])
            slice_df.insert(0, "slice_index", used_slice_indices)

            # 把 bbox 也存一下，方便复现实验
            bbox_cols = ["ymin", "ymax", "xmin", "xmax"]
            bbox_arr = np.array(crop_boxes, dtype=int)
            for j, c in enumerate(bbox_cols):
                slice_df.insert(1 + j, c, bbox_arr[:, j])

            slice_df.insert(0, "roi_label", lab)
            slice_df.insert(0, "nii", os.path.basename(args.nii))
            slice_df.insert(0, "roi_mask", os.path.basename(args.roi))
            slice_df.to_csv(slice_csv, index=False)
            print(f"[label={lab}] 保存 ROI 每层特征: {slice_csv}")

    # save a summary file
    if len(roi_rows) > 0:
        summary_csv = os.path.join(args.out_dir, f"{base}_roi_feature_summary.csv")
        pd.DataFrame(roi_rows).to_csv(summary_csv, index=False)
        print(f"\n✅ 全部ROI处理完成！汇总表: {summary_csv}")
    else:
        print("\n⚠️ 没有任何 ROI 输出（可能 ROI 全空或筛选 label 不存在）")


if __name__ == "__main__":
    main()
