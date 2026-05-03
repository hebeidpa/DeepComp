import os
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
from PIL import Image
import glob
from tqdm import tqdm
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

try:
    from scipy.ndimage import distance_transform_edt
except ImportError:
    distance_transform_edt = None


LOCAL_MODEL_PATH = "./ct"


def nifti_stem(path: str) -> str:
    base = os.path.basename(path)
    lower = base.lower()
    if lower.endswith(".nii.gz"):
        return base[:-7]
    if lower.endswith(".nii"):
        return base[:-4]
    return os.path.splitext(base)[0]


def format_mm(mm: float) -> str:
    if float(mm).is_integer():
        return str(int(mm))
    return str(mm).replace(".", "p")


def window_ct(slice_hu: np.ndarray, wl: float, ww: float) -> np.ndarray:
    low = wl - ww / 2.0
    high = wl + ww / 2.0
    x = np.clip(slice_hu, low, high)
    x = (x - low) / (high - low + 1e-6)
    return x


def slice_to_rgb(
    slice_hu: np.ndarray,
    mode: str = "triple",
    mask: np.ndarray = None,
    crop_to_region: bool = True
) -> Image.Image:

    if mode == "gray":
        wl, ww = -600, 1500
        x = (window_ct(slice_hu, wl, ww) * 255.0).astype(np.uint8)
        rgb = np.stack([x, x, x], axis=-1)

    elif mode == "triple":
        lung = (window_ct(slice_hu, -600, 1500) * 255.0).astype(np.uint8)
        medi = (window_ct(slice_hu, 40, 400) * 255.0).astype(np.uint8)
        bone = (window_ct(slice_hu, 300, 1500) * 255.0).astype(np.uint8)
        rgb = np.stack([lung, medi, bone], axis=-1)

    else:
        raise ValueError(f"Unknown mode={mode}. Use gray or triple.")

    if mask is not None:
        mask = mask.astype(bool)

        if mask.shape != slice_hu.shape:
            raise ValueError(f"mask shape {mask.shape} != slice shape {slice_hu.shape}")

        if mask.sum() == 0:
            raise ValueError("Empty mask slice.")

        rgb[~mask] = 0

        if crop_to_region:
            ys, xs = np.where(mask)
            y1, y2 = ys.min(), ys.max()
            x1, x2 = xs.min(), xs.max()
            rgb = rgb[y1:y2 + 1, x1:x2 + 1, :]

    return Image.fromarray(rgb, mode="RGB")


def ensure_hwd(vol: np.ndarray, slice_axis: int) -> np.ndarray:
    if slice_axis == 2:
        return vol
    if slice_axis == 0:
        return np.transpose(vol, (1, 2, 0))
    if slice_axis == 1:
        return np.transpose(vol, (0, 2, 1))
    raise ValueError("slice_axis must be 0, 1, or 2")


def spacing_to_hwd(zooms, slice_axis: int):
    zooms = tuple(float(z) for z in zooms[:3])

    if slice_axis == 2:
        return zooms
    if slice_axis == 0:
        return zooms[1], zooms[2], zooms[0]
    if slice_axis == 1:
        return zooms[0], zooms[2], zooms[1]

    raise ValueError("slice_axis must be 0, 1, or 2")


def find_nifti_files(input_path, recursive=True):
    nifti_files = []

    if os.path.isfile(input_path):
        if input_path.lower().endswith((".nii", ".nii.gz")):
            nifti_files.append(input_path)

    elif os.path.isdir(input_path):
        patterns = ["**/*.nii", "**/*.nii.gz"] if recursive else ["*.nii", "*.nii.gz"]
        for pattern in patterns:
            files = glob.glob(os.path.join(input_path, pattern), recursive=recursive)
            nifti_files.extend(files)

    return sorted(list(set(nifti_files)))


def normalize_roi_name(name: str) -> str:
    name = name.lower()

    suffixes = [
        "_roi", "-roi",
        "_mask", "-mask",
        "_seg", "-seg",
        "_label", "-label",
        "_lesion", "-lesion"
    ]

    for suf in suffixes:
        if name.endswith(suf):
            name = name[:-len(suf)]

    return name


def build_roi_map(roi_input, recursive=True):
    if roi_input is None:
        return {}

    roi_files = find_nifti_files(roi_input, recursive=recursive)

    roi_map = {}
    for p in roi_files:
        stem = nifti_stem(p)
        roi_map[stem.lower()] = p
        roi_map[normalize_roi_name(stem)] = p

    return roi_map


def find_matching_roi(ct_path, roi_input, roi_map):
    if roi_input is None:
        return None

    if os.path.isfile(roi_input):
        return roi_input

    ct_stem = nifti_stem(ct_path)
    ct_key = ct_stem.lower()
    ct_norm = normalize_roi_name(ct_stem)

    if ct_key in roi_map:
        return roi_map[ct_key]

    if ct_norm in roi_map:
        return roi_map[ct_norm]

    candidates = []
    for k, v in roi_map.items():
        if ct_norm in k or k in ct_norm:
            candidates.append(v)

    candidates = sorted(list(set(candidates)))

    if len(candidates) == 1:
        return candidates[0]

    return None


def load_roi_mask(roi_path, slice_axis, threshold=0.5):
    roi_nii = nib.load(roi_path)
    roi = roi_nii.get_fdata().astype(np.float32)

    if roi.ndim == 4:
        roi = roi[..., 0]

    if roi.ndim != 3:
        raise ValueError(f"ROI is not 3D: shape={roi.shape}")

    roi = ensure_hwd(roi, slice_axis)
    roi_mask = roi > threshold

    return roi_mask


def build_region_mask(roi_mask, region, spacing_hwd, peri_mm=None):
    roi_mask = roi_mask.astype(bool)

    if region == "roi":
        return roi_mask

    if region == "peri":
        if distance_transform_edt is None:
            raise ImportError("scipy is required for peri-region extraction.")

        if peri_mm is None or peri_mm <= 0:
            raise ValueError("When region=peri, --peri_mm must be greater than 0.")

        dist_outside = distance_transform_edt(~roi_mask, sampling=spacing_hwd)
        dilated = dist_outside <= float(peri_mm)
        peri_mask = dilated & (~roi_mask)

        return peri_mask

    raise ValueError(f"Unknown region={region}")


def get_vision_model(model):
    possible_names = ["vision_model", "vision_tower", "vision_encoder", "vision"]

    for name in possible_names:
        if hasattr(model, name):
            return getattr(model, name)

    if hasattr(model, "model"):
        for name in possible_names:
            if hasattr(model.model, name):
                return getattr(model.model, name)

    print("\nModel structure debug information:")
    print(f"Model type: {type(model)}")
    print("Model attributes:")
    for attr in dir(model):
        if not attr.startswith("_"):
            print(f"  - {attr}")

    if hasattr(model, "model"):
        print("\nmodel.model attributes:")
        for attr in dir(model.model):
            if not attr.startswith("_"):
                print(f"  - {attr}")

    raise RuntimeError("Vision encoder not found.")


def make_output_prefix(base, region, peri_mm=None):
    if region == "whole":
        return base

    if region == "roi":
        return f"{base}_roi"

    if region == "peri":
        return f"{base}_peri{format_mm(peri_mm)}mm"

    return base


def extract_all_slices_from_nifti(
    nii_path,
    args,
    image_processor,
    vision,
    device,
    dtype,
    region="whole",
    roi_path=None,
    peri_mm=None
):
    try:
        print(f"\nProcessing: {os.path.basename(nii_path)} | region={region}", end="")
        if region == "peri":
            print(f" | peri_mm={peri_mm}")
        else:
            print("")

        nii = nib.load(nii_path)
        vol = nii.get_fdata().astype(np.float32)

        if vol.ndim == 4:
            vol = vol[..., 0]

        if vol.ndim != 3:
            print(f"  Skip: not a 3D volume. shape={vol.shape}")
            return None

        spacing_hwd = spacing_to_hwd(nii.header.get_zooms(), args.slice_axis)

        vol = ensure_hwd(vol, args.slice_axis)
        H, W, D = vol.shape

        print(f"  Volume shape: {H}x{W}x{D}")
        print(f"  Spacing HWD: {spacing_hwd}")

        region_mask = None

        if region in ["roi", "peri"]:
            if roi_path is None:
                print("  ROI file not provided.")
                return None

            print(f"  ROI: {os.path.basename(roi_path)}")

            roi_mask = load_roi_mask(
                roi_path,
                slice_axis=args.slice_axis,
                threshold=args.roi_threshold
            )

            if roi_mask.shape != vol.shape:
                print(f"  ROI shape {roi_mask.shape} does not match CT shape {vol.shape}.")
                return None

            if roi_mask.sum() == 0:
                print("  Empty ROI.")
                return None

            region_mask = build_region_mask(
                roi_mask=roi_mask,
                region=region,
                spacing_hwd=spacing_hwd,
                peri_mm=peri_mm
            )

            if region_mask.sum() == 0:
                print("  Empty extraction region.")
                return None

            print(f"  ROI voxels: {int(roi_mask.sum())}")
            print(f"  Extraction voxels: {int(region_mask.sum())}")

            slice_pixel_counts = region_mask.sum(axis=(0, 1))
            selected_slices = np.where(slice_pixel_counts >= args.min_region_pixels)[0]

            if len(selected_slices) == 0:
                print(f"  No slices meet min_region_pixels={args.min_region_pixels}.")
                return None

            print(f"  Selected slices: {len(selected_slices)}/{D}")

        else:
            selected_slices = np.arange(D)
            print(f"  Selected slices: {len(selected_slices)}/{D}")

        all_slice_features = []
        used_slice_indices = []

        batch_size = args.batch_size

        for start_pos in range(0, len(selected_slices), batch_size):
            batch_slice_indices = selected_slices[start_pos:start_pos + batch_size]

            print(
                f"  Batch slices: "
                f"{start_pos + 1}-{start_pos + len(batch_slice_indices)}/{len(selected_slices)}"
            )

            pil_images = []
            valid_indices = []

            for i in batch_slice_indices:
                try:
                    mask_slice = None

                    if region_mask is not None:
                        mask_slice = region_mask[..., i]
                        if mask_slice.sum() < args.min_region_pixels:
                            continue

                    img = slice_to_rgb(
                        vol[..., i],
                        mode=args.rgb_mode,
                        mask=mask_slice,
                        crop_to_region=args.crop_to_region
                    )

                    pil_images.append(img)
                    valid_indices.append(int(i))

                except Exception as e:
                    print(f"    Slice {i} failed: {e}")
                    continue

            if not pil_images:
                continue

            all_pixel_values = []

            for img in pil_images:
                try:
                    inputs = image_processor(images=img, return_tensors="pt")
                    pixel_values = inputs["pixel_values"]
                    all_pixel_values.append(pixel_values)

                except Exception:
                    try:
                        img_resized = img.resize((224, 224))
                        img_array = np.array(img_resized).transpose(2, 0, 1)
                        img_tensor = torch.from_numpy(img_array).unsqueeze(0)
                        img_tensor = img_tensor.float() / 255.0
                        all_pixel_values.append(img_tensor)

                    except Exception as e:
                        print(f"    Image processing failed: {e}")
                        continue

            if not all_pixel_values:
                continue

            pixel_values = torch.cat(all_pixel_values, dim=0)
            pixel_values = pixel_values.to(device, dtype=dtype)

            with torch.no_grad():
                vout = vision(pixel_values=pixel_values, return_dict=True)
                patch_feats = vout.last_hidden_state
                slice_feats = patch_feats.mean(dim=1)
                slice_feats = torch.nn.functional.normalize(slice_feats, dim=-1)

            all_slice_features.append(slice_feats.detach().cpu().numpy())
            used_slice_indices.extend(valid_indices)

        if not all_slice_features:
            print("  All slices failed.")
            return None

        all_slice_features = np.vstack(all_slice_features)

        vol_feat = all_slice_features.mean(axis=0)
        norm = np.linalg.norm(vol_feat)

        if norm > 0:
            vol_feat = vol_feat / norm

        result = {
            "filename": os.path.basename(nii_path),
            "base_name": nifti_stem(nii_path),
            "roi_file": os.path.basename(roi_path) if roi_path else "",
            "region": region,
            "peri_mm": peri_mm if peri_mm is not None else "",
            "volume_feature": vol_feat,
            "all_slice_features": all_slice_features,
            "slice_indices": np.array(used_slice_indices, dtype=int),
            "total_slices": D,
            "used_slices": len(used_slice_indices),
            "shape": vol.shape,
            "spacing_hwd": spacing_hwd,
            "feature_dim": vol_feat.shape[0]
        }

        print(f"  Done. Used slices: {len(used_slice_indices)}, feature shape: {all_slice_features.shape}")

        return result

    except Exception as e:
        print(f"  Failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_result(result, args, region, peri_mm=None):
    base = result["base_name"]
    prefix = make_output_prefix(base, region, peri_mm)
    out_prefix = os.path.join(args.out_dir, prefix)

    all_slices_npy = f"{out_prefix}_all_slices.npy"
    np.save(all_slices_npy, result["all_slice_features"])

    slice_indices_npy = f"{out_prefix}_slice_indices.npy"
    np.save(slice_indices_npy, result["slice_indices"])

    if len(result["all_slice_features"]) < 100:
        all_slices_csv = f"{out_prefix}_all_slices.csv"

        slice_df = pd.DataFrame(
            result["all_slice_features"],
            columns=[f"f{i}" for i in range(result["all_slice_features"].shape[1])]
        )

        slice_df.insert(0, "slice_index", result["slice_indices"])
        slice_df.insert(0, "peri_mm", result["peri_mm"])
        slice_df.insert(0, "region", result["region"])
        slice_df.insert(0, "filename", result["filename"])
        slice_df.to_csv(all_slices_csv, index=False)

    vol_npy = f"{out_prefix}_vol_feat.npy"
    np.save(vol_npy, result["volume_feature"])

    vol_csv = f"{out_prefix}_vol_feat.csv"

    vol_df = pd.DataFrame(
        [result["volume_feature"]],
        columns=[f"f{i}" for i in range(result["volume_feature"].shape[0])]
    )

    vol_df.insert(0, "filename", result["filename"])
    vol_df.insert(1, "roi_file", result["roi_file"])
    vol_df.insert(2, "region", result["region"])
    vol_df.insert(3, "peri_mm", result["peri_mm"])
    vol_df.insert(4, "shape", str(result["shape"]))
    vol_df.insert(5, "spacing_hwd", str(result["spacing_hwd"]))
    vol_df.insert(6, "total_slices", result["total_slices"])
    vol_df.insert(7, "used_slices", result["used_slices"])
    vol_df.insert(8, "feature_dim", result["feature_dim"])

    vol_df.to_csv(vol_csv, index=False)


def main():
    ap = argparse.ArgumentParser(description="Extract NIfTI image features.")

    ap.add_argument("--input", required=True, help="Input NIfTI file or directory.")
    ap.add_argument("--out_dir", required=True, help="Output directory.")
    ap.add_argument("--model", default=LOCAL_MODEL_PATH, help="Model path.")

    ap.add_argument("--slice_axis", type=int, default=2, help="Slice axis: 0, 1, or 2.")
    ap.add_argument("--rgb_mode", choices=["gray", "triple"], default="triple", help="CT window mode.")
    ap.add_argument("--recursive", action="store_true", help="Search subdirectories.")
    ap.add_argument("--skip_existing", action="store_true", help="Skip existing results.")
    ap.add_argument("--batch_size", type=int, default=4, help="Batch size.")

    ap.add_argument("--roi", default=None, help="ROI mask file or directory.")
    ap.add_argument(
        "--region",
        nargs="+",
        choices=["whole", "roi", "peri"],
        default=["whole"],
        help="Extraction region."
    )

    ap.add_argument(
        "--peri_mm",
        nargs="+",
        type=float,
        default=[3.0],
        help="Peritumoral distance in mm."
    )

    ap.add_argument(
        "--roi_threshold",
        type=float,
        default=0.5,
        help="ROI binarization threshold."
    )

    ap.add_argument(
        "--min_region_pixels",
        type=int,
        default=10,
        help="Minimum region pixels per slice."
    )

    ap.add_argument(
        "--crop_to_region",
        dest="crop_to_region",
        action="store_true",
        default=True,
        help="Crop to region bounding box."
    )

    ap.add_argument(
        "--no_crop_to_region",
        dest="crop_to_region",
        action="store_false",
        help="Disable region cropping."
    )

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 80)
    print("NIfTI feature extraction")
    print("=" * 80)
    print(f"Regions: {args.region}")

    if "peri" in args.region and distance_transform_edt is None:
        raise ImportError("scipy is required for --region peri.")

    need_roi = any(r in ["roi", "peri"] for r in args.region)

    if need_roi and args.roi is None:
        raise ValueError("--roi is required when using --region roi or --region peri.")

    nifti_files = find_nifti_files(args.input, args.recursive)

    if not nifti_files:
        print(f"No NIfTI files found: {args.input}")
        return

    print(f"Found CT files: {len(nifti_files)}")

    roi_map = {}

    if need_roi:
        roi_map = build_roi_map(args.roi, recursive=args.recursive)

        if not roi_map and not os.path.isfile(args.roi):
            print(f"No ROI files found: {args.roi}")
            return

        print(f"Found ROI files: {len(set(roi_map.values()))}")

    device0 = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device0 == "cuda" else torch.float32

    print(f"Device: {device0}")
    print(f"Dtype: {dtype}")

    print(f"\nLoading model: {args.model}")

    processor = AutoProcessor.from_pretrained(args.model)
    image_processor = processor.image_processor if hasattr(processor, "image_processor") else processor

    try:
        model = AutoModelForImageTextToText.from_pretrained(
            args.model,
            dtype=dtype,
            device_map="auto" if device0 == "cuda" else None,
        ).eval()
    except TypeError:
        model = AutoModelForImageTextToText.from_pretrained(
            args.model,
            torch_dtype=dtype,
            device_map="auto" if device0 == "cuda" else None,
        ).eval()

    print("Model loaded.")

    vision = get_vision_model(model)
    vision_device = next(vision.parameters()).device

    print(f"Vision model: {type(vision)}")
    print(f"Vision device: {vision_device}")

    successful = 0
    failed = 0
    all_results = []

    for nii_path in tqdm(nifti_files, desc="Progress"):
        roi_path = None

        if need_roi:
            roi_path = find_matching_roi(nii_path, args.roi, roi_map)

            if roi_path is None:
                print(f"\nROI not found for: {os.path.basename(nii_path)}")
                failed += 1
                continue

        jobs = []

        if "whole" in args.region:
            jobs.append(("whole", None))

        if "roi" in args.region:
            jobs.append(("roi", None))

        if "peri" in args.region:
            for mm in args.peri_mm:
                jobs.append(("peri", mm))

        for region, peri_mm in jobs:
            base = nifti_stem(nii_path)
            prefix = make_output_prefix(base, region, peri_mm)
            out_file = os.path.join(args.out_dir, f"{prefix}_vol_feat.npy")

            if args.skip_existing and os.path.exists(out_file):
                print(f"\nSkip existing: {prefix}")
                continue

            result = extract_all_slices_from_nifti(
                nii_path=nii_path,
                args=args,
                image_processor=image_processor,
                vision=vision,
                device=vision_device,
                dtype=dtype,
                region=region,
                roi_path=roi_path,
                peri_mm=peri_mm
            )

            if result is not None:
                save_result(result, args, region, peri_mm)
                all_results.append(result)
                successful += 1
            else:
                failed += 1

    if all_results:
        print("\nGenerating summary files...")

        all_vol_features = np.array([r["volume_feature"] for r in all_results])
        summary_npy = os.path.join(args.out_dir, "all_vol_features.npy")
        np.save(summary_npy, all_vol_features)

        print(f"Saved: {summary_npy}, shape={all_vol_features.shape}")

        summary_csv = os.path.join(args.out_dir, "all_vol_features.csv")

        summary_data = []

        for r in all_results:
            row_data = {
                "filename": r["filename"],
                "roi_file": r["roi_file"],
                "region": r["region"],
                "peri_mm": r["peri_mm"],
                "shape": str(r["shape"]),
                "spacing_hwd": str(r["spacing_hwd"]),
                "total_slices": r["total_slices"],
                "used_slices": r["used_slices"],
                "feature_dim": r["feature_dim"]
            }

            for i in range(r["feature_dim"]):
                row_data[f"f{i}"] = r["volume_feature"][i]

            summary_data.append(row_data)

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(summary_csv, index=False)

        print(f"Saved: {summary_csv}, shape={summary_df.shape}")

        stats_csv = os.path.join(args.out_dir, "summary_statistics.csv")
        stats_data = []

        for r in all_results:
            stats_data.append({
                "filename": r["filename"],
                "roi_file": r["roi_file"],
                "region": r["region"],
                "peri_mm": r["peri_mm"],
                "height": r["shape"][0],
                "width": r["shape"][1],
                "depth": r["shape"][2],
                "total_slices": r["total_slices"],
                "used_slices": r["used_slices"],
                "feature_dim": r["feature_dim"],
                "feature_mean": float(np.mean(r["volume_feature"])),
                "feature_std": float(np.std(r["volume_feature"])),
                "feature_min": float(np.min(r["volume_feature"])),
                "feature_max": float(np.max(r["volume_feature"]))
            })

        stats_df = pd.DataFrame(stats_data)
        stats_df.to_csv(stats_csv, index=False)

        print(f"Saved: {stats_csv}")

        readme_file = os.path.join(args.out_dir, "README.txt")

        with open(readme_file, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("MedGemma NIfTI feature extraction results\n")
            f.write("=" * 60 + "\n\n")
            f.write("Supported regions:\n")
            f.write("1. whole: whole CT volume\n")
            f.write("2. roi: ROI region only\n")
            f.write("3. peri: peritumoral ring region only\n\n")
            f.write("Files:\n")
            f.write("1. *_vol_feat.npy\n")
            f.write("2. *_vol_feat.csv\n")
            f.write("3. *_all_slices.npy\n")
            f.write("4. *_slice_indices.npy\n")
            f.write("5. all_vol_features.npy\n")
            f.write("6. all_vol_features.csv\n")
            f.write("7. summary_statistics.csv\n\n")
            f.write("Naming:\n")
            f.write("whole: patient001_vol_feat.npy\n")
            f.write("roi: patient001_roi_vol_feat.npy\n")
            f.write("peri: patient001_peri3mm_vol_feat.npy\n\n")
            f.write(f"Number of results: {len(all_results)}\n")
            f.write(f"Time: {pd.Timestamp.now()}\n")

    print("\n" + "=" * 80)
    print("Finished.")
    print(f"Successful results: {successful}")
    print(f"Failed results: {failed}")
    print(f"Output directory: {args.out_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()