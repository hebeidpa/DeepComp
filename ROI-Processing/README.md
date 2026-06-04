# ROI-Processing

This folder provides ROI segmentation and processing code for the DeepComp project.

It contains two major components:

- `nnUNet/`: nnU-Net-based code for gastric tumor and peritumoral ROI segmentation.
- `L3-BodyComp/`: code for L3-level body composition segmentation, including skeletal muscle, subcutaneous fat, and visceral fat regions.

The segmented ROIs can be used for downstream CT feature extraction and image embedding generation in the DeepComp pipeline.

## Folder structure

```text
ROI-Processing/
├── nnUNet/
├── L3-BodyComp/
└── README.md
```

## 1. Gastric tumor and peritumoral ROI segmentation

The `nnUNet/` folder contains the nnU-Net-based segmentation code copied from `Gastric-nnUNet`.

Example command-line usage:

```bash
python ROI-Processing/nnUNet/inference.py \
    --input ./demo_cases/case_001/venous_CT.nii.gz \
    --output ./outputs/case_001/venous_tumor_peritumor_label.nii.gz
```

If the project is configured with nnU-Net v2, the inference command may also follow the standard nnU-Net format:

```bash
nnUNetv2_predict \
    -i ./demo_cases/case_001/venous_CT/ \
    -o ./outputs/case_001/nnunet_prediction/ \
    -d Dataset_GastricTumor \
    -c 3d_fullres \
    -f 0
```

Please adjust the dataset name, model folder, and checkpoint path according to the local nnU-Net configuration.

## 2. L3-level body composition segmentation

The `L3-BodyComp/` folder contains the L3-level body composition segmentation code copied from `VoxTell-L3-BodyComp`.

Example command-line usage:

```bash
python ROI-Processing/L3-BodyComp/inference.py \
    --input ./demo_cases/case_001/plain_CT.nii.gz \
    --output ./outputs/case_001/plain_L3_bodycomp_label.nii.gz
```

The output label is expected to include L3-level body composition regions, such as skeletal muscle, subcutaneous fat, and visceral fat.

## 3. Extract image feature embeddings

Download the pretrained MedGemma1.5 model and put it into:

```text
./TabM/weights/
```

Load the model:

```python
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

model_path = "./TabM/weights"
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
).eval()

vision = model.model.vision_tower
```

Use MedGemma1.5 to extract ROI-level image embeddings:

```python
import nibabel as nib
import numpy as np
import torch

ct = ensure_hwd(
    nib.load("./ct.nii.gz").get_fdata().astype(np.float32),
    slice_axis=2
)

roi = ensure_hwd(
    nib.load("./roi_mask.nii.gz").get_fdata(),
    slice_axis=2
)

roi_int = np.rint(roi).astype(np.int32)

for lab in sorted([int(x) for x in np.unique(roi_int) if x > 0]):
    mask = (roi_int == lab).astype(np.uint8)
    roi_slices = np.where(mask.sum(axis=(0, 1)) > 0)[0]

    images = [
        slice_to_rgb(
            crop_with_pad(
                ct[..., z],
                bbox_from_mask(mask[..., z]),
                pad=10
            )[0],
            mode="triple"
        )
        for z in roi_slices
    ]

    slice_feats = extract_feats_from_pil_images(
        images,
        image_processor,
        vision,
        model.device,
        dtype
    )

    roi_feat = torch.nn.functional.normalize(
        slice_feats.mean(dim=0),
        dim=-1
    )
```

## Notes

- Patient-level CT images and manual masks are not included in this folder.
- Demo cases are provided separately in `demo_cases/`.
- The command-line examples above should be adjusted according to the final script names and model paths.
- Large medical imaging files, such as `.nii.gz`, should be managed using Git LFS.
