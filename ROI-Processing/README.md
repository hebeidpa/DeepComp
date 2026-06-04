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

If the project is configured with nnU-Net v2, the inference command follow the standard nnU-Net format:

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
    --input ./*nii.gz \
    --output ./outputs/*.nii.gz
```

The output label is expected to include L3-level body composition regions, such as skeletal muscle, subcutaneous fat, and visceral fat.
