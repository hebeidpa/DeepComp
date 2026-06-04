# VoxTell-L3 Body Composition Module

This folder contains the VoxTell-based L3 body composition module used in DeepComp.

The module performs L3-level body composition segmentation and quantification from abdominal CT NIfTI images. It uses VoxTell for L3 vertebra localization and coarse text-prompted segmentation, followed by HU-window refinement and morphology-based separation of skeletal muscle, visceral fat, subcutaneous fat, and intermuscular fat.

## Folder structure

~~~text
VoxTell-L3-BodyComp/
├── voxtell_l3_bodycomp/
│   ├── __init__.py
│   └── l3_body_composition.py
├── scripts/
│   └── install_into_voxtell.sh
├── examples/
│   ├── run_single_case.sh
│   ├── expected_output.json
│   └── expected_overlay.png
├── tests/
│   └── test_l3_pipeline_synthetic.py
└── sample_data/
    └── README.md
~~~

## Pipeline overview

The module performs the following steps:

1. Use VoxTell to localize the L3 vertebral level from abdominal CT.
2. Apply text-prompted segmentation to obtain coarse skeletal muscle and anatomical masks.
3. Refine skeletal muscle using HU thresholding.
4. Extract adipose tissue using HU thresholding.
5. Separate visceral fat and subcutaneous fat using a morphology-based abdominal wall envelope.
6. Export L3-level label maps, quantitative metrics, and visual overlays.

## Output files

For each input CT NIfTI file, the module exports:

~~~text
*_L3_bodycomp.nii.gz
*_L3_metrics.json
*_L3_metrics.csv
*_L3_overlay.png
~~~

The JSON and CSV files include:

~~~text
l3_slice_index
skeletal_muscle_area_cm2
visceral_fat_area_cm2
subcutaneous_fat_area_cm2
intermuscular_fat_area_cm2
total_fat_area_cm2
vat_sat_ratio
skeletal_muscle_mean_hu
skeletal_muscle_index
~~~

## Installation into VoxTell

This module is designed as an extension of VoxTell. After downloading or cloning VoxTell, install this module into the VoxTell source tree:

~~~bash
bash scripts/install_into_voxtell.sh /path/to/VoxTell
~~~

Then install VoxTell in your Python environment:

~~~bash
cd /path/to/VoxTell
pip install .
~~~

## Required external models

The model weights are not included in this repository. Please download the following models separately:

~~~text
VoxTell model weights: voxtell_v1.1
Text encoder: Qwen/Qwen3-Embedding-4B
~~~

For offline servers, the Qwen text encoder should be downloaded locally. The default text encoder path in VoxTell should be changed from:

~~~python
Qwen/Qwen3-Embedding-4B
~~~

to a local path, for example:

~~~python
/mnt/data/hiteam/models/text_encoder/Qwen3-Embedding-4B
~~~

Recommended offline environment variables:

~~~bash
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
~~~

## Example command

~~~bash
python -m voxtell.applications.l3_body_composition \
  -i "sample_data/sample_001.nii.gz" \
  -m "/path/to/voxtell_v1.1" \
  -o "outputs/sample_001" \
  --device cuda
~~~

If patient height is available, skeletal muscle index can be calculated by adding:

~~~bash
--height_m 1.72
~~~

## Expected output for the test case

~~~json
{
  "l3_slice_index": 178,
  "n_slices_averaged": 1,
  "spacing_zyx": [
    1.4116315841674805,
    0.8730469942092896,
    0.8730469942092896
  ],
  "skeletal_muscle_area_cm2": 83.23,
  "visceral_fat_area_cm2": 68.2,
  "subcutaneous_fat_area_cm2": 141.41,
  "intermuscular_fat_area_cm2": 61.11,
  "total_fat_area_cm2": 270.71,
  "vat_sat_ratio": 0.482,
  "skeletal_muscle_mean_hu": 35.9,
  "skeletal_muscle_index": null,
  "notes": []
}
~~~

## Synthetic test

A synthetic phantom test is provided to validate the model-independent post-processing components:

~~~bash
python VoxTell-L3-BodyComp/tests/test_l3_pipeline_synthetic.py
~~~

## Data privacy

Only fully anonymized sample NIfTI data should be included in this repository. Real patient imaging data should not be uploaded unless all identifying information has been removed and data sharing has been approved by the relevant institution or ethics committee.

## Disclaimer

This code is intended for research use. HU thresholds and morphology-based VAT/SAT separation should be validated on local CT protocols before clinical application.
