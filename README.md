# DeepComp model
An Interpretable Deep Learning Model for Preoperative Prediction of Postoperative Complications in Gastric Cancer 
<img width="1467" height="614" alt="aaa" src="https://github.com/user-attachments/assets/c0a287a2-17ad-4cf4-9136-29a9dbd3d984" />
## Pre-requisites
All experiments are run on a machine with
- 1 × NVIDIA H20 GPU (96 GB HBM3)
- Python (Python 3.11.0) and PyTorch (torch==2.5.1)
## Installation
1. Install Anaconda
2. Clone this reposity and cd into the directory:
```plaintext
git clone https://github.com/hebeidpa/DeepComp.git
cd DeepComp
```
3. Create a new environment and install dependencies:
```plaintext
conda create -n DeepComp python=3.11 -y --no-default-packages
conda activate DeepComp
pip install --upgrade pip
pip install -r requirements.txt
```

## CT Image Processing Pipeline
Extract Image Feature Embeddings
1. Download the pretrained [MedGemma1.5](https://huggingface.co/google/medgemma-1.5-4b-it) , put it to ./Processing/weights/ and load the model  
```python
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
model_path = "./Processing/weights"
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForImageTextToText.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto").eval()
vision = model.model.vision_tower
```
2. Use medgemma-1.5 to extract image embeddings
```python
import nibabel as nib
import numpy as np
ct = ensure_hwd(nib.load("./ct.nii.gz").get_fdata().astype(np.float32), slice_axis=2)
roi = ensure_hwd(nib.load("./roi_mask.nii.gz").get_fdata(), slice_axis=2)
roi_int = np.rint(roi).astype(np.int32)
for lab in sorted([int(x) for x in np.unique(roi_int) if x > 0]):
    mask = (roi_int == lab).astype(np.uint8)
    roi_slices = np.where(mask.sum(axis=(0, 1)) > 0)[0]
    images = [slice_to_rgb(crop_with_pad(ct[..., z], bbox_from_mask(mask[..., z]), pad=10)[0], mode="triple") for z in roi_slices]
    slice_feats = extract_feats_from_pil_images(images, image_processor, vision, model.device, dtype)
    roi_feat = torch.nn.functional.normalize(slice_feats.mean(dim=0), dim=-1)  # ROI embedding
```
## Basic Usage: Predict Postoperative Complications Risk with DeepComp

### Model Download
The DeepComp model can be accessed from [here](https://drive.google.com/file/d/1Txp0wIdhqIBy1z_nvG98UPlx00wmPRKm/view?usp=drive_link).
1. Load the DeepComp model
```python

import torch
from test_tabm_model import read_table, load_prep_yaml, encode_numeric_from_yaml, load_model
device = "cuda" if torch.cuda.is_available() else "cpu"
data = read_table("./Usage/data.csv")
X_num, _ = encode_numeric_from_yaml(data.drop(columns=["label"], errors="ignore"), load_prep_yaml("./DeepComp_prep.yaml"))
model, _ = load_model("./Usage/final_best.pt", "./Usage/tabm.py", X_num.shape[1], device)
```
2. Predict Patient Postoperative Complications Risk.
```python
from test_tabm_model import predict_proba
prob = predict_proba(model, X_num, device)
pred = (prob >= 0.5).astype(int)
print("Complication risk:", prob)
print("Prediction:", pred)
```
## Evaluation
To reproduce the results in our paper, we provide a reproducible result on IVC-I dataset
- First download our processed IVC-I frozen features [here](https://drive.google.com/file/d/1aszWi0EFluhO3AaJcq-f2MW8SSXJrr6q/view?usp=drive_link)
- Put the extracted features to ./Evaluation/
- Run the following command:
```python
python3 test.py --data ./Evaluation/test.csv  --model_path ./Evaluation/final_best.pt --prep_yaml ./Evaluation/final_prep.yaml --tabm_path ./Evaluation/tabm.py --out_csv ./Evaluation/prediction.csv 
```
The AUC and accuracy will be printed to the screen, and the prediction results will be saved to ./Evaluation/prediction.csv.
```
AUC: 0.8934
Accuracy: 0.8829
Predictions saved to ./prediction.csv
```
### Acknowledgements
 - We would like to express our sincere gratitude to all contributors and collaborators who supported this project. Special thanks to our research  - team members for their invaluable discussions and technical insights, which greatly enhanced the development and implementation of this work.

 - We also acknowledge the open-source community and developers of libraries such as PyTorch, and scikit-learn, whose efforts made it possible to build and refine our models efficiently. Finally, we are grateful to the institutions and organizations that provided the datasets and computational resources necessary for completing this project.

