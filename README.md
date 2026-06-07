# DeepComp model
A Multimodal Deep Learning Model for Preoperative Prediction of Postoperative Complications in Gastric Cancer

<img width="2769" height="909" alt="image" src="https://github.com/user-attachments/assets/9decca67-8f74-40a1-98ae-f637f7f27e69" />

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
2. Use medgemma-1.5 to extract image embeddings
```python
python  feature_extraction.py --nii venous_CT.nii.gz --roi venous_tumor_peritumor_label.nii.gz --out_dir out1  --k -1 --pad 10^C

```

```python
--nii ./Processing/CT/venous_CT.nii.gz	                        Path to CT file
```
```
--roi ./Processing/Label/venous_tumor_peritumor_label.nii.gz	Path to ROI mask file
```
```
--out_dir out1	                                                Save outputs to out1 folder
```
```
--k -1	                                                        Use all slices containing ROI (no sampling)
```
```
--model  ./Processing/weights/                                  MedGemma1.5 HuggingFace model path 
```
## Basic Usage: Predict Postoperative Complications Risk with DeepComp

### Model Download
The DeepComp model can be accessed from [here](https://drive.google.com/file/d/1Txp0wIdhqIBy1z_nvG98UPlx00wmPRKm/view?usp=drive_link).
1. Load the DeepComp model
```python

import torch
from test_tabm_model import read_table, load_prep_yaml, encode_numeric_from_yaml, load_model
device = "cuda" if torch.cuda.is_available() else "cpu"
data = read_table("./Usage/feature.csv")
X_num, _ = encode_numeric_from_yaml(data.drop(columns=["label"], errors="ignore"), load_prep_yaml("./final_prep.yaml"))
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
python3 test.py --data ./Evaluation/data.csv  --model_path ./Evaluation/final_best.pt --prep_yaml ./Evaluation/final_prep.yaml --tabm_path ./Evaluation/tabm.py --out_csv ./Evaluation/prediction.csv 
```
The AUC and accuracy will be printed to the screen, and the prediction results will be saved to ./Evaluation/prediction.csv.
```
AUC: 0.8934
Accuracy: 0.8829
Predictions saved to ./prediction.csv
```
### Acknowledgements
 - The project was built on many amazing repositories: [MedGemma1.5](https://huggingface.co/google/medgemma-1.5-4b-it), [nnU-Net](https://github.com/mic-dkfz/nnunet), [Tabm](https://github.com/yandex-research/tabm),and [VoxTell](https://github.com/MIC-DKFZ/VoxTell). We thank the authors and developers for their contributions.

 - We also acknowledge the open-source community and developers of libraries such as PyTorch, and scikit-learn, whose efforts made it possible to build and refine our models efficiently. Finally, we are grateful to the institutions and organizations that provided the datasets and computational resources necessary for completing this project.

