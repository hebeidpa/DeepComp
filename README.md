## DeepComp model
An Interpretable Deep Learning Model for Preoperative Prediction of Postoperative Complications in Gastric Cancer 
## Overview
## This project mainly covers two core modules:
- GastricNN-UNet: A nnU-Net-based CT segmentation model for gastric medical imaging.
- TabM: A pipeline for constructing DeepComp and Prognostic Prediction models from medical data and gastric medical imaging.
## GastricNN-UNet
- GastricNN-UNet A customized implementation of the nnU-Net framework for 3D CT image segmentation focusing on gastric structures. 
- Key features:
  
      Preprocessing pipelines optimized for abdominal CT scans. 

      Configuration files for gastric organ/tumor segmentation. 

      Trained model weights (optional, contact for access).
## TabM
## Directory Structure
```plaintext
train.py
test.py
testOS.py
```
## Scripts

### 1. Training Script (`train.py`)
#### Functionality
Train the DeepComp classification model using 5-fold cross-validation to optimize model parameters and generate out-of-fold predictions.

Independently train the OS prognostic model using Cox loss to estimate patient-level survival risk scores.

Evaluate the classification model on the independent validation set using AUC and accuracy, and evaluate the OS prognostic model using the C-index and risk stratification performance.
#### Usage
```bash
python train.py --train .csv --test .csv --label label --os_time OS_time --os_event OS_event --train_os_model --tabm_path ./tabm.py --out_dir ./ --device cuda --amp --use_focal --fs_enable --fs_alpha  --fs_corr_th  --fs_max_vars  --fs_inner_folds 5 --fs_repeats 3 --fs_min_freq 0.6 --keep_features  
```
### 2. Testing Script (`test.py`)
#### Functionality
- Loads a trained model and evaluates it on the test dataset.
- Saves the test evaluation results.

#### Usage
```bash
python test.py
```
## Acknowledgements
 - We would like to express our sincere gratitude to all contributors and collaborators who supported this project. Special thanks to our research  - team members for their invaluable discussions and technical insights, which greatly enhanced the development and implementation of this work.

 - We also acknowledge the open-source community and developers of libraries such as PyTorch, and scikit-learn, whose efforts made it possible to build and refine our models efficiently. Finally, we are grateful to the institutions and organizations that provided the datasets and computational resources necessary for completing this project.

