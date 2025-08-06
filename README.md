## DeepComp model
An Interpretable Deep Learning Model for Preoperative Prediction of Postoperative Complications in Gastric Cancer 
## Overview
## This project mainly covers two core modules:
-GastricNN-UNet: A nnU-Net-based CT segmentation model for gastric medical imaging.
-FT-transformer: A pipeline for constructing DeepComp models from medical data and gastric medical imaging.
## GastricNN-UNet
GastricNN-UNet A customized implementation of the nnU-Net framework for 3D CT image segmentation focusing on gastric structures. Key features:
Preprocessing pipelines optimized for abdominal CT scans
Configuration files for gastric organ/tumor segmentation
Trained model weights (optional, contact for access)
## FT-transformer
## Directory Structure
```plaintext
train.py
test.py
```
## Scripts

### 1. Training Script (`train.py`)
#### Functionality
Train the predictive model using 5-fold cross-validation to determine the optimal parameters.
Evaluate the model performance on the final validation set.
#### Usage
```bash
python train.py
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

