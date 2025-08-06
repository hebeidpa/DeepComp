import torch
import pandas as pd
import numpy as np
import joblib
from ft_transformer import FTTransformer

LABEL_COL = 'label'
META_COLS = ['', '']

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64

df = pd.read_excel(".xlsx", sheet_name="")

with open("model_feature_names.txt", "r") as f:
    feature_names = [line.strip() for line in f]
meta_names = META_COLS
ab_names = feature_names[len(meta_names):] 

X_meta = df[meta_names].values
X_ab = df[ab_names].values 

ab_selected_scaler = joblib.load("ab_selected_scaler.pkl")
X_ab_s = ab_selected_scaler.transform(X_ab)
X_all = np.hstack([X_meta, X_ab_s])  

# best_params： k_features/dim/gamma/lr

model = FTTransformer(
    categories=[],
    num_continuous=X_all.shape[1],
    dim=256,                 
    depth=4,
    heads=8,
    dim_head=32,             
    num_special_tokens=2,
    dim_out=1,
    ff_dropout=0.1,
    attn_dropout=0.2
).to(DEVICE)

model.load_state_dict(torch.load("fttransformer_best_model.pth", map_location=DEVICE))
model.eval()

X_tensor = torch.tensor(X_all.astype(np.float32)).to(DEVICE)
with torch.no_grad():
    logits = model(None, X_tensor).cpu().numpy().reshape(-1)
    probs = 1 / (1 + np.exp(-logits))  # sigmoid

if LABEL_COL in df.columns:
    from sklearn.metrics import roc_auc_score
    y_true = df[LABEL_COL].values
    auc = roc_auc_score(y_true, probs)
    print(f"Test set AUC: {auc:.4f}")

df_out = df.copy()
df_out["pred_prob"] = probs
df_out.to_csv("test_pred_result.csv", index=False)
print("test_pred_result.csv")
