import shap
import torch
import pandas as pd
import numpy as np
import joblib
from ft_transformer import FTTransformer


LABEL_COL = 'label'
META_COLS = ['', '']

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


model = FTTransformer(
    categories=[],
    num_continuous=X_all.shape[1],
    dim=256,                 # 训练时dim
    depth=4,
    heads=8,
    dim_head=32,             # 训练时dim_head
    num_special_tokens=2,
    dim_out=1,
    ff_dropout=0.1,
    attn_dropout=0.2
)
model.load_state_dict(torch.load("fttransformer_best_model.pth", map_location="cpu"))
model.eval()


background = torch.from_numpy(X_all[:100].astype(np.float32))
to_explain = torch.from_numpy(X_all[:300].astype(np.float32))  

explainer = shap.DeepExplainer(lambda x: model(None, x), background)
shap_values = explainer.shap_values(to_explain)


shap.summary_plot(shap_values, X_all[:300], feature_names=feature_names, show=True)
shap.summary_plot(shap_values, X_all[:300], feature_names=feature_names, plot_type="bar", show=True)
top_feat = feature_names[np.abs(shap_values).mean(0).argmax()]
shap.dependence_plot(top_feat, shap_values, X_all[:300], feature_names=feature_names, show=True)
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[0], X_all[0], feature_names=feature_names, matplotlib=True, show=True)
shap.decision_plot(explainer.expected_value, shap_values, X_all[:300], feature_names=feature_names, show=True)
shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[0], X_all[0], feature_names)

np.save("shap_values_test.npy", shap_values)
pd.DataFrame(shap_values, columns=feature_names).to_csv("shap_values_test.csv", index=False)

