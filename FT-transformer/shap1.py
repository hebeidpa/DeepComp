import shap
import torch
import pandas as pd
import numpy as np
import joblib
from ft_transformer import FTTransformer

# ===== 1. 加载特征名和数据（区分meta和ab部分） =====
LABEL_COL = 'label'
META_COLS = ['CCI', '手术评分']

df = pd.read_excel("合并数据集.xlsx", sheet_name="测试集")
with open("model_feature_names.txt", "r") as f:
    feature_names = [line.strip() for line in f]
meta_names = META_COLS
ab_names = feature_names[len(meta_names):]

# ==== 2. 数据处理：meta保持原始，ab做Z标准化 ====
X_meta = df[meta_names].values
X_ab = df[ab_names].values
ab_selected_scaler = joblib.load("ab_selected_scaler.pkl")
X_ab_s = ab_selected_scaler.transform(X_ab)
X_all = np.hstack([X_meta, X_ab_s])   # 顺序和训练完全一致

# ==== 3. 加载模型权重（参数同训练） ====
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

# ==== 4. SHAP解释，背景数据和解释对象都要按上面特征处理 ====
background = torch.from_numpy(X_all[:100].astype(np.float32))
to_explain = torch.from_numpy(X_all[:300].astype(np.float32))   # 可以换成全部

# ----------- 核心解释器（标准官方用法，不要用deep_pytorch）-------------
explainer = shap.DeepExplainer(lambda x: model(None, x), background)
shap_values = explainer.shap_values(to_explain)

# ==== 5. 画summary图、bar图、dependence图、force图等主流解释 ====
shap.summary_plot(shap_values, X_all[:300], feature_names=feature_names, show=True)
shap.summary_plot(shap_values, X_all[:300], feature_names=feature_names, plot_type="bar", show=True)
top_feat = feature_names[np.abs(shap_values).mean(0).argmax()]
shap.dependence_plot(top_feat, shap_values, X_all[:300], feature_names=feature_names, show=True)
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[0], X_all[0], feature_names=feature_names, matplotlib=True, show=True)
shap.decision_plot(explainer.expected_value, shap_values, X_all[:300], feature_names=feature_names, show=True)
shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[0], X_all[0], feature_names)

# ==== 6. 保存shap值、可导出csv ====
np.save("shap_values_test.npy", shap_values)
pd.DataFrame(shap_values, columns=feature_names).to_csv("shap_values_test.csv", index=False)
