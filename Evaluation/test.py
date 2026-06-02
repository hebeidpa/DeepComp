import argparse
import importlib.util
import os

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, roc_auc_score


def load_data(path):
    return pd.read_csv(path) if path.endswith(".csv") else pd.read_excel(path)


def load_tabm(tabm_path):
    spec = importlib.util.spec_from_file_location("tabm", tabm_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TabM


def preprocess(df, prep):
    cols = prep["features_order"]
    z_cols, raw_cols = prep["z_features"], prep["raw_features"]
    z_median, z_mean, z_std = map(dict, [zip(z_cols, prep["z_median"]), zip(z_cols, prep["z_mean"]), zip(z_cols, prep["z_std"])])
    raw_median = dict(zip(raw_cols, prep["raw_median"]))
    X = np.zeros((len(df), len(cols)), dtype=np.float32)

    for i, c in enumerate(cols):
        v = pd.to_numeric(df[c], errors="coerce").to_numpy()
        if c in z_cols:
            v = np.where(np.isfinite(v), v, z_median[c])
            X[:, i] = (v - z_mean[c]) / (z_std[c] if z_std[c] > 1e-12 else 1.0)
        else:
            X[:, i] = np.where(np.isfinite(v), v, raw_median[c])

    return X


def load_model(model_path, tabm_path, n_features, device):
    ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
    cfg = ckpt["tabm_args"]
    TabM = load_tabm(tabm_path)

    model = TabM(
        n_num_features=n_features,
        cat_cardinalities=None,
        d_out=1,
        num_embeddings=None,
        k=cfg.get("k", 32),
        n_blocks=cfg.get("n_blocks", 4),
        d_block=cfg.get("d_block", 512),
        dropout=cfg.get("dropout", 0.05),
        activation=cfg.get("activation", "ReLU"),
        arch_type=cfg.get("arch_type", "tabm"),
        start_scaling_init=cfg.get("start_scaling_init", "random-signs")
        if cfg.get("arch_type", "tabm") != "tabm-packed" else None,
    ).to(device)

    model.load_state_dict(ckpt["state_dict"])
    return model.eval()


@torch.no_grad()
def predict(model, X, device):
    logits = model(torch.from_numpy(X).to(device), None)
    if logits.ndim == 3:
        logits = logits.mean(dim=1)
    return torch.sigmoid(logits.squeeze(-1)).cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--label", default=None)
    parser.add_argument("--model_path", default="./out_tabm_fs/models/final_best.pt")
    parser.add_argument("--prep_yaml", default="./out_tabm_fs/prep/final_prep.yaml")
    parser.add_argument("--tabm_path", default="./tabm.py")
    parser.add_argument("--out_csv", default="./prediction.csv")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    data = load_data(args.data)
    y = data[args.label].astype(int).to_numpy() if args.label else None
    X_df = data.drop(columns=[args.label]) if args.label else data

    with open(args.prep_yaml, "r", encoding="utf-8") as f:
        prep = yaml.safe_load(f)

    X = preprocess(X_df, prep)
    model = load_model(args.model_path, args.tabm_path, X.shape[1], device)

    prob = predict(model, X, device)
    pred = (prob >= args.threshold).astype(int)

    result = data.copy()
    result["y_prob"] = prob
    result["y_pred"] = pred
    result.to_csv(args.out_csv, index=False)

    if y is not None:
        print(f"AUC: {roc_auc_score(y, prob):.4f}")
        print(f"Accuracy: {accuracy_score(y, pred):.4f}")

    print(f"Predictions saved to {args.out_csv}")


if __name__ == "__main__":
    main()