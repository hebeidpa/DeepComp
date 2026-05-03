# -*- coding: utf-8 -*-

import os
import sys
import re
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import shap


def resolve_path(path: str) -> str:
    if os.path.exists(path):
        return path
    if not os.path.isabs(path):
        p2 = os.path.abspath(path)
        if os.path.exists(p2):
            return p2
    raise FileNotFoundError(f"File not found: {path}")


def read_table(path: str, sheet_name=None) -> pd.DataFrame:
    path = resolve_path(path)
    ext = os.path.splitext(path)[1].lower()

    if ext == ".csv":
        return pd.read_csv(path, encoding="utf-8-sig")

    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path, sheet_name=sheet_name)

    raise ValueError(f"Unsupported file format: {ext}")


def load_tabm_class(tabm_path: str):
    import types
    import typing as T

    tabm_path = resolve_path(tabm_path)
    src = open(tabm_path, "r", encoding="utf-8").read()

    src = re.sub(r"\blist\[(.*?)\]", r"List[\1]", src)
    src = re.sub(r"\bdict\[(.*?)\]", r"Dict[\1]", src)
    src = re.sub(r"\btuple\[(.*?)\]", r"Tuple[\1]", src)
    src = re.sub(r"\bset\[(.*?)\]", r"Set[\1]", src)

    if not src.lstrip().startswith("from __future__ import annotations"):
        src = "from __future__ import annotations\n" + src

    module_name = "tabm_shap_mod"
    mod = types.ModuleType(module_name)
    mod.__file__ = tabm_path
    mod.__package__ = ""
    mod.__dict__.update({
        "List": T.List,
        "Dict": T.Dict,
        "Tuple": T.Tuple,
        "Set": T.Set,
        "Optional": T.Optional,
        "Union": T.Union,
        "Any": T.Any,
    })

    sys.modules[module_name] = mod
    exec(compile(src, tabm_path, "exec"), mod.__dict__)

    for name in ["TabM", "TABM", "Model", "Tabm", "tabm"]:
        if hasattr(mod, name):
            obj = getattr(mod, name)
            if isinstance(obj, type):
                return obj

    raise RuntimeError("TabM class not found in tabm.py.")


def load_prep_npz(prep_npz_path: str):
    data = np.load(resolve_path(prep_npz_path), allow_pickle=True)

    features_order = list(data["features_order"].astype(object))
    z_features = list(data["z_features"].astype(object))
    raw_features = list(data["raw_features"].astype(object))

    z_median = data["z_median"].astype(np.float32)
    z_mean = data["z_mean"].astype(np.float32)
    z_std = data["z_std"].astype(np.float32)
    raw_median = data["raw_median"].astype(np.float32)

    z_stats = {
        f: (float(med), float(mu), float(sd))
        for f, med, mu, sd in zip(z_features, z_median, z_mean, z_std)
    }

    raw_stats = {
        f: float(med)
        for f, med in zip(raw_features, raw_median)
    }

    return {
        "features_order": features_order,
        "z_features": z_features,
        "raw_features": raw_features,
        "z_stats": z_stats,
        "raw_stats": raw_stats,
    }


def encode_mixed_from_prep(df: pd.DataFrame, prep: dict):
    cols = prep["features_order"]
    z_stats = prep["z_stats"]
    raw_stats = prep["raw_stats"]

    Xn = np.zeros((len(df), len(cols)), dtype=np.float32)

    for j, c in enumerate(cols):
        if c in df.columns:
            v = pd.to_numeric(df[c], errors="coerce").to_numpy()
        else:
            v = np.full(len(df), np.nan)

        if c in z_stats:
            med, mu, sd = z_stats[c]
            v = np.where(np.isfinite(v), v, med)
            v = (v - mu) / (sd if sd != 0 else 1.0)

        elif c in raw_stats:
            med = raw_stats[c]
            v = np.where(np.isfinite(v), v, med)

        else:
            v = np.where(np.isfinite(v), v, 0.0)

        Xn[:, j] = v.astype(np.float32)

    return Xn


@torch.no_grad()
def predict_proba_torch(model, x_num, device, batch_size=4096):
    model.eval()
    n = len(x_num)
    probs = np.zeros((n,), dtype=np.float32)

    for i in range(0, n, batch_size):
        j = min(n, i + batch_size)
        xb = torch.from_numpy(x_num[i:j].astype(np.float32)).to(device)

        logits = model(xb, None)

        if logits.ndim == 3:
            logits = logits.mean(dim=1)

        logits = logits.squeeze(-1)
        probs[i:j] = torch.sigmoid(logits).float().cpu().numpy().astype(np.float32)

    return probs


def build_model(model_pt, tabm_path, n_features, device):
    ckpt = torch.load(resolve_path(model_pt), map_location="cpu")
    tabm_args = ckpt.get("tabm_args", {})

    TabM = load_tabm_class(tabm_path)

    model = TabM(
        n_num_features=n_features,
        cat_cardinalities=None,
        d_out=1,
        num_embeddings=None,
        k=int(tabm_args.get("k", 32)),
        n_blocks=int(tabm_args.get("n_blocks", 4)),
        d_block=int(tabm_args.get("d_block", 512)),
        dropout=float(tabm_args.get("dropout", 0.05)),
        activation=str(tabm_args.get("activation", "ReLU")),
        arch_type=str(tabm_args.get("arch_type", "tabm")),
        start_scaling_init=tabm_args.get("start_scaling_init", "random-signs")
        if str(tabm_args.get("arch_type", "tabm")) != "tabm-packed"
        else None,
    ).to(device)

    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    return model


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--sheet", type=str, default=None)

    ap.add_argument("--model_pt", type=str, required=True)
    ap.add_argument("--prep_npz", type=str, required=True)
    ap.add_argument("--tabm_path", type=str, default="tabm.py")

    ap.add_argument("--out_dir", type=str, default="tabm_shap_outputs")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--label", type=str, default="label")

    ap.add_argument("--background_n", type=int, default=100)
    ap.add_argument("--explain_n", type=int, default=200)
    ap.add_argument("--nsamples", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    print("[DEVICE]", device)

    np.random.seed(args.seed)

    df = read_table(args.data, sheet_name=args.sheet)

    prep = load_prep_npz(args.prep_npz)
    feat_order = prep["features_order"]

    X_all = encode_mixed_from_prep(df, prep)

    model = build_model(
        model_pt=args.model_pt,
        tabm_path=args.tabm_path,
        n_features=len(feat_order),
        device=device
    )

    n = X_all.shape[0]
    all_idx = np.arange(n)

    if n > args.explain_n:
        explain_idx = np.random.choice(all_idx, size=args.explain_n, replace=False)
    else:
        explain_idx = all_idx

    remain_idx = all_idx

    if len(remain_idx) > args.background_n:
        bg_idx = np.random.choice(remain_idx, size=args.background_n, replace=False)
    else:
        bg_idx = remain_idx

    X_bg = X_all[bg_idx]
    X_exp = X_all[explain_idx]

    def predict_fn(x):
        x = np.asarray(x, dtype=np.float32)
        return predict_proba_torch(
            model,
            x,
            device=device,
            batch_size=args.batch_size
        )

    print("[SHAP] background:", X_bg.shape)
    print("[SHAP] explain:", X_exp.shape)

    explainer = shap.KernelExplainer(predict_fn, X_bg)
    shap_values = explainer.shap_values(X_exp, nsamples=args.nsamples)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_values = np.asarray(shap_values)

    np.save(os.path.join(args.out_dir, "shap_values.npy"), shap_values)
    np.save(os.path.join(args.out_dir, "X_explain_encoded.npy"), X_exp)
    np.save(os.path.join(args.out_dir, "explain_indices.npy"), explain_idx)

    mean_abs = np.abs(shap_values).mean(axis=0)

    imp_df = pd.DataFrame({
        "feature": feat_order,
        "mean_abs_shap": mean_abs
    }).sort_values("mean_abs_shap", ascending=False)

    imp_df.to_csv(
        os.path.join(args.out_dir, "mean_abs_shap.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    df_exp = df.iloc[explain_idx].copy()
    df_exp.insert(0, "explain_row_id", explain_idx)
    pred_exp = predict_fn(X_exp)
    df_exp["y_prob"] = pred_exp

    df_exp.to_csv(
        os.path.join(args.out_dir, "explained_samples_with_pred.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    plt.figure()
    shap.summary_plot(
        shap_values,
        X_exp,
        feature_names=feat_order,
        show=False,
        max_display=30
    )
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "shap_summary_dot.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(args.out_dir, "shap_summary_dot.pdf"), bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(
        shap_values,
        X_exp,
        feature_names=feat_order,
        plot_type="bar",
        show=False,
        max_display=30
    )
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "shap_summary_bar.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(args.out_dir, "shap_summary_bar.pdf"), bbox_inches="tight")
    plt.close()

    top_df = imp_df.head(30).copy()
    top_df.to_csv(
        os.path.join(args.out_dir, "top30_mean_abs_shap.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    print("[DONE]")
    print("Output directory:", os.path.abspath(args.out_dir))


if __name__ == "__main__":
    main()