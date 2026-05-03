import os, sys, re, argparse, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import torch
from sklearn.metrics import roc_auc_score, accuracy_score


def resolve_path(path: str) -> str:
    if os.path.exists(path):
        return path
    if not os.path.isabs(path):
        p2 = os.path.abspath(path)
        if os.path.exists(p2):
            return p2
    raise FileNotFoundError(f"File not found: {path}")


def parse_name_list(s: str):
    if s is None:
        return []
    s = str(s).strip()
    if s == "":
        return []
    parts = re.split(r"[,;\s]+", s)
    return [p.strip() for p in parts if p.strip()]


def load_tabm_class(tabm_path: str):
    import types, typing as T
    tabm_path = resolve_path(tabm_path)
    src = open(tabm_path, "r", encoding="utf-8").read()

    src = re.sub(r"\blist\[(.*?)\]", r"List[\1]", src)
    src = re.sub(r"\bdict\[(.*?)\]", r"Dict[\1]", src)
    src = re.sub(r"\btuple\[(.*?)\]", r"Tuple[\1]", src)
    src = re.sub(r"\bset\[(.*?)\]", r"Set[\1]", src)

    if not src.lstrip().startswith("from __future__ import annotations"):
        src = "from __future__ import annotations\n" + src

    module_name = "tabm_infer_mod"
    mod = types.ModuleType(module_name)
    mod.__file__ = tabm_path
    mod.__package__ = ""
    mod.__dict__.update({
        "List": T.List, "Dict": T.Dict, "Tuple": T.Tuple, "Set": T.Set,
        "Optional": T.Optional, "Union": T.Union, "Any": T.Any,
    })
    sys.modules[module_name] = mod
    exec(compile(src, tabm_path, "exec"), mod.__dict__)

    for name in ["TabM", "TABM", "Model", "Tabm", "tabm"]:
        if hasattr(mod, name):
            obj = getattr(mod, name)
            if isinstance(obj, type):
                return obj
    raise RuntimeError("No TabM class found in tabm.py.")


def map_binary_labels(y_raw):
    y_raw = np.asarray(y_raw)
    y_num = pd.to_numeric(pd.Series(y_raw), errors="coerce").to_numpy()

    if np.any(np.isnan(y_num)):
        s = pd.Series(y_raw).astype(str).str.strip().str.lower()
        mp = {
            "1": 1,
            "0": 0,
            "y": 1,
            "n": 0,
            "yes": 1,
            "no": 0,
            "true": 1,
            "false": 0,
        }
        y = s.map(mp)
        if y.isna().any():
            uniq = pd.unique(s)
            if len(uniq) != 2:
                raise ValueError(f"Label is not binary: {uniq}")
            mapping = {uniq[0]: 0, uniq[1]: 1}
            return np.asarray([mapping[v] for v in s], dtype=int)
        return y.fillna(0).astype(int).to_numpy()

    y_num = np.nan_to_num(y_num, nan=0.0)
    y = (y_num > 0).astype(int)
    if len(np.unique(y)) != 2:
        raise ValueError(f"Label is not binary 0/1: {np.unique(y)}")
    return y.astype(int)


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
def predict_proba(model, x_num, device, batch_size=4096):
    model.eval()
    n = len(x_num)
    probs = np.zeros((n,), dtype=np.float32)

    for i in range(0, n, batch_size):
        j = min(n, i + batch_size)
        bn = torch.from_numpy(x_num[i:j]).to(device)
        logits = model(bn, None)
        if logits.ndim == 3:
            logits = logits.mean(dim=1)
        logits = logits.squeeze(-1)
        probs[i:j] = torch.sigmoid(logits).float().cpu().numpy().astype(np.float32)

    return probs


def select_sheets(all_sheets, include_names, include_regex, exclude_names):
    chosen = list(all_sheets)

    if include_names:
        name_set = set(include_names)
        chosen = [s for s in chosen if s in name_set]

    if include_regex:
        pat = re.compile(include_regex)
        chosen = [s for s in chosen if pat.search(s)]

    if exclude_names:
        ex_set = set(exclude_names)
        chosen = [s for s in chosen if s not in ex_set]

    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", type=str, required=True, help="Input XLSX file with one or more sheets.")
    ap.add_argument("--out_dir", type=str, default="infer_outputs", help="Output directory.")

    ap.add_argument("--model_pt", type=str, required=True, help="Path to final_best.pt.")
    ap.add_argument("--prep_npz", type=str, required=True, help="Path to final_prep.npz.")
    ap.add_argument("--tabm_path", type=str, default="tabm.py", help="Path to tabm.py.")

    ap.add_argument("--label", type=str, default="label", help="Label column name. Metrics are computed if this column exists.")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--threshold", type=float, default=0.5, help="Threshold for class prediction and accuracy.")

    ap.add_argument("--sheets", type=str, default="", help="Comma-separated sheet names to include.")
    ap.add_argument("--sheet_regex", type=str, default="", help="Regular expression for sheet selection.")
    ap.add_argument("--exclude_sheets", type=str, default="", help="Comma-separated sheet names to exclude.")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    include_names = parse_name_list(args.sheets)
    exclude_names = parse_name_list(args.exclude_sheets)
    include_regex = args.sheet_regex.strip() if args.sheet_regex else ""

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA is not available. Falling back to CPU.")
        device = "cpu"

    print("[DEVICE]", device)
    if device == "cuda":
        print("[GPU]", torch.cuda.get_device_name(0))

    prep = load_prep_npz(args.prep_npz)
    feat_order = prep["features_order"]

    print("[PREP] n_features_model_input =", len(feat_order))
    print("[PREP] z_features =", len(prep["z_features"]), "raw_features_no_z =", len(prep["raw_features"]))

    ckpt = torch.load(resolve_path(args.model_pt), map_location="cpu")
    tabm_args = ckpt.get("tabm_args", {})

    TabM = load_tabm_class(args.tabm_path)
    model = TabM(
        n_num_features=len(feat_order),
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
        if str(tabm_args.get("arch_type", "tabm")) != "tabm-packed" else None,
    ).to(device)

    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    xlsx_path = resolve_path(args.xlsx)
    xls = pd.ExcelFile(xlsx_path)
    all_sheets = xls.sheet_names

    print("[XLSX] all sheets:", all_sheets)

    chosen_sheets = select_sheets(all_sheets, include_names, include_regex, exclude_names)
    if not chosen_sheets:
        raise ValueError("No sheets selected. Check --sheets, --sheet_regex, or --exclude_sheets.")

    print("[XLSX] chosen sheets:", chosen_sheets)

    out_xlsx = os.path.join(args.out_dir, "predictions_by_sheet.xlsx")
    metrics_rows = []

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        for sname in chosen_sheets:
            df = pd.read_excel(xlsx_path, sheet_name=sname)

            if df.shape[0] == 0:
                print(f"[SKIP] sheet={sname} is empty.")
                continue

            df_out = df.copy()

            missing_feats = [c for c in feat_order if c not in df_out.columns]
            if missing_feats:
                print(f"[WARN] sheet={sname} is missing {len(missing_feats)} features. They will be imputed.")
                for c in missing_feats:
                    df_out[c] = np.nan

            Xn = encode_mixed_from_prep(df_out, prep)
            probs = predict_proba(model, Xn, device=device, batch_size=args.batch_size)
            df_out["y_prob"] = probs.astype(float)

            has_label = args.label in df_out.columns
            auc = np.nan
            acc = np.nan
            n = int(len(df_out))
            pos = np.nan

            if has_label:
                try:
                    y = map_binary_labels(df_out[args.label].to_numpy())
                    pos = int((y == 1).sum())
                    if len(np.unique(y)) == 2:
                        auc = float(roc_auc_score(y, probs))
                    acc = float(accuracy_score(y, (probs >= float(args.threshold)).astype(int)))
                    df_out["y_true"] = y.astype(int)
                    df_out["y_pred"] = (probs >= float(args.threshold)).astype(int)
                except Exception as e:
                    print(f"[WARN] sheet={sname} has a label column, but metrics could not be computed: {e}")

            safe_name = sname[:31]
            df_out.to_excel(writer, sheet_name=safe_name, index=False)

            out_csv = os.path.join(args.out_dir, f"pred_{sname}.csv")
            out_csv = re.sub(r'[\\/:*?"<>|]+', "_", out_csv)
            df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")

            metrics_rows.append({
                "sheet": sname,
                "n": n,
                "pos": pos,
                "has_label": bool(has_label),
                "auc": auc,
                "acc_at_threshold": acc,
                "threshold": float(args.threshold),
                "missing_features": len(missing_feats),
            })

    mdf = pd.DataFrame(metrics_rows)
    mdf.to_csv(os.path.join(args.out_dir, "metrics_summary.csv"), index=False, encoding="utf-8-sig")

    print("[DONE]")
    print("  outputs:", os.path.abspath(args.out_dir))
    print("  xlsx:", os.path.abspath(out_xlsx))
    print("  metrics_summary.csv saved.")


if __name__ == "__main__":
    main()
