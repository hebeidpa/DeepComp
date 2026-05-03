import os
import sys
import re
import json
import argparse
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch


def resolve_path(path):
    if os.path.exists(path):
        return path
    if not os.path.isabs(path):
        p2 = os.path.abspath(path)
        if os.path.exists(p2):
            return p2
    raise FileNotFoundError(f"File not found: {path}")


def parse_name_list(s):
    if s is None:
        return []
    s = str(s).strip()
    if s == "":
        return []
    parts = re.split(r"[,;\s]+", s)
    return [p.strip() for p in parts if p.strip()]


def load_tabm_class(tabm_path):
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

    module_name = "tabm_os_infer_mod"
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

    raise RuntimeError("TabM class was not found in tabm.py.")


def load_prep_npz(prep_npz_path):
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


def encode_mixed_from_prep(df, prep):
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


def concordance_index(time, event, risk):
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    risk = np.asarray(risk, dtype=float)

    valid = np.isfinite(time) & np.isfinite(event) & np.isfinite(risk)
    time = time[valid]
    event = event[valid]
    risk = risk[valid]

    n = len(time)
    den = 0.0
    num = 0.0

    for i in range(n):
        if event[i] != 1:
            continue
        for j in range(n):
            if time[i] < time[j]:
                den += 1.0
                if risk[i] > risk[j]:
                    num += 1.0
                elif risk[i] == risk[j]:
                    num += 0.5

    if den == 0:
        return float("nan")

    return float(num / den)


@torch.no_grad()
def predict_risk(model, x_num, device, batch_size=4096):
    model.eval()
    n = len(x_num)
    risk = np.zeros((n,), dtype=np.float32)

    for i in range(0, n, batch_size):
        j = min(n, i + batch_size)
        xb = torch.from_numpy(x_num[i:j].astype(np.float32)).to(device)

        out = model(xb, None)

        if out.ndim == 3:
            out = out.mean(dim=1)

        out = out.squeeze(-1)
        risk[i:j] = out.float().cpu().numpy().astype(np.float32)

    return risk


def safe_torch_load(path):
    path = resolve_path(path)
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def build_model(model_pt, tabm_path, n_features, device):
    ckpt = safe_torch_load(model_pt)
    tabm_args = ckpt.get("tabm_args", {})

    TabM = load_tabm_class(tabm_path)

    arch_type = str(tabm_args.get("arch_type", "tabm"))

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
        arch_type=arch_type,
        start_scaling_init=tabm_args.get("start_scaling_init", "random-signs") if arch_type != "tabm-packed" else None,
    ).to(device)

    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    return model


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


def make_sheet_name(name, used):
    base = re.sub(r"[\[\]\:\*\?\/\\]", "_", str(name))[:31]
    if base == "":
        base = "Sheet"
    final = base
    k = 1
    while final in used:
        suffix = f"_{k}"
        final = base[:31 - len(suffix)] + suffix
        k += 1
    used.add(final)
    return final


def run_one_dataframe(df, prep, model, device, batch_size, time_col, event_col, risk_cutoff):
    df_out = df.copy()

    feat_order = prep["features_order"]
    missing_feats = [c for c in feat_order if c not in df_out.columns]

    for c in missing_feats:
        df_out[c] = np.nan

    Xn = encode_mixed_from_prep(df_out, prep)
    risk = predict_risk(model, Xn, device=device, batch_size=batch_size)

    df_out["os_risk"] = risk.astype(float)

    if risk_cutoff is None:
        cutoff = float(np.nanmedian(risk))
    else:
        cutoff = float(risk_cutoff)

    df_out["os_risk_group"] = np.where(df_out["os_risk"] >= cutoff, "high", "low")

    has_os = time_col in df_out.columns and event_col in df_out.columns
    cindex = np.nan
    n = int(len(df_out))
    events = np.nan

    if has_os:
        time = pd.to_numeric(df_out[time_col], errors="coerce").to_numpy(dtype=float)
        event = pd.to_numeric(df_out[event_col], errors="coerce").fillna(0).to_numpy(dtype=int)
        events = int((event == 1).sum())
        cindex = concordance_index(time, event, risk)

    metrics = {
        "n": n,
        "events": events,
        "has_os": bool(has_os),
        "cindex": cindex,
        "risk_cutoff": cutoff,
        "missing_features": int(len(missing_feats)),
    }

    return df_out, metrics


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--model_pt", type=str, required=True)
    ap.add_argument("--prep_npz", type=str, required=True)
    ap.add_argument("--tabm_path", type=str, default="tabm.py")
    ap.add_argument("--out_dir", type=str, default="os_infer_outputs")

    ap.add_argument("--time", type=str, default="OS_time")
    ap.add_argument("--event", type=str, default="OS_event")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--risk_cutoff", type=float, default=None)

    ap.add_argument("--sheets", type=str, default="")
    ap.add_argument("--sheet_regex", type=str, default="")
    ap.add_argument("--exclude_sheets", type=str, default="")

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

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
    print("[PREP] z_features =", len(prep["z_features"]), "raw_features =", len(prep["raw_features"]))

    model = build_model(
        model_pt=args.model_pt,
        tabm_path=args.tabm_path,
        n_features=len(feat_order),
        device=device,
    )

    data_path = resolve_path(args.data)
    ext = os.path.splitext(data_path)[1].lower()

    metrics_rows = []

    if ext == ".csv":
        df = pd.read_csv(data_path, encoding="utf-8-sig")
        df_out, metrics = run_one_dataframe(
            df=df,
            prep=prep,
            model=model,
            device=device,
            batch_size=args.batch_size,
            time_col=args.time,
            event_col=args.event,
            risk_cutoff=args.risk_cutoff,
        )

        out_csv = os.path.join(args.out_dir, "os_predictions.csv")
        df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")

        metrics["dataset"] = os.path.basename(data_path)
        metrics_rows.append(metrics)

    elif ext in [".xlsx", ".xls"]:
        xls = pd.ExcelFile(data_path)
        all_sheets = xls.sheet_names

        include_names = parse_name_list(args.sheets)
        exclude_names = parse_name_list(args.exclude_sheets)
        include_regex = args.sheet_regex.strip() if args.sheet_regex else ""

        chosen_sheets = select_sheets(
            all_sheets=all_sheets,
            include_names=include_names,
            include_regex=include_regex,
            exclude_names=exclude_names,
        )

        if not chosen_sheets:
            raise ValueError("No sheet was selected.")

        out_xlsx = os.path.join(args.out_dir, "os_predictions_by_sheet.xlsx")
        used_sheet_names = set()

        with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
            for sname in chosen_sheets:
                df = pd.read_excel(data_path, sheet_name=sname)

                if df.shape[0] == 0:
                    continue

                df_out, metrics = run_one_dataframe(
                    df=df,
                    prep=prep,
                    model=model,
                    device=device,
                    batch_size=args.batch_size,
                    time_col=args.time,
                    event_col=args.event,
                    risk_cutoff=args.risk_cutoff,
                )

                safe_name = make_sheet_name(sname, used_sheet_names)
                df_out.to_excel(writer, sheet_name=safe_name, index=False)

                out_csv = os.path.join(args.out_dir, f"os_pred_{sname}.csv")
                out_csv = re.sub(r'[\\/:*?"<>|]+', "_", out_csv)
                df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")

                metrics["dataset"] = sname
                metrics_rows.append(metrics)

    else:
        raise ValueError(f"Unsupported input format: {ext}")

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_csv = os.path.join(args.out_dir, "os_metrics_summary.csv")
    metrics_df.to_csv(metrics_csv, index=False, encoding="utf-8-sig")

    metrics_json = os.path.join(args.out_dir, "os_metrics_summary.json")
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics_rows, f, ensure_ascii=False, indent=2)

    print("[DONE]")
    print("Output directory:", os.path.abspath(args.out_dir))
    print("Metrics:", os.path.abspath(metrics_csv))


if __name__ == "__main__":
    main()