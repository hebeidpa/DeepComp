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
import torch.nn as nn

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, KFold
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.stats import mannwhitneyu
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


def parse_keep_features(s):
    if s is None:
        return []
    s = str(s).strip()
    if s == "":
        return []
    parts = re.split(r"[,;\s]+", s)
    return [p.strip() for p in parts if p.strip()]


def ensure_keep_features_exist(cols, keep_list):
    keep = [c for c in keep_list if c in cols]
    missing = [c for c in keep_list if c not in cols]
    if missing:
        print(f"[KEEP] Missing mandatory features will be ignored: {missing}")
    return keep


def resolve_path(path):
    if os.path.exists(path):
        return path
    if not os.path.isabs(path):
        p2 = os.path.abspath(path)
        if os.path.exists(p2):
            return p2
    cands = [path + ext for ext in [".csv", ".xlsx", ".xls"]]
    hits = [p for p in cands if os.path.exists(p)]
    if len(hits) == 1:
        print(f"[PATH] auto-resolve: {path} -> {hits[0]}")
        return hits[0]
    if len(hits) > 1:
        raise FileNotFoundError(f"Ambiguous path: {path}; candidates={hits}")
    raise FileNotFoundError(f"File not found: {path}; candidates={cands}")


def read_table(path):
    path = resolve_path(path)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path, encoding="utf-8-sig")
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported format: {ext}")


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

    module_name = "tabm_dynamic_model"
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


def pick_feature_types(df_X, cat_max_unique=50):
    cat_cols = []
    num_cols = []

    for c in df_X.columns:
        s = df_X[c]
        if str(s.dtype) in ("object", "category", "bool"):
            cat_cols.append(c)
        elif np.issubdtype(s.dtype, np.integer):
            nunq = s.nunique(dropna=True)
            if nunq <= cat_max_unique:
                cat_cols.append(c)
            else:
                num_cols.append(c)
        else:
            num_cols.append(c)

    return num_cols, cat_cols


def feature_select_3step(
    X,
    y,
    alpha=0.10,
    corr_th=0.95,
    max_vars=128,
    lasso_Cs=None,
    lasso_cv=5,
    random_state=42,
    keep_features=None,
):
    y = np.asarray(y).astype(int)
    cols = list(X.columns)

    if len(np.unique(y)) < 2:
        keep_list_guard = ensure_keep_features_exist(cols, keep_features or [])
        keep_set_guard = set(keep_list_guard)
        rest = [c for c in cols if c not in keep_set_guard]
        chosen = keep_list_guard + rest
        if max_vars is not None and max_vars > 0:
            chosen = chosen[:max_vars]
        return chosen

    keep_list = ensure_keep_features_exist(cols, keep_features or [])
    keep_set = set(keep_list)

    pvals = {c: 1.0 for c in cols}

    if SCIPY_AVAILABLE:
        for c in cols:
            s = pd.to_numeric(X[c], errors="coerce")
            g0 = s[y == 0].dropna()
            g1 = s[y == 1].dropna()

            if len(g0) < 5 or len(g1) < 5:
                pvals[c] = 1.0
                continue

            try:
                _, p = mannwhitneyu(g0, g1, alternative="two-sided")
                pvals[c] = float(p)
            except Exception:
                pvals[c] = 1.0

        cand = [c for c in cols if pvals[c] <= alpha]
        if len(cand) == 0:
            cand = cols

        for k in keep_list:
            if k not in cand:
                cand.append(k)
    else:
        cand = cols

    if len(cand) > 1:
        df_corr = X[cand].apply(pd.to_numeric, errors="coerce")
        df_corr = df_corr.fillna(df_corr.median())
        corr = df_corr.corr(method="spearman").fillna(0.0)

        sorted_cols = sorted(cand, key=lambda c: pvals.get(c, 1.0))
        kept = list(keep_list)

        for c in sorted_cols:
            if c in keep_set:
                continue
            ok = True
            for k in kept:
                if abs(corr.loc[c, k]) >= corr_th:
                    ok = False
                    break
            if ok:
                kept.append(c)

        cand2 = kept
    else:
        cand2 = cand

    df_lasso = X[cand2].apply(pd.to_numeric, errors="coerce")
    imp = SimpleImputer(strategy="median")
    scl = StandardScaler()
    X_imp = imp.fit_transform(df_lasso)
    X_std = scl.fit_transform(X_imp)

    if lasso_Cs is None:
        lasso_Cs = [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]

    clf = LogisticRegressionCV(
        penalty="l1",
        solver="saga",
        Cs=lasso_Cs,
        cv=lasso_cv,
        scoring="roc_auc",
        max_iter=4000,
        n_jobs=-1,
        refit=True,
        random_state=random_state,
    )
    clf.fit(X_std, y)

    coef = clf.coef_.ravel()
    non_zero = [(cand2[i], abs(coef[i])) for i in range(len(cand2)) if abs(coef[i]) > 1e-8]

    if len(non_zero) == 0:
        chosen = cand2
    else:
        non_zero.sort(key=lambda x: x[1], reverse=True)
        if max_vars is not None and max_vars > 0:
            non_zero = non_zero[:max_vars]
        chosen = [c for c, _ in non_zero]

    ordered = []

    for k in keep_list:
        if k in cand2 and k not in ordered:
            ordered.append(k)

    for c in chosen:
        if c not in ordered:
            ordered.append(c)

    chosen = ordered

    if max_vars is not None and len(chosen) > max_vars:
        keep_first = [c for c in chosen if c in keep_set]
        rest = [c for c in chosen if c not in keep_set]
        rest = rest[:max(0, max_vars - len(keep_first))]
        chosen = keep_first + rest

    return chosen


def stability_select(
    X_train,
    y_train,
    inner_folds=5,
    repeats=3,
    min_freq=0.6,
    fs_alpha=0.10,
    fs_corr_th=0.95,
    fs_max_vars=128,
    lasso_cv=5,
    random_state=42,
    keep_features=None,
):
    y_train = np.asarray(y_train).astype(int)
    cols = list(X_train.columns)
    counts = {c: 0 for c in cols}
    total_runs = 0

    for r in range(repeats):
        skf = StratifiedKFold(
            n_splits=inner_folds,
            shuffle=True,
            random_state=random_state + 1000 * r,
        )

        for tr_idx, _ in skf.split(X_train, y_train):
            total_runs += 1

            X_sub = X_train.iloc[tr_idx]
            y_sub = y_train[tr_idx]

            chosen = feature_select_3step(
                X_sub,
                y_sub,
                alpha=fs_alpha,
                corr_th=fs_corr_th,
                max_vars=fs_max_vars,
                lasso_cv=lasso_cv,
                random_state=random_state + 17 * total_runs,
                keep_features=keep_features,
            )

            for c in chosen:
                counts[c] += 1

    freqs = {c: counts[c] / max(total_runs, 1) for c in cols}
    selected = [c for c in cols if freqs[c] >= min_freq]

    if len(selected) < max(16, min(32, fs_max_vars)):
        ranked = sorted(cols, key=lambda c: freqs[c], reverse=True)
        selected = ranked[:min(fs_max_vars, len(ranked))]

    final = feature_select_3step(
        X_train[selected],
        y_train,
        alpha=1.0,
        corr_th=0.99,
        max_vars=fs_max_vars,
        lasso_cv=lasso_cv,
        random_state=random_state + 99999,
        keep_features=keep_features,
    )

    return final, freqs


def compute_mixed_stats(train_df, all_cols, keep_raw_cols):
    keep_raw_set = set(keep_raw_cols)
    z_cols = [c for c in all_cols if c not in keep_raw_set]
    raw_cols = [c for c in all_cols if c in keep_raw_set]

    meds = {}
    means = {}
    stds = {}
    raw_meds = {}

    for c in z_cols:
        v = pd.to_numeric(train_df[c], errors="coerce").to_numpy()
        med = np.nanmedian(v)
        med = float(med) if np.isfinite(med) else 0.0
        v = np.where(np.isfinite(v), v, med)
        m = float(np.mean(v))
        s = float(np.std(v))
        if not np.isfinite(s) or s <= 1e-12:
            s = 1.0
        meds[c] = med
        means[c] = m
        stds[c] = s

    for c in raw_cols:
        v = pd.to_numeric(train_df[c], errors="coerce").to_numpy()
        med = np.nanmedian(v)
        med = float(med) if np.isfinite(med) else 0.0
        raw_meds[c] = med

    return z_cols, raw_cols, meds, means, stds, raw_meds


def encode_numeric_mixed(df_X, z_cols, raw_cols, meds, means, stds, raw_meds):
    cols = list(z_cols) + list(raw_cols)
    Xn = np.zeros((len(df_X), len(cols)), dtype=np.float32)

    for j, c in enumerate(z_cols):
        v = pd.to_numeric(df_X[c], errors="coerce").to_numpy()
        v = np.where(np.isfinite(v), v, meds[c])
        v = (v - means[c]) / stds[c]
        Xn[:, j] = v.astype(np.float32)

    base = len(z_cols)

    for j, c in enumerate(raw_cols):
        v = pd.to_numeric(df_X[c], errors="coerce").to_numpy()
        v = np.where(np.isfinite(v), v, raw_meds[c])
        Xn[:, base + j] = v.astype(np.float32)

    return Xn, cols


def save_prep_mixed(path, z_cols, raw_cols, meds, means, stds, raw_meds):
    features_order = list(z_cols) + list(raw_cols)
    np.savez_compressed(
        path,
        features_order=np.array(features_order, dtype=object),
        z_features=np.array(z_cols, dtype=object),
        z_median=np.array([meds[c] for c in z_cols], dtype=np.float32),
        z_mean=np.array([means[c] for c in z_cols], dtype=np.float32),
        z_std=np.array([stds[c] for c in z_cols], dtype=np.float32),
        raw_features=np.array(raw_cols, dtype=object),
        raw_median=np.array([raw_meds[c] for c in raw_cols], dtype=np.float32),
    )


class FocalLossWithLogits(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction="mean"):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.reduction = reduction

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        pt = torch.where(targets >= 0.5, p, 1 - p)
        alpha_t = torch.where(
            targets >= 0.5,
            torch.tensor(self.alpha, device=logits.device),
            torch.tensor(1.0 - self.alpha, device=logits.device),
        )
        loss = alpha_t * (1 - pt).pow(self.gamma) * bce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()

        return loss


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


@torch.no_grad()
def eval_cls_loss(model, x_num, y, device, criterion, batch_size=4096):
    model.eval()
    n = len(x_num)
    total = 0.0
    nb = 0

    for i in range(0, n, batch_size):
        j = min(n, i + batch_size)
        xb = torch.from_numpy(x_num[i:j]).to(device)
        yb = torch.from_numpy(y[i:j].astype(np.float32)).to(device)

        logits = model(xb, None)

        if logits.ndim == 3:
            logits = logits.mean(dim=1)

        logits = logits.squeeze(-1)
        loss = criterion(logits, yb)

        total += float(loss.detach().cpu().item())
        nb += 1

    return total / max(nb, 1)


def save_cls_curves(run_dir, history, title_prefix=""):
    os.makedirs(run_dir, exist_ok=True)
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title_prefix} Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "loss_curve.png"), dpi=200)
    plt.savefig(os.path.join(run_dir, "loss_curve.pdf"))
    plt.close()

    plt.figure()
    plt.plot(epochs, history["train_auc"], label="train_auc")
    plt.plot(epochs, history["val_auc"], label="val_auc")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.title(f"{title_prefix} AUC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "auc_curve.png"), dpi=200)
    plt.savefig(os.path.join(run_dir, "auc_curve.pdf"))
    plt.close()

    pd.DataFrame(history).to_csv(os.path.join(run_dir, "history.csv"), index=False, encoding="utf-8-sig")


def train_tabm_one(
    TabM,
    Xtr_num,
    ytr,
    Xva_num,
    yva,
    k=32,
    arch_type="tabm",
    d_block=512,
    n_blocks=4,
    dropout=0.05,
    start_scaling_init="random-signs",
    activation="ReLU",
    use_focal=True,
    focal_gamma=2.0,
    focal_alpha="auto",
    use_pos_weight=True,
    pos_weight_cap=50.0,
    lr=1e-3,
    weight_decay=1e-4,
    epochs=400,
    batch_size=256,
    patience=60,
    grad_clip=1.0,
    amp=True,
    seed=42,
    device="cuda",
    curve_eval_batch=4096,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = TabM(
        n_num_features=Xtr_num.shape[1],
        cat_cardinalities=None,
        d_out=1,
        num_embeddings=None,
        k=k,
        n_blocks=n_blocks,
        d_block=d_block,
        dropout=dropout,
        activation=activation,
        arch_type=arch_type,
        start_scaling_init=start_scaling_init if arch_type != "tabm-packed" else None,
    ).to(device)

    pos = float((ytr == 1).sum())
    neg = float((ytr == 0).sum())

    if use_focal:
        if focal_alpha == "auto":
            alpha = float(min(0.95, max(0.05, neg / max(pos + neg, 1.0))))
        else:
            alpha = float(focal_alpha)
        criterion = FocalLossWithLogits(gamma=focal_gamma, alpha=alpha)
    else:
        pos_weight = None
        if use_pos_weight and pos > 0:
            w = min(pos_weight_cap, neg / max(pos, 1.0))
            pos_weight = torch.tensor([w], dtype=torch.float32, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(amp and device == "cuda"))

    idx = np.arange(len(ytr))
    best_auc = -1.0
    best_state = None
    best_epoch = 0
    bad = 0
    rng = np.random.RandomState(seed)

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_auc": [],
        "val_auc": [],
    }

    for epoch in range(1, epochs + 1):
        model.train()
        rng.shuffle(idx)

        total_loss = 0.0
        nb = 0

        for i0 in range(0, len(idx), batch_size):
            bidx = idx[i0:i0 + batch_size]
            xb = torch.from_numpy(Xtr_num[bidx]).to(device)
            yb = torch.from_numpy(ytr[bidx].astype(np.float32)).to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(amp and device == "cuda")):
                logits = model(xb, None)

                if logits.ndim == 3:
                    logits = logits.mean(dim=1)

                logits = logits.squeeze(-1)
                loss = criterion(logits, yb)

            scaler.scale(loss).backward()

            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.detach().cpu().item())
            nb += 1

        train_loss = total_loss / max(nb, 1)

        train_probs = predict_proba(model, Xtr_num, device=device, batch_size=curve_eval_batch)
        val_probs = predict_proba(model, Xva_num, device=device, batch_size=curve_eval_batch)

        train_auc = float(roc_auc_score(ytr, train_probs)) if len(np.unique(ytr)) == 2 else float("nan")
        val_auc = float(roc_auc_score(yva, val_probs)) if len(np.unique(yva)) == 2 else float("nan")
        val_loss = float(eval_cls_loss(model, Xva_num, yva, device=device, criterion=criterion, batch_size=curve_eval_batch))

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_auc"].append(train_auc)
        history["val_auc"].append(val_auc)

        if val_auc > best_auc + 1e-6:
            best_auc = float(val_auc)
            best_state = {k0: v.detach().cpu().clone() for k0, v in model.state_dict().items()}
            best_epoch = epoch
            bad = 0
        else:
            bad += 1

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"[E{epoch:03d}] train_loss={train_loss:.6f} "
                f"val_loss={val_loss:.6f} train_auc={train_auc:.6f} "
                f"val_auc={val_auc:.6f} best={best_auc:.6f} bad={bad}/{patience}"
            )

        if bad >= patience:
            print("[EARLY STOP]")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history, best_auc, best_epoch, best_state


def best_threshold_by_acc(y_true, y_prob):
    thrs = np.unique(np.clip(y_prob, 1e-6, 1 - 1e-6))

    if len(thrs) > 400:
        thrs = np.quantile(thrs, np.linspace(0.0, 1.0, 400))

    best_t = 0.5
    best_acc = -1.0

    for t in thrs:
        acc = (((y_prob >= t).astype(int)) == y_true).mean()
        if acc > best_acc:
            best_acc = acc
            best_t = float(t)

    return best_t


def cox_ph_loss(risk, time, event):
    risk = risk.view(-1)
    time = time.view(-1)
    event = event.view(-1)

    valid = torch.isfinite(risk) & torch.isfinite(time) & torch.isfinite(event)
    risk = risk[valid]
    time = time[valid]
    event = event[valid]

    if risk.numel() < 2 or event.sum() < 1:
        return risk.sum() * 0.0

    order = torch.argsort(time, descending=True)
    risk = risk[order]
    event = event[order]

    log_cumsum = torch.logcumsumexp(risk, dim=0)
    loss = -((risk - log_cumsum) * event).sum() / event.sum().clamp_min(1.0)

    return loss


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
        xb = torch.from_numpy(x_num[i:j]).to(device)

        out = model(xb, None)

        if out.ndim == 3:
            out = out.mean(dim=1)

        out = out.squeeze(-1)
        risk[i:j] = out.float().cpu().numpy().astype(np.float32)

    return risk


@torch.no_grad()
def eval_os_loss(model, x_num, time, event, device, batch_size=4096):
    model.eval()
    n = len(x_num)
    total = 0.0
    nb = 0

    for i in range(0, n, batch_size):
        j = min(n, i + batch_size)

        xb = torch.from_numpy(x_num[i:j]).to(device)
        tb = torch.from_numpy(time[i:j].astype(np.float32)).to(device)
        eb = torch.from_numpy(event[i:j].astype(np.float32)).to(device)

        out = model(xb, None)

        if out.ndim == 3:
            out = out.mean(dim=1)

        risk = out.squeeze(-1)
        loss = cox_ph_loss(risk, tb, eb)

        total += float(loss.detach().cpu().item())
        nb += 1

    return total / max(nb, 1)


def save_os_curves(run_dir, history, title_prefix=""):
    os.makedirs(run_dir, exist_ok=True)
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cox Loss")
    plt.title(f"{title_prefix} OS Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "os_loss_curve.png"), dpi=200)
    plt.savefig(os.path.join(run_dir, "os_loss_curve.pdf"))
    plt.close()

    plt.figure()
    plt.plot(epochs, history["train_cindex"], label="train_cindex")
    plt.plot(epochs, history["val_cindex"], label="val_cindex")
    plt.xlabel("Epoch")
    plt.ylabel("C-index")
    plt.title(f"{title_prefix} OS C-index")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "os_cindex_curve.png"), dpi=200)
    plt.savefig(os.path.join(run_dir, "os_cindex_curve.pdf"))
    plt.close()

    pd.DataFrame(history).to_csv(os.path.join(run_dir, "history.csv"), index=False, encoding="utf-8-sig")


def train_os_one(
    TabM,
    Xtr_num,
    ttr,
    etr,
    Xva_num,
    tva,
    eva,
    k=32,
    arch_type="tabm",
    d_block=512,
    n_blocks=4,
    dropout=0.05,
    start_scaling_init="random-signs",
    activation="ReLU",
    lr=1e-3,
    weight_decay=1e-4,
    epochs=400,
    batch_size=256,
    patience=60,
    grad_clip=1.0,
    amp=True,
    seed=42,
    device="cuda",
    curve_eval_batch=4096,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = TabM(
        n_num_features=Xtr_num.shape[1],
        cat_cardinalities=None,
        d_out=1,
        num_embeddings=None,
        k=k,
        n_blocks=n_blocks,
        d_block=d_block,
        dropout=dropout,
        activation=activation,
        arch_type=arch_type,
        start_scaling_init=start_scaling_init if arch_type != "tabm-packed" else None,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(amp and device == "cuda"))

    idx = np.arange(len(ttr))
    rng = np.random.RandomState(seed)
    best_cindex = -1.0
    best_state = None
    best_epoch = 0
    bad = 0

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_cindex": [],
        "val_cindex": [],
    }

    for epoch in range(1, epochs + 1):
        model.train()
        rng.shuffle(idx)

        total_loss = 0.0
        nb = 0

        for i0 in range(0, len(idx), batch_size):
            bidx = idx[i0:i0 + batch_size]

            xb = torch.from_numpy(Xtr_num[bidx]).to(device)
            tb = torch.from_numpy(ttr[bidx].astype(np.float32)).to(device)
            eb = torch.from_numpy(etr[bidx].astype(np.float32)).to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(amp and device == "cuda")):
                out = model(xb, None)

                if out.ndim == 3:
                    out = out.mean(dim=1)

                risk = out.squeeze(-1)
                loss = cox_ph_loss(risk, tb, eb)

            scaler.scale(loss).backward()

            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.detach().cpu().item())
            nb += 1

        train_loss = total_loss / max(nb, 1)

        train_risk = predict_risk(model, Xtr_num, device=device, batch_size=curve_eval_batch)
        val_risk = predict_risk(model, Xva_num, device=device, batch_size=curve_eval_batch)

        train_cindex = concordance_index(ttr, etr, train_risk)
        val_cindex = concordance_index(tva, eva, val_risk)

        val_loss = eval_os_loss(
            model,
            Xva_num,
            tva,
            eva,
            device=device,
            batch_size=curve_eval_batch,
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_cindex"].append(train_cindex)
        history["val_cindex"].append(val_cindex)

        score = val_cindex if np.isfinite(val_cindex) else -1.0

        if score > best_cindex + 1e-6:
            best_cindex = float(score)
            best_state = {k0: v.detach().cpu().clone() for k0, v in model.state_dict().items()}
            best_epoch = epoch
            bad = 0
        else:
            bad += 1

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"[OS E{epoch:03d}] train_loss={train_loss:.6f} "
                f"val_loss={val_loss:.6f} train_cindex={train_cindex:.6f} "
                f"val_cindex={val_cindex:.6f} best={best_cindex:.6f} bad={bad}/{patience}"
            )

        if bad >= patience:
            print("[OS EARLY STOP]")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history, best_cindex, best_epoch, best_state


def make_os_splitter(event, n_splits, seed):
    event = np.asarray(event).astype(int)
    if len(np.unique(event)) == 2 and min(np.bincount(event)) >= n_splits:
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed), event
    return KFold(n_splits=n_splits, shuffle=True, random_state=seed), None


def train_independent_os_model(
    TabM,
    X_tr_numdf,
    X_te_numdf,
    y_tr_time,
    y_tr_event,
    y_te_time,
    y_te_event,
    os_cols,
    keep_ok,
    args,
    device,
):
    os_root = os.path.join(args.out_dir, args.os_out_name)
    curves_dir = os.path.join(os_root, "curves")
    models_dir = os.path.join(os_root, "models")
    prep_dir = os.path.join(os_root, "prep")

    os.makedirs(curves_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(prep_dir, exist_ok=True)

    os_cols = [c for c in os_cols if c in X_tr_numdf.columns]
    keep_os = [c for c in keep_ok if c in os_cols]

    pd.DataFrame({"feature": os_cols}).to_csv(
        os.path.join(os_root, "os_model_features.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\n[OS] independent OS model enabled")
    print(f"[OS] features={len(os_cols)}")

    splitter, strat_y = make_os_splitter(y_tr_event, args.folds, args.seed + 707)
    oof_risk = np.zeros(len(X_tr_numdf), dtype=np.float32)
    run_summaries = []

    os_epochs = args.os_epochs if args.os_epochs is not None else args.epochs
    os_patience = args.os_patience if args.os_patience is not None else args.patience
    os_lr = args.os_lr if args.os_lr is not None else args.lr
    os_weight_decay = args.os_weight_decay if args.os_weight_decay is not None else args.weight_decay

    split_iter = splitter.split(X_tr_numdf[os_cols], strat_y) if strat_y is not None else splitter.split(X_tr_numdf[os_cols])

    for fold, (tri, vai) in enumerate(split_iter, 1):
        run_name = f"fold{fold}"
        print(f"\n===== OS CV {run_name}/{args.folds} =====")

        Xtr_f = X_tr_numdf.iloc[tri][os_cols]
        Xva_f = X_tr_numdf.iloc[vai][os_cols]

        ttr_f = y_tr_time[tri]
        etr_f = y_tr_event[tri]
        tva_f = y_tr_time[vai]
        eva_f = y_tr_event[vai]

        z_cols, raw_cols, meds, means, stds, raw_meds = compute_mixed_stats(
            Xtr_f,
            all_cols=os_cols,
            keep_raw_cols=keep_os,
        )

        Xtr_num, feat_order = encode_numeric_mixed(Xtr_f, z_cols, raw_cols, meds, means, stds, raw_meds)
        Xva_num, _ = encode_numeric_mixed(Xva_f, z_cols, raw_cols, meds, means, stds, raw_meds)

        model, history, best_cindex, best_epoch, best_state = train_os_one(
            TabM,
            Xtr_num,
            ttr_f,
            etr_f,
            Xva_num,
            tva_f,
            eva_f,
            k=args.k,
            arch_type=args.arch_type,
            d_block=args.d_block,
            n_blocks=args.n_blocks,
            dropout=args.dropout,
            start_scaling_init=args.start_scaling_init,
            activation=args.activation,
            lr=os_lr,
            weight_decay=os_weight_decay,
            epochs=os_epochs,
            batch_size=args.batch_size,
            patience=os_patience,
            grad_clip=args.grad_clip,
            amp=args.amp,
            seed=args.seed + 5000 + fold,
            device=device,
        )

        save_os_curves(os.path.join(curves_dir, run_name), history, title_prefix=run_name)

        model_path = os.path.join(models_dir, f"{run_name}_best.pt")
        torch.save(
            {
                "state_dict": best_state,
                "best_epoch": int(best_epoch),
                "best_val_cindex": float(best_cindex),
                "features_order": feat_order,
                "keep_raw_features": keep_os,
                "tabm_args": {
                    "k": args.k,
                    "arch_type": args.arch_type,
                    "d_block": args.d_block,
                    "n_blocks": args.n_blocks,
                    "dropout": args.dropout,
                    "start_scaling_init": args.start_scaling_init,
                    "activation": args.activation,
                },
                "task": "os_cox",
            },
            model_path,
        )

        prep_path = os.path.join(prep_dir, f"{run_name}_prep.npz")
        save_prep_mixed(prep_path, z_cols, raw_cols, meds, means, stds, raw_meds)

        p_va = predict_risk(model, Xva_num, device=device)
        oof_risk[vai] = p_va

        cidx = concordance_index(tva_f, eva_f, p_va)

        print(f"[OS {run_name}] C-index={cidx:.4f} best_epoch={best_epoch} best_val_cindex={best_cindex:.4f}")

        run_summaries.append({
            "run": run_name,
            "best_epoch": int(best_epoch),
            "best_val_cindex": float(best_cindex),
            "model_path": os.path.abspath(model_path),
            "prep_path": os.path.abspath(prep_path),
            "n_features_model_input": int(len(feat_order)),
        })

    cv_cindex = concordance_index(y_tr_time, y_tr_event, oof_risk)
    print(f"\n[OS CV-OOF] C-index={cv_cindex:.4f}")

    pd.DataFrame({
        "row_id": np.arange(len(y_tr_time)),
        "OS_time": y_tr_time.astype(float),
        "OS_event": y_tr_event.astype(int),
        "os_risk": oof_risk.astype(float),
    }).to_csv(os.path.join(os_root, "oof_os_pred.csv"), index=False, encoding="utf-8-sig")

    print("\n===== OS FINAL FIT on TRAIN, then TEST ONCE =====")

    valid_event = np.asarray(y_tr_event).astype(int)

    if len(np.unique(valid_event)) == 2 and min(np.bincount(valid_event)) >= 2:
        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=args.os_final_val_ratio,
            random_state=args.seed + 9090,
        )
        tr_idx, va_idx = next(sss.split(X_tr_numdf[os_cols], valid_event))
    else:
        rng = np.random.RandomState(args.seed + 9090)
        idx = np.arange(len(X_tr_numdf))
        rng.shuffle(idx)
        nva = max(1, int(round(len(idx) * args.os_final_val_ratio)))
        va_idx = idx[:nva]
        tr_idx = idx[nva:]

    Xtr_fit = X_tr_numdf.iloc[tr_idx][os_cols]
    Xva_fit = X_tr_numdf.iloc[va_idx][os_cols]

    ttr_fit = y_tr_time[tr_idx]
    etr_fit = y_tr_event[tr_idx]
    tva_fit = y_tr_time[va_idx]
    eva_fit = y_tr_event[va_idx]

    z_cols, raw_cols, meds, means, stds, raw_meds = compute_mixed_stats(
        Xtr_fit,
        all_cols=os_cols,
        keep_raw_cols=keep_os,
    )

    Xtr_num, feat_order = encode_numeric_mixed(Xtr_fit, z_cols, raw_cols, meds, means, stds, raw_meds)
    Xva_num, _ = encode_numeric_mixed(Xva_fit, z_cols, raw_cols, meds, means, stds, raw_meds)

    model, history, best_cindex, best_epoch, best_state = train_os_one(
        TabM,
        Xtr_num,
        ttr_fit,
        etr_fit,
        Xva_num,
        tva_fit,
        eva_fit,
        k=args.k,
        arch_type=args.arch_type,
        d_block=args.d_block,
        n_blocks=args.n_blocks,
        dropout=args.dropout,
        start_scaling_init=args.start_scaling_init,
        activation=args.activation,
        lr=os_lr,
        weight_decay=os_weight_decay,
        epochs=os_epochs,
        batch_size=args.batch_size,
        patience=os_patience,
        grad_clip=args.grad_clip,
        amp=args.amp,
        seed=args.seed + 9999,
        device=device,
    )

    run_name = "final"
    save_os_curves(os.path.join(curves_dir, run_name), history, title_prefix=run_name)

    final_model_path = os.path.join(models_dir, f"{run_name}_best.pt")
    torch.save(
        {
            "state_dict": best_state,
            "best_epoch": int(best_epoch),
            "best_val_cindex": float(best_cindex),
            "features_order": feat_order,
            "keep_raw_features": keep_os,
            "tabm_args": {
                "k": args.k,
                "arch_type": args.arch_type,
                "d_block": args.d_block,
                "n_blocks": args.n_blocks,
                "dropout": args.dropout,
                "start_scaling_init": args.start_scaling_init,
                "activation": args.activation,
            },
            "task": "os_cox",
        },
        final_model_path,
    )

    final_prep_path = os.path.join(prep_dir, f"{run_name}_prep.npz")
    save_prep_mixed(final_prep_path, z_cols, raw_cols, meds, means, stds, raw_meds)

    os_test_cindex = None

    if y_te_time is not None and y_te_event is not None:
        Xte_num, _ = encode_numeric_mixed(X_te_numdf[os_cols], z_cols, raw_cols, meds, means, stds, raw_meds)
        test_risk = predict_risk(model, Xte_num, device=device)
        os_test_cindex = concordance_index(y_te_time, y_te_event, test_risk)

        print(f"[OS TEST] C-index={os_test_cindex:.4f}")

        pd.DataFrame({
            "row_id": np.arange(len(y_te_time)),
            "OS_time": y_te_time.astype(float),
            "OS_event": y_te_event.astype(int),
            "os_risk": test_risk.astype(float),
        }).to_csv(os.path.join(os_root, "test_os_pred.csv"), index=False, encoding="utf-8-sig")
    else:
        print("[OS TEST] OS columns not found in test data. Test OS metrics were skipped.")

    metrics = {
        "task": "os_cox",
        "features": {
            "n_features": int(len(os_cols)),
            "os_use_cls_features": bool(args.os_use_cls_features),
            "features_file": "os_model_features.csv",
        },
        "cv_oof": {
            "cindex": float(cv_cindex),
        },
        "test": {
            "cindex": None if os_test_cindex is None else float(os_test_cindex),
        },
        "artifacts": {
            "os_root": os.path.abspath(os_root),
            "curves_dir": os.path.abspath(curves_dir),
            "models_dir": os.path.abspath(models_dir),
            "prep_dir": os.path.abspath(prep_dir),
            "final_best_model": os.path.abspath(final_model_path),
            "final_prep": os.path.abspath(final_prep_path),
        },
        "runs": run_summaries + [{
            "run": "final",
            "best_epoch": int(best_epoch),
            "best_val_cindex": float(best_cindex),
            "model_path": os.path.abspath(final_model_path),
            "prep_path": os.path.abspath(final_prep_path),
            "n_features_model_input": int(len(feat_order)),
        }],
    }

    with open(os.path.join(os_root, "os_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return metrics


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--train", type=str, default="./splits_8020/train")
    ap.add_argument("--test", type=str, default="./splits_8020/test")
    ap.add_argument("--label", type=str, default="label")
    ap.add_argument("--tabm_path", type=str, default="tabm.py")
    ap.add_argument("--out_dir", type=str, default="tabm_fs_global_mixed")

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=12)

    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--arch_type", type=str, default="tabm", choices=["tabm", "tabm-mini", "tabm-packed"])
    ap.add_argument("--d_block", type=int, default=512)
    ap.add_argument("--n_blocks", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--start_scaling_init", type=str, default="random-signs", choices=["random-signs", "normal"])
    ap.add_argument("--activation", type=str, default="ReLU")

    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=60)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--amp", action="store_true")

    ap.add_argument("--use_focal", action="store_true")
    ap.add_argument("--focal_gamma", type=float, default=2.0)
    ap.add_argument("--focal_alpha", type=str, default="auto")

    ap.add_argument("--use_pos_weight", action="store_true")
    ap.add_argument("--pos_weight_cap", type=float, default=50.0)

    ap.add_argument("--fs_enable", action="store_true")
    ap.add_argument("--fs_alpha", type=float, default=0.10)
    ap.add_argument("--fs_corr_th", type=float, default=0.95)
    ap.add_argument("--fs_max_vars", type=int, default=128)
    ap.add_argument("--fs_lasso_cv", type=int, default=5)
    ap.add_argument("--fs_inner_folds", type=int, default=5)
    ap.add_argument("--fs_repeats", type=int, default=3)
    ap.add_argument("--fs_min_freq", type=float, default=0.6)

    ap.add_argument("--final_val_ratio", type=float, default=0.2)
    ap.add_argument("--keep_features", type=str, default="CCI")

    ap.add_argument("--train_os_model", action="store_true")
    ap.add_argument("--os_time", type=str, default="OS_time")
    ap.add_argument("--os_event", type=str, default="OS_event")
    ap.add_argument("--os_out_name", type=str, default="os_model")
    ap.add_argument("--os_use_cls_features", action="store_true")
    ap.add_argument("--os_final_val_ratio", type=float, default=0.2)
    ap.add_argument("--os_epochs", type=int, default=None)
    ap.add_argument("--os_patience", type=int, default=None)
    ap.add_argument("--os_lr", type=float, default=None)
    ap.add_argument("--os_weight_decay", type=float, default=None)

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    curves_dir = os.path.join(args.out_dir, "curves")
    models_dir = os.path.join(args.out_dir, "models")
    prep_dir = os.path.join(args.out_dir, "prep")

    os.makedirs(curves_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(prep_dir, exist_ok=True)

    keep_list = parse_keep_features(args.keep_features)

    device = args.device

    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA is not available. Falling back to CPU.")
        device = "cpu"

    print("[DEVICE]", device)

    if device == "cuda":
        print("[GPU]", torch.cuda.get_device_name(0))

    if args.focal_alpha != "auto":
        args.focal_alpha = float(args.focal_alpha)

    TabM = load_tabm_class(args.tabm_path)
    print("[TABM] loaded:", TabM)

    df_tr = read_table(args.train)
    df_te = read_table(args.test)

    if args.label not in df_tr.columns or args.label not in df_te.columns:
        raise ValueError(f"Label column does not exist: {args.label}")

    y_tr = map_binary_labels(df_tr[args.label].to_numpy())
    y_te = map_binary_labels(df_te[args.label].to_numpy())

    if args.train_os_model:
        if args.os_time not in df_tr.columns or args.os_event not in df_tr.columns:
            raise ValueError(f"OS columns not found in training data: {args.os_time}, {args.os_event}")

        y_tr_os_time = pd.to_numeric(df_tr[args.os_time], errors="coerce").to_numpy(dtype=np.float32)
        y_tr_os_event = pd.to_numeric(df_tr[args.os_event], errors="coerce").fillna(0).to_numpy(dtype=np.float32)

        if args.os_time in df_te.columns and args.os_event in df_te.columns:
            y_te_os_time = pd.to_numeric(df_te[args.os_time], errors="coerce").to_numpy(dtype=np.float32)
            y_te_os_event = pd.to_numeric(df_te[args.os_event], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
        else:
            y_te_os_time = None
            y_te_os_event = None
    else:
        y_tr_os_time = None
        y_tr_os_event = None
        y_te_os_time = None
        y_te_os_event = None

    drop_cols_tr = [args.label]
    drop_cols_te = [args.label]

    for c in [args.os_time, args.os_event]:
        if c and c in df_tr.columns:
            drop_cols_tr.append(c)
        if c and c in df_te.columns:
            drop_cols_te.append(c)

    X_tr = df_tr.drop(columns=list(dict.fromkeys(drop_cols_tr))).copy()
    X_te = df_te.drop(columns=list(dict.fromkeys(drop_cols_te))).copy()
    X_te = X_te[list(X_tr.columns)]

    num_cols, cat_cols = pick_feature_types(X_tr, cat_max_unique=50)

    keep_ok = ensure_keep_features_exist(list(X_tr.columns), keep_list)

    for k in keep_ok:
        if k not in num_cols:
            num_cols.append(k)
        if k in cat_cols:
            cat_cols.remove(k)

    if len(cat_cols) > 0:
        print("[WARN] Categorical columns were detected but this version uses numeric columns only. Count:", len(cat_cols))

    X_tr_numdf = X_tr[num_cols].copy()
    X_te_numdf = X_te[num_cols].copy()

    print(f"[NUM] total={len(num_cols)}")

    GLOBAL_SEL_COLS = list(X_tr_numdf.columns)
    freqs = None

    if args.fs_enable:
        GLOBAL_SEL_COLS, freqs = stability_select(
            X_tr_numdf,
            y_tr,
            inner_folds=args.fs_inner_folds,
            repeats=args.fs_repeats,
            min_freq=args.fs_min_freq,
            fs_alpha=args.fs_alpha,
            fs_corr_th=args.fs_corr_th,
            fs_max_vars=args.fs_max_vars,
            lasso_cv=args.fs_lasso_cv,
            random_state=args.seed + 9999,
            keep_features=keep_list,
        )

        for k in keep_ok:
            if k not in GLOBAL_SEL_COLS:
                GLOBAL_SEL_COLS = [k] + GLOBAL_SEL_COLS

        print("\n[GLOBAL FS] Selected features count =", len(GLOBAL_SEL_COLS))
        print("[GLOBAL FS] Keep(raw, no Z):", keep_ok)
        print("[GLOBAL FS] Features list:\n", GLOBAL_SEL_COLS)

        pd.DataFrame({"feature": GLOBAL_SEL_COLS}).to_csv(
            os.path.join(args.out_dir, "global_selected_features.csv"),
            index=False,
            encoding="utf-8-sig",
        )

        with open(os.path.join(args.out_dir, "global_selected_features.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(GLOBAL_SEL_COLS) + "\n")

        if freqs is not None:
            freq_path = os.path.join(args.out_dir, "global_selected_freqs.csv")
            df_freq = pd.DataFrame({"feature": list(freqs.keys()), "freq": list(freqs.values())})
            df_freq = df_freq.sort_values(["freq", "feature"], ascending=[False, True]).reset_index(drop=True)
            df_freq.to_csv(freq_path, index=False, encoding="utf-8-sig")
            print("[GLOBAL FS] Saved freqs ->", os.path.abspath(freq_path))

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    oof_prob = np.zeros(len(X_tr_numdf), dtype=np.float32)
    run_summaries = []

    for fold, (tri, vai) in enumerate(skf.split(X_tr_numdf, y_tr), 1):
        run_name = f"fold{fold}"
        print(f"\n===== OUTER CV {run_name}/{args.folds} =====")

        Xtr_f = X_tr_numdf.iloc[tri][GLOBAL_SEL_COLS]
        ytr_f = y_tr[tri]
        Xva_f = X_tr_numdf.iloc[vai][GLOBAL_SEL_COLS]
        yva_f = y_tr[vai]

        z_cols, raw_cols, meds, means, stds, raw_meds = compute_mixed_stats(
            Xtr_f,
            all_cols=GLOBAL_SEL_COLS,
            keep_raw_cols=keep_ok,
        )

        Xtr_num, feat_order = encode_numeric_mixed(Xtr_f, z_cols, raw_cols, meds, means, stds, raw_meds)
        Xva_num, _ = encode_numeric_mixed(Xva_f, z_cols, raw_cols, meds, means, stds, raw_meds)

        model, history, best_auc, best_epoch, best_state = train_tabm_one(
            TabM,
            Xtr_num,
            ytr_f,
            Xva_num,
            yva_f,
            k=args.k,
            arch_type=args.arch_type,
            d_block=args.d_block,
            n_blocks=args.n_blocks,
            dropout=args.dropout,
            start_scaling_init=args.start_scaling_init,
            activation=args.activation,
            use_focal=args.use_focal,
            focal_gamma=args.focal_gamma,
            focal_alpha=args.focal_alpha,
            use_pos_weight=args.use_pos_weight,
            pos_weight_cap=args.pos_weight_cap,
            lr=args.lr,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            grad_clip=args.grad_clip,
            amp=args.amp,
            seed=args.seed + fold,
            device=device,
        )

        save_cls_curves(os.path.join(curves_dir, run_name), history, title_prefix=run_name)

        model_path = os.path.join(models_dir, f"{run_name}_best.pt")

        torch.save(
            {
                "state_dict": best_state,
                "best_epoch": int(best_epoch),
                "best_val_auc": float(best_auc),
                "features_order": feat_order,
                "keep_raw_features": keep_ok,
                "tabm_args": {
                    "k": args.k,
                    "arch_type": args.arch_type,
                    "d_block": args.d_block,
                    "n_blocks": args.n_blocks,
                    "dropout": args.dropout,
                    "start_scaling_init": args.start_scaling_init,
                    "activation": args.activation,
                },
            },
            model_path,
        )

        prep_path = os.path.join(prep_dir, f"{run_name}_prep.npz")
        save_prep_mixed(prep_path, z_cols, raw_cols, meds, means, stds, raw_meds)

        p_va = predict_proba(model, Xva_num, device=device)
        oof_prob[vai] = p_va

        auc = float(roc_auc_score(yva_f, p_va))
        acc = float(accuracy_score(yva_f, (p_va >= 0.5).astype(int)))

        print(f"[{run_name}] AUC={auc:.4f} ACC@0.5={acc:.4f} best_epoch={best_epoch} best_val_auc={best_auc:.4f}")

        run_summaries.append({
            "run": run_name,
            "best_epoch": int(best_epoch),
            "best_val_auc": float(best_auc),
            "model_path": os.path.abspath(model_path),
            "prep_path": os.path.abspath(prep_path),
            "n_features_model_input": int(len(feat_order)),
        })

    cv_auc = float(roc_auc_score(y_tr, oof_prob))
    cv_acc05 = float(accuracy_score(y_tr, (oof_prob >= 0.5).astype(int)))
    thr = best_threshold_by_acc(y_tr, oof_prob)
    cv_acc_thr = float(accuracy_score(y_tr, (oof_prob >= thr).astype(int)))

    print(f"\n[CV-OOF] AUC={cv_auc:.4f} ACC@0.5={cv_acc05:.4f} ACC@thr({thr:.6f})={cv_acc_thr:.4f}")

    pd.DataFrame({
        "row_id": np.arange(len(y_tr)),
        "y_true": y_tr.astype(int),
        "y_prob": oof_prob.astype(float),
    }).to_csv(os.path.join(args.out_dir, "oof_train_pred.csv"), index=False, encoding="utf-8-sig")

    print("\n===== FINAL FIT on TRAIN, then TEST ONCE =====")

    final_cols = GLOBAL_SEL_COLS

    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=args.final_val_ratio,
        random_state=args.seed + 2026,
    )

    tr_idx, va_idx = next(sss.split(X_tr_numdf, y_tr))

    Xtr_fit = X_tr_numdf.iloc[tr_idx][final_cols]
    ytr_fit = y_tr[tr_idx]
    Xva_fit = X_tr_numdf.iloc[va_idx][final_cols]
    yva_fit = y_tr[va_idx]

    z_cols, raw_cols, meds, means, stds, raw_meds = compute_mixed_stats(
        Xtr_fit,
        all_cols=final_cols,
        keep_raw_cols=keep_ok,
    )

    Xtr_num, feat_order = encode_numeric_mixed(Xtr_fit, z_cols, raw_cols, meds, means, stds, raw_meds)
    Xva_num, _ = encode_numeric_mixed(Xva_fit, z_cols, raw_cols, meds, means, stds, raw_meds)

    model, history, best_auc, best_epoch, best_state = train_tabm_one(
        TabM,
        Xtr_num,
        ytr_fit,
        Xva_num,
        yva_fit,
        k=args.k,
        arch_type=args.arch_type,
        d_block=args.d_block,
        n_blocks=args.n_blocks,
        dropout=args.dropout,
        start_scaling_init=args.start_scaling_init,
        activation=args.activation,
        use_focal=args.use_focal,
        focal_gamma=args.focal_gamma,
        focal_alpha=args.focal_alpha,
        use_pos_weight=args.use_pos_weight,
        pos_weight_cap=args.pos_weight_cap,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        grad_clip=args.grad_clip,
        amp=args.amp,
        seed=args.seed + 999,
        device=device,
    )

    run_name = "final"

    save_cls_curves(os.path.join(curves_dir, run_name), history, title_prefix=run_name)

    final_model_path = os.path.join(models_dir, f"{run_name}_best.pt")

    torch.save(
        {
            "state_dict": best_state,
            "best_epoch": int(best_epoch),
            "best_val_auc": float(best_auc),
            "features_order": feat_order,
            "keep_raw_features": keep_ok,
            "tabm_args": {
                "k": args.k,
                "arch_type": args.arch_type,
                "d_block": args.d_block,
                "n_blocks": args.n_blocks,
                "dropout": args.dropout,
                "start_scaling_init": args.start_scaling_init,
                "activation": args.activation,
            },
        },
        final_model_path,
    )

    final_prep_path = os.path.join(prep_dir, f"{run_name}_prep.npz")
    save_prep_mixed(final_prep_path, z_cols, raw_cols, meds, means, stds, raw_meds)

    Xte_num, _ = encode_numeric_mixed(X_te_numdf[final_cols], z_cols, raw_cols, meds, means, stds, raw_meds)
    test_prob = predict_proba(model, Xte_num, device=device)

    test_auc = float(roc_auc_score(y_te, test_prob))
    test_acc05 = float(accuracy_score(y_te, (test_prob >= 0.5).astype(int)))
    test_acc_thr = float(accuracy_score(y_te, (test_prob >= thr).astype(int)))

    print(f"[TEST] AUC={test_auc:.4f} ACC@0.5={test_acc05:.4f} ACC@thr({thr:.6f})={test_acc_thr:.4f}")

    pd.DataFrame({
        "row_id": np.arange(len(y_te)),
        "y_true": y_te.astype(int),
        "y_prob": test_prob.astype(float),
    }).to_csv(os.path.join(args.out_dir, "test_pred.csv"), index=False, encoding="utf-8-sig")

    metrics = {
        "keep_features_raw_no_z": keep_ok,
        "global_selected_features": {
            "fs_enabled": bool(args.fs_enable),
            "n_features_before_mixed": int(len(final_cols)),
            "features_file": "global_selected_features.csv",
            "freqs_file": "global_selected_freqs.csv" if args.fs_enable else None,
        },
        "cv_oof": {
            "auc": cv_auc,
            "acc@0.5": cv_acc05,
            "thr": float(thr),
            "acc@thr": cv_acc_thr,
        },
        "test": {
            "auc": test_auc,
            "acc@0.5": test_acc05,
            "acc@thr": test_acc_thr,
        },
        "fs_params": {
            "alpha": float(args.fs_alpha),
            "corr_th": float(args.fs_corr_th),
            "max_vars": int(args.fs_max_vars),
            "inner_folds": int(args.fs_inner_folds),
            "repeats": int(args.fs_repeats),
            "min_freq": float(args.fs_min_freq),
        },
        "artifacts": {
            "curves_dir": os.path.abspath(curves_dir),
            "models_dir": os.path.abspath(models_dir),
            "prep_dir": os.path.abspath(prep_dir),
            "final_best_model": os.path.abspath(final_model_path),
            "final_prep": os.path.abspath(final_prep_path),
        },
        "runs": run_summaries + [{
            "run": "final",
            "best_epoch": int(best_epoch),
            "best_val_auc": float(best_auc),
            "model_path": os.path.abspath(final_model_path),
            "prep_path": os.path.abspath(final_prep_path),
            "n_features_model_input": int(len(feat_order)),
        }],
    }

    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    if args.train_os_model:
        os_cols = GLOBAL_SEL_COLS if args.os_use_cls_features else list(X_tr_numdf.columns)

        os_metrics = train_independent_os_model(
            TabM=TabM,
            X_tr_numdf=X_tr_numdf,
            X_te_numdf=X_te_numdf,
            y_tr_time=y_tr_os_time,
            y_tr_event=y_tr_os_event,
            y_te_time=y_te_os_time,
            y_te_event=y_te_os_event,
            os_cols=os_cols,
            keep_ok=keep_ok,
            args=args,
            device=device,
        )

        metrics["independent_os_model"] = os_metrics

        with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
    else:
        print("\n[OS] independent OS model was not trained. Add --train_os_model to enable it.")

    print("\n[DONE] outputs in:", os.path.abspath(args.out_dir))


if __name__ == "__main__":
    main()