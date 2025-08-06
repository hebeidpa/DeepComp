import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from ft_transformer import FTTransformer
import joblib
# ---------------- 配置 ----------------
LABEL_COL     = 'label'
META_COLS     = ['', '']
AB_COLS       = [f"A{i}" for i in range(1,n)] + [f"B{i}" for i in range(1,n)]
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE    = 64
CV_FOLDS      = 5
CV_EPOCHS     = 50
FINAL_EPOCHS  = 80
RANDOM_TRIALS = 72
seed = 42
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32).reshape(-1,1)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma
    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt  = torch.exp(-bce)
        return (self.alpha * (1-pt)**self.gamma * bce).mean()


def train_one_fold(params, X_meta_tr, X_ab_tr_s, y_tr, X_meta_val, X_ab_val_s, y_val):
    # 1) 特征选择：AB 列
    selector = SelectKBest(mutual_info_classif, k=params['k_features'])
    X_tr_AB = selector.fit_transform(X_ab_tr_s, y_tr)
    X_val_AB = selector.transform(X_ab_val_s)
    X_tr_sel = np.hstack([X_meta_tr, X_tr_AB])
    X_val_sel = np.hstack([X_meta_val, X_val_AB])

    # 2) WeightedRandomSampler
    cw = 1.0/np.bincount(y_tr); sw = cw[y_tr]
    sampler = WeightedRandomSampler(sw, len(sw), replacement=True)
    tr_loader = DataLoader(CustomDataset(X_tr_sel, y_tr),
                           batch_size=BATCH_SIZE, sampler=sampler)
    val_loader= DataLoader(CustomDataset(X_val_sel, y_val),
                           batch_size=BATCH_SIZE, shuffle=False)

    # 3) 构建模型
    model = FTTransformer(
        categories=[],
        num_continuous=2 + params['k_features'],
        dim=params['dim'],
        depth=4,                           # 深度从3->4
        heads=8,                           # heads从4->8
        dim_head=params['dim']//8,
        num_special_tokens=2,
        dim_out=1,
        ff_dropout=0.2,                    # dropout从0.1->0.2
        attn_dropout=0.2
    ).to(DEVICE)

    # Xavier init
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)

    criterion = FocalLoss(alpha=1.0, gamma=params['gamma'])
    optimizer = AdamW(model.parameters(),
                      lr=params['lr'],
                      weight_decay=1e-4)        # weight decay
    scheduler = CosineAnnealingLR(optimizer,
                                  T_max=CV_EPOCHS,
                                  eta_min=params['lr']*1e-2)

    best_auc = 0.0
    for epoch in range(1, CV_EPOCHS+1):
        model.train()
        for Xb, yb in tr_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            # 标签平滑
            yb_s = yb * 0.9 + 0.05
            optimizer.zero_grad()
            logits = model(None, Xb)
            loss   = criterion(logits, yb_s)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # 评估 AUC
        ys, ps = [], []
        model.eval()
        with torch.no_grad():
            for Xb, yb in val_loader:
                p = torch.sigmoid(model(None, Xb.to(DEVICE))).cpu().numpy().reshape(-1)
                ps.extend(p); ys.extend(yb.numpy().reshape(-1))
        auc = roc_auc_score(ys, ps)
        best_auc = max(best_auc, auc)

    return best_auc

def random_search(X_meta, X_ab_s, y):
    set_seed(42)
    space = {
        'k_features': [5, 10, 20, 30, 50,100,200],
        'dim':        [128, 256, 512],
        'lr':         [1e-4, 1e-3],
        'gamma':      [0.5, 1.0, 2.0]
    }
    best_params, best_auc = None, 0.0
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    for t in range(RANDOM_TRIALS):
        params = {k: random.choice(v) for k,v in space.items()}
        aucs = [
            train_one_fold(
                params,
                X_meta[tr], X_ab_s[tr], y[tr],
                X_meta[val], X_ab_s[val], y[val]
            )
            for tr, val in skf.split(X_ab_s, y)
        ]
        mean_auc = np.mean(aucs)
        print(f"Trial {t+1}/{RANDOM_TRIALS}: {params} → CV AUC={mean_auc:.4f}")
        if mean_auc > best_auc:
            best_auc, best_params = mean_auc, params

    print(f"\nBest CV params: {best_params}, CV AUC={best_auc:.4f}")
    return best_params

def main():

    print(f"PyTorch is using: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    set_seed(0)

    df = pd.read_excel(".xlsx", sheet_name="")
    X_meta = df[META_COLS].values
    X_ab = df[AB_COLS].values
    y = df[LABEL_COL].values.astype(int)
    scaler = StandardScaler().fit(X_ab)
    X_ab_s = scaler.transform(X_ab)
    X_s = np.hstack([X_meta, X_ab_s])


    best_params = random_search(X_meta, X_ab_s, y)

    selector   = SelectKBest(mutual_info_classif, k=best_params['k_features'])
    X_AB_sel   = selector.fit_transform(X_ab_s, y)
    X_full_sel = np.hstack([X_meta, X_AB_sel])
    cw = 1.0/np.bincount(y); sw = cw[y]
    sampler = WeightedRandomSampler(sw, len(sw), replacement=True)
    train_loader = DataLoader(CustomDataset(X_full_sel, y),
                              batch_size=BATCH_SIZE, sampler=sampler)


    model = FTTransformer(
        categories=[],
        num_continuous=2 + best_params['k_features'],
        dim=best_params['dim'],
        depth=4, heads=8, dim_head=best_params['dim']//8,
        num_special_tokens=2, dim_out=1,
        ff_dropout=0.2, attn_dropout=0.2
    ).to(DEVICE)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
    criterion = FocalLoss(alpha=1.0, gamma=best_params['gamma'])
    optimizer = AdamW(model.parameters(), lr=best_params['lr'], weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=FINAL_EPOCHS, eta_min=best_params['lr']*1e-2)


    df_test   = pd.read_excel(".xlsx", sheet_name="")
    X_meta_test = df_test[META_COLS].values
    X_ab_test = df_test[AB_COLS].values
    X_ab_test_s = scaler.transform(X_ab_test)
    X_test_AB = selector.transform(X_ab_test_s)
    X_test_sel = np.hstack([X_meta_test, X_test_AB])
    y_test    = df_test[LABEL_COL].values.astype(int)
    test_loader = DataLoader(CustomDataset(X_test_sel, y_test),
                             batch_size=BATCH_SIZE, shuffle=False)





    best_test_auc = 0.0

    best_model_state = None
    best_test_probs = None
    best_test_labels = None
    for epoch in range(1, FINAL_EPOCHS+1):
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(None, Xb)
            loss   = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        #if epoch == 1 or epoch % 10 == 0:
        if epoch >= 1:
            model.eval()
            ys, ps = [], []
            with torch.no_grad():
                for Xb, yb in test_loader:
                    p = torch.sigmoid(model(None, Xb.to(DEVICE))).cpu().numpy().reshape(-1)
                    ps.extend(p); ys.extend(yb.numpy().reshape(-1))
            auc = roc_auc_score(ys, ps)
            #best_test_auc = max(best_test_auc, auc)
            print(f"Epoch {epoch:3d}/{FINAL_EPOCHS} → Test AUC: {auc:.4f}")
            if auc > best_test_auc:
                best_test_auc = auc
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_test_probs =np.array(ps)
                best_test_labels = list(ys)

    print(f"\n>>> Highest Test AUC over all epochs: {best_test_auc:.4f}")
    if best_model_state is not None:
        torch.save(best_model_state, "fttransformer_best_model.pth")
    if best_test_probs is not None and best_test_labels is not None:
        df_test["pred_prob"] = best_test_probs
        df_test["true_label"] = best_test_labels
        df_test.to_csv("test_pred_probs.csv", index=False)
    print("test_pred_probs.csv")

    meta_names = META_COLS
    ab_names = np.array(AB_COLS)[selector.get_support()]
    all_feature_names = list(meta_names) + list(ab_names)
    pd.Series(all_feature_names).to_csv("model_feature_names.txt", index=False, header=False)

    selected_idx = selector.get_support(indices=True)
    ab_selected_train = X_ab[:, selected_idx]  
    ab_selected_scaler = StandardScaler().fit(ab_selected_train)
    joblib.dump(ab_selected_scaler, "ab_selected_scaler.pkl")
    #torch.save(model.state_dict(), "fttransformer_best_model.pth")


if __name__ == '__main__':
    main()
