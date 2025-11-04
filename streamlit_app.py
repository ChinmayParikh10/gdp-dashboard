
import os, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st
import torch
import torch.nn as nn

from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ------------------------------
# Page & theme
# ------------------------------
st.set_page_config(page_title="TDAM – EV Battery SoH & RUL", layout="wide")
st.title("⚡ TDAM — EV Battery Health (SoH) & RUL")
st.caption("Windowed z-scores ➜ fast temporal attention ➜ RandomForest heads. Trains on your bundled CSV and supports manual prediction.")

torch.manual_seed(42)
np.random.seed(42)

# ------------------------------
# Load CSV – your dataset path
# ------------------------------
CANDIDATES = [
    "/content/Battery_RUL_with_features.csv",  # Colab path (your file)
    "Battery_RUL_with_features.csv",           # same folder as app.py
    "data/Battery_RUL_with_features.csv",      # repo-style path
]

def load_csv():
    for p in CANDIDATES:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                return df, p
            except Exception:
                pass
    return None, None

df_raw, csv_path = load_csv()
if df_raw is None:
    st.error("File `Battery_RUL_with_features.csv` not found. Upload it to **/content** in Colab or next to app.py.")
    st.stop()

if "Cycle_Index" not in df_raw.columns:
    st.error("Dataset must have a **Cycle_Index** column.")
    st.stop()

st.success(f"Loaded: `{csv_path}`  shape={df_raw.shape}")
with st.expander("Preview & Columns", expanded=False):
    st.dataframe(df_raw.head(20), use_container_width=True)
    st.code(", ".join(df_raw.columns))

# ------------------------------
# Sidebar config
# ------------------------------
st.sidebar.header("Training Config")
T = st.sidebar.number_input("Window length (T)", min_value=5, max_value=200, value=50, step=1)
H = st.sidebar.number_input("Attention hidden size (H)", min_value=16, max_value=512, value=64, step=16)
N_SPLITS = st.sidebar.number_input("GroupKFold splits", min_value=2, max_value=10, value=5, step=1)
N_TREES = st.sidebar.number_input("RandomForest trees", min_value=50, max_value=1000, value=250, step=50)
LIMIT_CYCLES_MAX = st.sidebar.number_input("LIMIT_CYCLES_MAX (0 = all)", min_value=0, value=0, step=100)
st.sidebar.caption("If you see 'no windows', reduce T or set LIMIT_CYCLES_MAX to 0.")

# ------------------------------
# Cell_ID stitching (Hungarian; SciPy optional)
# ------------------------------
try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

def _hungarian_fallback(cost):
    n_rows, n_cols = cost.shape
    assigned_cols = set()
    row_to_col = [-1]*n_rows
    for i in range(n_rows):
        order = np.argsort(cost[i])
        for j in order:
            if j not in assigned_cols:
                row_to_col[i] = j
                assigned_cols.add(j)
                break
    return np.arange(n_rows), np.array(row_to_col)

def _build_cost(A, B, cap=10.0, soh_a=None, soh_b=None, soh_cap=2.0):
    diff = A[:, None, :] - B[None, :, :]
    d = np.sqrt(np.sum(diff**2, axis=2))
    if soh_a is not None and soh_b is not None:
        d += np.abs(soh_b[None, :] - soh_a[:, None]) / max(soh_cap, 1e-6)
    return np.minimum(d, cap)

@st.cache_data(show_spinner=False)
def stitch_cell_ids(df: pd.DataFrame) -> pd.DataFrame:
    base = ["Discharge Time (s)","Charging time (s)","Decrement 3.6-3.4V (s)",
            "Max. Voltage Dischar. (V)","Min. Voltage Charg. (V)",
            "Discharge_Time_Norm","Voltage_Range","Delta_SOH"]
    rolling = [c for c in df.columns if c.lower().startswith("rolling")]
    feats = [c for c in base + rolling if c in df.columns]
    if not feats:
        raise ValueError("No suitable feature columns found to reconstruct Cell_ID.")

    X = df[feats].copy().replace([np.inf, -np.inf], np.nan).fillna(df[feats].median())
    mu, sigma = X.mean(), X.std().replace(0, 1.0)
    Z = ((X - mu) / sigma).to_numpy()
    soh = df["SOH"].to_numpy() if "SOH" in df.columns else None

    df = df.sort_values("Cycle_Index").reset_index(drop=True)
    Z = Z[df.index]
    soh_sorted = soh[df.index] if soh is not None else None

    cycles = np.sort(df["Cycle_Index"].unique())
    cycle_rows = {c: np.where(df["Cycle_Index"].values == c)[0] for c in cycles}

    first = cycles[0]
    idx_first = cycle_rows[first]
    cell_ids = np.full(len(df), -1, dtype=int)
    cell_ids[idx_first] = np.arange(len(idx_first), dtype=int)
    next_new = int(len(idx_first))

    BIG = 1e6
    for t, t_next in zip(cycles[:-1], cycles[1:]):
        a = cycle_rows[t]; b = cycle_rows[t_next]
        A = Z[a, :]; B = Z[b, :]
        soh_a = soh_sorted[a] if soh_sorted is not None else None
        soh_b = soh_sorted[b] if soh_sorted is not None else None
        cost = _build_cost(A, B, cap=10.0, soh_a=soh_a, soh_b=soh_b, soh_cap=2.0)

        n_prev, n_next = cost.shape
        n = max(n_prev, n_next)
        cost_sq = np.full((n, n), BIG, dtype=float)
        cost_sq[:n_prev, :n_next] = cost

        if SCIPY_AVAILABLE:
            row_ind, col_ind = linear_sum_assignment(cost_sq)
        else:
            row_ind, col_ind = _hungarian_fallback(cost_sq)

        used_next = set()
        for r, c in zip(row_ind, col_ind):
            if r < n_prev and c < n_next and cost_sq[r, c] < 1e5:
                cell_ids[b[c]] = cell_ids[a[r]]
                used_next.add(c)
        for j in range(n_next):
            if j not in used_next:
                cell_ids[b[j]] = next_new; next_new += 1
    out = df.copy()
    out["Cell_ID"] = cell_ids.astype(int)
    return out

# ------------------------------
# Features, z-score, windows
# ------------------------------
def autodetect_features(df: pd.DataFrame):
    base = ["Discharge Time (s)","Charging time (s)","Decrement 3.6-3.4V (s)",
            "Max. Voltage Dischar. (V)","Min. Voltage Charg. (V)",
            "Discharge_Time_Norm","Voltage_Range","Delta_SOH"]
    rolling = [c for c in df.columns if c.lower().startswith("rolling")]
    feats = [c for c in base + rolling if c in df.columns]
    if not feats:
        exclude = {"Cycle_Index","Cell_ID","SOH","RUL"}
        feats = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
    return feats

def zscore_inplace(df: pd.DataFrame, feat_cols):
    X = df[feat_cols].copy().replace([np.inf,-np.inf], np.nan).fillna(df[feat_cols].median())
    mu, sigma = X.mean(), X.std().replace(0,1.0)
    Z = (X - mu) / sigma
    for c in feat_cols:
        df[f"_z_{c}"] = Z[c]
    return mu, sigma

def build_windows(df: pd.DataFrame, T: int, feat_cols):
    y_soh = "SOH" if "SOH" in df.columns else None
    y_rul = "RUL" if "RUL" in df.columns else None
    Xb, yS, yR, gids = [], [], [], []
    for cid, g in df.groupby("Cell_ID"):
        g = g.sort_values("Cycle_Index")
        Xz = g[[f"_z_{c}" for c in feat_cols]].to_numpy()
        ys = g[y_soh].to_numpy() if y_soh else None
        yr = g[y_rul].to_numpy() if y_rul else None
        for t in range(T, len(g)):
            Xb.append(Xz[t-T:t]); gids.append(cid)
            if y_soh: yS.append(ys[t])
            if y_rul: yR.append(yr[t])
    if len(Xb) == 0:
        return np.empty((0, T, len(feat_cols))), (None if y_soh is None else np.array([])), \
               (None if y_rul is None else np.array([])), np.array([], dtype=int)
    return np.array(Xb), (None if y_soh is None else np.array(yS)), \
           (None if y_rul is None else np.array(yR)), np.array(gids, dtype=int)

# ------------------------------
# Encoder (fixed __init__)
# ------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)
    def forward(self, x):  # (B,T,D)
        Tloc = x.size(1); return x + self.pe[:Tloc, :]

class FastTemporalAttention(nn.Module):
    def __init__(self, in_f, hid=64):
        super().__init__()
        self.proj = nn.Linear(in_f, hid)
        self.pos  = PositionalEncoding(hid)
        self.Wq = nn.Linear(hid, hid, bias=False)
        self.Wk = nn.Linear(hid, hid, bias=False)
        self.Wv = nn.Linear(hid, hid, bias=False)
    def forward(self, x):  # x: (B,T,F)
        h = self.pos(self.proj(x))
        q = self.Wq(h[:, -1, :]).unsqueeze(1)
        K = self.Wk(h); V = self.Wv(h)
        logits = (q @ K.transpose(1,2)) / math.sqrt(K.size(-1))
        w = torch.softmax(logits, dim=-1)
        ctx = (w @ V).squeeze(1)
        return ctx, w.squeeze(1)

@torch.no_grad()
def encode_fast(X_seq, hid=64, batch_size=2048):
    if X_seq is None or len(X_seq) == 0:
        return np.empty((0, hid)), np.empty((0, 0))
    in_f = X_seq.shape[-1]
    enc = FastTemporalAttention(in_f=in_f, hid=hid).eval()
    C_list, W_list = [], []
    for i in range(0, len(X_seq), batch_size):
        xb = torch.from_numpy(X_seq[i:i+batch_size]).float()
        Cb, Wb = enc(xb)
        C_list.append(Cb.numpy()); W_list.append(Wb.numpy())
    return np.vstack(C_list), np.vstack(W_list)

# ------------------------------
# Train (cached)
# ------------------------------
@st.cache_resource(show_spinner=True)
def train_all(df_raw: pd.DataFrame, T: int, H: int, n_splits: int, n_trees: int, limit_cycles: int):
    df = df_raw.copy()
    if limit_cycles and limit_cycles > 0:
        df = df[df["Cycle_Index"] <= int(limit_cycles)].copy()

    if "Cell_ID" not in df.columns:
        df = stitch_cell_ids(df)
    else:
        df["Cell_ID"] = df["Cell_ID"].astype(int)

    feat_cols = autodetect_features(df)
    mu, sigma = zscore_inplace(df, feat_cols)
    X_seq, y_soh, y_rul, groups = build_windows(df, T, feat_cols)

    # nothing to train
    if len(X_seq) == 0:
        meta = {"feat_cols": feat_cols, "mu": mu, "sigma": sigma, "T": T, "H": H}
        cache = {"df": df, "C": np.empty((0,H)), "W": np.empty((0,0)),
                 "y_soh": y_soh, "y_rul": y_rul, "groups": groups}
        return meta, {"rf_soh": None, "rf_rul": None}, {"SoH": None, "RUL": None}, cache

    C, W = encode_fast(X_seq, hid=H)

    def cv_eval(C, y, groups):
        if y is None or len(y) == 0: return None
        gkf = GroupKFold(n_splits=n_splits)
        preds, trues = [], []
        for tr, te in gkf.split(C, y, groups=groups):
            rf = RandomForestRegressor(n_estimators=n_trees, n_jobs=-1, random_state=42)
            rf.fit(C[tr], y[tr])
            preds.append(rf.predict(C[te])); trues.append(y[te])
        p, t = np.concatenate(preds), np.concatenate(trues)
        return {"MAE": float(mean_absolute_error(t,p)), "R2": float(r2_score(t,p))}

    cv_soh = cv_eval(C, y_soh, groups)
    cv_rul = cv_eval(C, y_rul, groups)

    rf_soh = RandomForestRegressor(n_estimators=n_trees, n_jobs=-1, random_state=42).fit(C, y_soh) if (y_soh is not None and len(y_soh)>0) else None
    rf_rul = RandomForestRegressor(n_estimators=n_trees, n_jobs=-1, random_state=42).fit(C, y_rul) if (y_rul is not None and len(y_rul)>0) else None

    meta = {"feat_cols": feat_cols, "mu": mu, "sigma": sigma, "T": T, "H": H}
    cache = {"df": df, "C": C, "W": W, "y_soh": y_soh, "y_rul": y_rul, "groups": groups}
    return meta, {"rf_soh": rf_soh, "rf_rul": rf_rul}, {"SoH": cv_soh, "RUL": cv_rul}, cache

# ------------------------------
# Train / retrain button
# ------------------------------
if "trained" not in st.session_state:
    st.session_state.trained = False

if st.button("🚀 Train / Re-train on CSV", type="primary"):
    st.session_state.trained = False

if not st.session_state.trained:
    with st.spinner("Training…"):
        meta, models, cv, cache = train_all(df_raw, int(T), int(H), int(N_SPLITS), int(N_TREES),
                                            int(LIMIT_CYCLES_MAX))
        st.session_state.meta = meta
        st.session_state.models = models
        st.session_state.cv = cv
        st.session_state.cache = cache
        st.session_state.trained = True

meta = st.session_state.meta
models = st.session_state.models
cv = st.session_state.cv
cache = st.session_state.cache

# ------------------------------
# Metrics
# ------------------------------
with st.expander("Overall Evaluation", expanded=True):
    rows = []
    if cv["SoH"] is not None: rows.append(["Random Forest — SoH", cv["SoH"]["MAE"], cv["SoH"]["R2"]])
    if cv["RUL"] is not None: rows.append(["Random Forest — RUL", cv["RUL"]["MAE"], cv["RUL"]["R2"]])
    if rows:
        dfm = pd.DataFrame(rows, columns=["Model","MAE","R²"]).set_index("Model")
        st.dataframe(dfm, use_container_width=True)
    else:
        st.info("No metrics (labels missing or not enough windows).")

# ------------------------------
# Manual input → Predict SoH & RUL
# ------------------------------
st.header("📝 Manual Input (single time step replicated to T)")
st.write("Enter **feature values** for one time step. We z-score with training μ/σ, replicate to a T×F window, encode, and predict.")

feat_cols = meta["feat_cols"]
mu, sigma = meta["mu"], meta["sigma"]
T_cfg, H_cfg = int(meta["T"]), int(meta["H"])

defaults = cache["df"][feat_cols].median(numeric_only=True)
columns3 = st.columns(3)
user_vals = {}
for i, c in enumerate(feat_cols):
    with columns3[i % 3]:
        user_vals[c] = st.number_input(c, value=float(defaults.get(c, 0.0)))

if st.button("🔮 Predict SoH & RUL", type="primary"):
    x = np.array([[user_vals[c] for c in feat_cols]], dtype=float)
    z = (x - mu.values) / sigma.replace(0, 1.0).values
    window = np.tile(z, (T_cfg, 1))[None, :, :]   # (1, T, F)
    C_now, W_now = encode_fast(window, hid=H_cfg, batch_size=1)

    pred_soh = pred_rul = None
    if models["rf_soh"] is not None: pred_soh = float(models["rf_soh"].predict(C_now)[0])
    if models["rf_rul"] is not None: pred_rul = float(models["rf_rul"].predict(C_now)[0])

    c1, c2 = st.columns(2)
    with c1:
        st.success("Predictions")
        st.write({"SoH": pred_soh, "RUL": pred_rul})
    with c2:
        st.caption("Attention weights for the generated window")
        if W_now.size > 0:
            fig, ax = plt.subplots()
            ax.plot(W_now[0]); ax.set_xlabel("Index in window"); ax.set_ylabel("Weight")
            st.pyplot(fig)
        else:
            st.info("No attention weights (empty window).")

st.caption("Tip: If you get a 'no windows' warning during training, reduce **T** or set **LIMIT_CYCLES_MAX = 0**.")

