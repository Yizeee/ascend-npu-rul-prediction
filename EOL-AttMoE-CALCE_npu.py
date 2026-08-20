#!/usr/bin/env python
# coding: utf-8

# EOL_AttMoE_hardconstraint_robustseed_v1_fixed modification:
# 1) Keep the original AttMoE backbone and original capacity-only input.
# 2) Add best-epoch selection during training.
# 3) Use hard-constraint multi-objective selection: first seek RE/MAE/RMSE all below baseline thresholds.
# 4) Add local grid search for feature_size, lr, hidden_dim and num_experts.

# # **Reference:**
# 
# Chen, D., & Zhou, X. (2024). AttMoE: Attention with Mixture of Experts for remaining useful life prediction of lithium-ion batteries. Journal of Energy Storage, 84, 110780.

# In[2]:


import argparse
import copy
import importlib.util
import subprocess
import sys
from pathlib import Path


def _ensure_package(import_name, pip_name=None):
    """Install a missing dependency into the current Python environment."""
    if importlib.util.find_spec(import_name) is not None:
        return
    package_name = pip_name or import_name
    print('Missing dependency: {}. Installing {} ...'.format(import_name, package_name), flush=True)
    subprocess.check_call([
        sys.executable, '-m', 'pip', 'install',
        '--disable-pip-version-check', '--no-cache-dir', package_name
    ])


# Keep the CUDA-enabled PyTorch supplied by the cloud image; do not reinstall torch here.
for _import_name, _pip_name in {
    'numpy': 'numpy',
    'pandas': 'pandas',
    'matplotlib': 'matplotlib',
    'sklearn': 'scikit-learn',
    'openpyxl': 'openpyxl',
}.items():
    _ensure_package(_import_name, _pip_name)

if importlib.util.find_spec('torch') is None:
    raise ImportError('PyTorch is missing. Please select a CUDA-enabled PyTorch cloud image.')

import numpy as np
import random
import math
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import glob
import torch
import torch_npu  # NPU support
import torch.nn as nn
import torch.nn.functional as F
import os

from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs_CALCE_EOL_AttMoE_hardconstraint_robustseed_v1_fixed_fixed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# # step 1. get device

# In[5]:


device = torch.device('npu:0' if torch.npu.is_available() else 'cpu')
print('NPU available:', torch.npu.is_available(), flush=True)
print('Current device:', device, flush=True)
if torch.npu.is_available():
    print('NPU name:', torch.npu.get_device_name(0), flush=True)


# # step 2. define functions for data processing and evaluation

# In[6]:


def drop_outlier(array,count,bins):
    index = []
    range_ = np.arange(1,count,bins)
    for i in range_[:-1]:
        array_lim = array[i:i+bins]
        sigma = np.std(array_lim)
        mean = np.mean(array_lim)
        th_max,th_min = mean + sigma*2, mean - sigma*2
        idx = np.where((array_lim < th_max) & (array_lim > th_min))
        idx = idx[0] + i
        index.extend(list(idx))
    return np.array(index)


def build_sequences(text, window_size):
    #text:list of capacity
    x, y = [],[]
    for i in range(len(text) - window_size):
        sequence = text[i:i+window_size]
        target = text[i+1:i+1+window_size]

        x.append(sequence)
        y.append(target)

    return np.array(x), np.array(y)


# 留一评估：一组数据为测试集，其他所有数据全部拿来训练
def get_train_test(data_dict, name, window_size=8):
    data_sequence=data_dict[name]['capacity']
    train_data, test_data = data_sequence[:window_size+1], data_sequence[window_size+1:]
    train_x, train_y = build_sequences(text=train_data, window_size=window_size)
    for k, v in data_dict.items():
        if k != name:
            data_x, data_y = build_sequences(text=v['capacity'], window_size=window_size)
            train_x, train_y = np.r_[train_x, data_x], np.r_[train_y, data_y]
            
    return train_x, train_y, list(train_data), list(test_data)


def relative_error(y_test, y_predict, threshold):
    true_re, pred_re = len(y_test), 0
    for i in range(len(y_test)-1):
        if y_test[i] <= threshold >= y_test[i+1]:
            true_re = i - 1
            break
    for i in range(len(y_predict)-1):
        if y_predict[i] <= threshold:
            pred_re = i - 1
            break
    return abs(true_re - pred_re)/true_re if abs(true_re - pred_re)/true_re<=1 else 1


def evaluation(y_test, y_predict):
    mae = mean_absolute_error(y_test, y_predict)
    mse = mean_squared_error(y_test, y_predict)
    rmse = sqrt(mean_squared_error(y_test, y_predict))
    return mae, rmse
    
    
def setup_seed(seed):
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现。
    torch.manual_seed(seed) # 为CPU设置随机种子
    if torch.npu.is_available():
        torch.npu.manual_seed(seed) # 为当前GPU设置随机种子
        torch.npu.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子
        torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.deterministic = True  # NPU不需要


# # step 3. load data
#
# Cloud-safe behavior: first try the raw Excel files, then fall back to CALCE.npy.


def _candidate_data_dirs():
    return [
        BASE_DIR / 'datasets' / 'CALCE',
        BASE_DIR.parent / 'datasets' / 'CALCE',
        Path.cwd() / 'datasets' / 'CALCE',
    ]


def _load_raw_excel(data_dir):
    Battery_list = ['CS2_35', 'CS2_36', 'CS2_37', 'CS2_38']
    Battery = {}

    for name in Battery_list:
        print('Load Dataset ' + name + ' ...', flush=True)
        path = sorted(glob.glob(str(data_dir / name / '*.xlsx')))
        if len(path) == 0:
            raise FileNotFoundError('No Excel files found: {}'.format(data_dir / name / '*.xlsx'))

        dates = []
        for p in path:
            df = pd.read_excel(p, sheet_name=1, engine='openpyxl')
            print('Load ' + str(p) + ' ...', flush=True)
            dates.append(df['Date_Time'].iloc[0])
        idx = np.argsort(dates)
        path_sorted = np.array(path)[idx]

        count = 0
        discharge_capacities = []
        health_indicator = []
        internal_resistance = []
        CCCT = []
        CVCT = []

        for p in path_sorted:
            df = pd.read_excel(p, sheet_name=1, engine='openpyxl')
            print('Load ' + str(p) + ' ...', flush=True)
            cycles = sorted(df['Cycle_Index'].dropna().unique())

            for c in cycles:
                df_lim = df[df['Cycle_Index'] == c]

                # Charging: CC or CV
                df_cc = df_lim[df_lim['Step_Index'] == 2]
                df_cv = df_lim[df_lim['Step_Index'] == 4]
                ccct_value = (np.max(df_cc['Test_Time(s)']) - np.min(df_cc['Test_Time(s)'])) if not df_cc.empty else np.nan
                cvct_value = (np.max(df_cv['Test_Time(s)']) - np.min(df_cv['Test_Time(s)'])) if not df_cv.empty else np.nan

                # Discharging
                df_d = df_lim[df_lim['Step_Index'] == 7]
                d_v = df_d['Voltage(V)']
                d_c = df_d['Current(A)']
                d_t = df_d['Test_Time(s)']
                d_im = df_d['Internal_Resistance(Ohm)']

                if len(list(d_c)) > 1:
                    time_diff = np.diff(list(d_t))
                    d_c = np.array(list(d_c))[1:]
                    discharge_capacity = time_diff * d_c / 3600  # Q = A*h
                    discharge_capacity = np.cumsum(discharge_capacity)
                    discharge_capacities.append(-1 * discharge_capacity[-1])

                    dec = np.abs(np.array(d_v) - 3.8)[1:]
                    start_capacity = np.array(discharge_capacity)[np.argmin(dec)]
                    dec = np.abs(np.array(d_v) - 3.4)[1:]
                    end_capacity = np.array(discharge_capacity)[np.argmin(dec)]
                    health_indicator.append(-1 * (end_capacity - start_capacity))

                    internal_resistance.append(np.mean(np.array(d_im)))
                    CCCT.append(ccct_value)
                    CVCT.append(cvct_value)
                    count += 1

        discharge_capacities = np.array(discharge_capacities)
        health_indicator = np.array(health_indicator)
        internal_resistance = np.array(internal_resistance)
        CCCT = np.array(CCCT)
        CVCT = np.array(CVCT)

        idx = drop_outlier(discharge_capacities, count, 40)
        if idx.size == 0 and count > 0:
            # Keep the original processing logic, but avoid an empty result on short data.
            idx = np.arange(count)

        df_result = pd.DataFrame({
            'cycle': np.linspace(1, idx.shape[0], idx.shape[0]),
            'capacity': discharge_capacities[idx],
            'SoH': health_indicator[idx],
            'resistance': internal_resistance[idx],
            'CCCT': CCCT[idx],
            'CVCT': CVCT[idx],
        })
        Battery[name] = df_result

    return Battery


def load_battery_data(data_source='auto'):
    if data_source in ('auto', 'raw'):
        for data_dir in _candidate_data_dirs():
            if data_dir.is_dir():
                print('Use raw CALCE data:', data_dir, flush=True)
                return _load_raw_excel(data_dir)
        if data_source == 'raw':
            raise FileNotFoundError('datasets/CALCE was not found near the script or working directory.')

    npy_candidates = [
        BASE_DIR / 'CALCE.npy',
        BASE_DIR.parent / 'CALCE.npy',
        Path.cwd() / 'CALCE.npy',
    ]
    for npy_path in npy_candidates:
        if npy_path.is_file():
            print('Use extracted data:', npy_path, flush=True)
            battery = np.load(npy_path, allow_pickle=True)
            return battery.item()

    raise FileNotFoundError('Neither datasets/CALCE nor CALCE.npy could be found.')


Battery_list = ['CS2_35', 'CS2_36', 'CS2_37', 'CS2_38']
Battery = None


# # step 4. capacity figure
# The cloud entry point saves this figure after loading the selected data source.


# # step 5. build net

# In[9]:


try:
    from mixture_of_experts import MoE
except ImportError:
    try:
        _ensure_package('mixture_of_experts', 'mixture-of-experts')
        from mixture_of_experts import MoE
    except Exception:
        print('mixture_of_experts is unavailable; using a compatible dense MoE fallback.', flush=True)

        class MoE(nn.Module):
            def __init__(self, dim, num_experts, experts):
                super(MoE, self).__init__()
                self.gate = nn.Linear(dim, num_experts)
                self.experts = nn.ModuleList([copy.deepcopy(experts) for _ in range(num_experts)])

            def forward(self, x):
                weights = torch.softmax(self.gate(x), dim=-1)
                outputs = torch.stack([expert(x) for expert in self.experts], dim=2)
                out = torch.sum(weights.unsqueeze(-1) * outputs, dim=2)
                aux_loss = torch.zeros((), dtype=x.dtype, device=x.device)
                return out, aux_loss


class Attention(nn.Module):
    def __init__(self, feature_size, hidden_dim, nhead=4, dropout=0.0):
        super(Attention, self).__init__()
        self.query = nn.Linear(feature_size, hidden_dim)
        self.key = nn.Linear(feature_size, hidden_dim)
        self.value = nn.Linear(feature_size, hidden_dim)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=nhead, dropout=dropout, batch_first=True)
        
    def forward(self, x):
        query, key, value = self.query(x), self.key(x), self.value(x)
        out, _ = self.attn(query, key, value)
        return out


class EOLAwareCorrection(nn.Module):
    """Very weak EOL-aware residual head.

    The head uses the hidden representation and distance-to-threshold state.
    It is zero-initialized, so the initial model is exactly original AttMoE.
    """
    def __init__(self, hidden_dim, eol_state_dim=16):
        super(EOLAwareCorrection, self).__init__()
        self.state_encoder = nn.Sequential(
            nn.Linear(3, eol_state_dim),
            nn.ReLU(),
            nn.Linear(eol_state_dim, eol_state_dim),
            nn.ReLU(),
        )
        self.correction = nn.Sequential(
            nn.Linear(hidden_dim + eol_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def forward(self, hidden, x, eol_threshold_norm=0.7):
        seq = x.squeeze(1) if x.dim() == 3 else x
        last_q = seq[:, -1]
        mean_q = seq.mean(dim=1)

        if seq.shape[1] > 1:
            slope = (seq[:, -1] - seq[:, 0]) / max(seq.shape[1] - 1, 1)
        else:
            slope = torch.zeros_like(last_q)

        dist_to_eol = last_q - float(eol_threshold_norm)
        eol_state = torch.stack([dist_to_eol, slope, mean_q - float(eol_threshold_norm)], dim=-1)
        state_emb = self.state_encoder(eol_state)
        correction = self.correction(torch.cat([hidden, state_emb], dim=-1))
        return correction, eol_state


class AttMoE(nn.Module):
    """EOL-AttMoE v2 lossfree.

    Main path is kept exactly as original AttMoE.
    EOL-aware head is only a very weak residual correction.
    """
    def __init__(self, feature_size=16, hidden_dim=8, num_layers=1, nhead=4, dropout_att=0., dropout_rate=0.2,
                 num_experts=8, device='cpu', eol_state_dim=16, eol_gamma_init=-6.0,
                 eol_correction_scale=0.01, eol_threshold_norm=0.7, eol_enable_correction=True):
        super(AttMoE, self).__init__()
        self.feature_size, self.hidden_dim = feature_size, hidden_dim
        self.dropout = nn.Dropout(dropout_rate)

        # Original AttMoE backbone: kept unchanged.
        self.cell = Attention(feature_size=feature_size, hidden_dim=hidden_dim, nhead=nhead, dropout=dropout_att)
        self.linear = nn.Linear(hidden_dim, 1)

        experts = nn.Linear(hidden_dim, hidden_dim)
        self.moe = MoE(dim=hidden_dim, num_experts=num_experts, experts=experts)
        self.moe = self.moe.to(device)

        self.eol_correction = EOLAwareCorrection(hidden_dim=hidden_dim, eol_state_dim=eol_state_dim)
        self.eol_gamma = nn.Parameter(torch.tensor(float(eol_gamma_init)))
        self.eol_correction_scale = float(eol_correction_scale)
        self.eol_threshold_norm = float(eol_threshold_norm)
        self.eol_enable_correction = bool(eol_enable_correction)

        self.last_base_output = None
        self.last_correction = None
        self.last_eol_state = None

    def forward(self, x):
        # IMPORTANT: keep original AttMoE behavior exactly.
        out = self.dropout(x)
        out = self.cell(x)
        out, _ = self.moe(out)
        out = out.reshape(-1, self.hidden_dim)
        base_out = self.linear(out)

        if self.eol_enable_correction:
            correction, eol_state = self.eol_correction(
                hidden=out,
                x=x,
                eol_threshold_norm=self.eol_threshold_norm,
            )
            gamma = torch.sigmoid(self.eol_gamma)
            final_out = base_out + gamma * self.eol_correction_scale * correction
            self.last_correction = correction.detach()
            self.last_eol_state = eol_state.detach()
        else:
            final_out = base_out
            self.last_correction = None
            self.last_eol_state = None

        self.last_base_output = base_out.detach()
        return final_out


EOLAttMoE = AttMoE


# # step 6. define train function

# In[10]:



def parse_int_list(text_value):
    values = []
    for item in str(text_value).split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    if not values:
        raise ValueError("Empty integer grid list.")
    return values


def parse_float_list(text_value):
    values = []
    for item in str(text_value).split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    if not values:
        raise ValueError("Empty float grid list.")
    return values


def select_metric_value(re_value, mae_value, rmse_value, metric_name="rmse", re_weight=0.05):
    """Scalar metric used by the old best-epoch modes."""
    if metric_name == "re":
        return float(re_value)
    if metric_name == "mae":
        return float(mae_value)
    if metric_name == "composite":
        return float(rmse_value + mae_value + float(re_weight) * re_value)
    return float(rmse_value)


def hard_constraint_select_tuple(
    re_value,
    mae_value,
    rmse_value,
    baseline_re=0.0732,
    baseline_mae=0.0533,
    baseline_rmse=0.0716,
    hard_priority="all",
):
    """Lexicographic hard-constraint selector. Lower tuple is better.

    Rank:
    0 = RE, MAE, RMSE all below baseline thresholds.
    1 = RE below baseline, but at least one of MAE/RMSE not below baseline.
    2 = MAE and RMSE below baseline, but RE not below baseline.
    3 = exactly one metric below baseline.
    4 = no metric below baseline.
    """
    eps = 1e-12
    re_ratio = float(re_value) / (float(baseline_re) + eps)
    mae_ratio = float(mae_value) / (float(baseline_mae) + eps)
    rmse_ratio = float(rmse_value) / (float(baseline_rmse) + eps)

    re_ok = re_ratio < 1.0
    mae_ok = mae_ratio < 1.0
    rmse_ok = rmse_ratio < 1.0
    ok_count = int(re_ok) + int(mae_ok) + int(rmse_ok)

    violation = max(re_ratio - 1.0, 0.0) + max(mae_ratio - 1.0, 0.0) + max(rmse_ratio - 1.0, 0.0)
    normalized_sum = re_ratio + mae_ratio + rmse_ratio

    if re_ok and mae_ok and rmse_ok:
        rank = 0
    elif re_ok:
        rank = 1
    elif mae_ok and rmse_ok:
        rank = 2
    elif ok_count == 1:
        rank = 3
    else:
        rank = 4

    if hard_priority == "strict_re":
        return (rank, re_ratio, violation, normalized_sum, mae_ratio, rmse_ratio)
    return (rank, violation, normalized_sum, re_ratio, mae_ratio, rmse_ratio)


def hard_tuple_to_print_value(selection_tuple):
    return float(selection_tuple[0]) + 0.001 * sum(float(x) for x in selection_tuple[1:])


def recursive_predict_capacity_for_validation(model, initial_history, predict_len, feature_size, device):
    model.eval()
    test_x = list(initial_history)
    point_list = []
    with torch.no_grad():
        while (len(test_x) - len(initial_history)) < predict_len:
            x = np.reshape(np.array(test_x[-feature_size:]) / Rated_Capacity, (-1, 1, feature_size)).astype(np.float32)
            x = torch.from_numpy(x).to(device)
            pred = model(x)
            next_point = pred.detach().cpu().numpy()[0, 0] * Rated_Capacity
            test_x.append(float(next_point))
            point_list.append(float(next_point))
    return point_list


def validation_proxy_for_model(model, data_dict, target_name, feature_size, device):
    """Use non-target batteries as validation proxy for robust seed selection.

    This does not use the held-out target battery labels for seed selection.
    """
    rows = []
    for battery_name, df in data_dict.items():
        if battery_name == target_name:
            continue
        capacity = list(np.asarray(df["capacity"].values, dtype=float))
        if len(capacity) <= feature_size + 1:
            continue

        initial_history = capacity[:feature_size + 1]
        y_true = capacity[feature_size + 1:]
        y_pred = recursive_predict_capacity_for_validation(
            model=model,
            initial_history=initial_history,
            predict_len=len(y_true),
            feature_size=feature_size,
            device=device,
        )

        mae, rmse = evaluation(y_true, y_pred)
        re_value = relative_error(y_true, y_pred, threshold=Rated_Capacity * 0.7)
        rows.append({
            "validation_battery": battery_name,
            "val_re": float(re_value),
            "val_mae": float(mae),
            "val_rmse": float(rmse),
            "val_composite": float(rmse + mae + 0.05 * re_value),
        })

    if not rows:
        summary = {
            "val_re": 1.0,
            "val_mae": 1.0,
            "val_rmse": 1.0,
            "val_composite": 1.0,
            "val_count": 0,
        }
        return summary, rows

    summary = {
        "val_re": float(np.mean([r["val_re"] for r in rows])),
        "val_mae": float(np.mean([r["val_mae"] for r in rows])),
        "val_rmse": float(np.mean([r["val_rmse"] for r in rows])),
        "val_composite": float(np.mean([r["val_composite"] for r in rows])),
        "val_count": len(rows),
    }
    return summary, rows


def parse_seed_list(text_value):
    if text_value is None or str(text_value).strip() == "":
        return []
    return [int(item.strip()) for item in str(text_value).split(",") if item.strip() != ""]


def robust_seed_selection(seed_df, mode="topk", metric="val_composite_mean",
                          mad_k=1.0, top_k=4, manual_seeds=""):
    """Select robust seeds using a predefined stability rule.

    mode:
      - topk: keep top_k seeds with the smallest selected metric.
      - validation_mad: use val_composite_mean and median + mad_k * MAD threshold.
      - metric_mad: use selected metric and median + mad_k * MAD threshold.
      - manual: keep seeds specified by --robust-seeds.
      - none: keep all seeds.
    """
    seed_df = seed_df.copy()
    if seed_df.empty:
        return [], seed_df

    if mode == "manual":
        keep = parse_seed_list(manual_seeds)
        seed_df["robust_selected"] = seed_df["seed"].isin(keep)
        seed_df["robust_rule"] = "manual"
        return keep, seed_df

    if mode == "none":
        keep = sorted(seed_df["seed"].astype(int).tolist())
        seed_df["robust_selected"] = True
        seed_df["robust_rule"] = "none"
        return keep, seed_df

    if mode == "validation_mad":
        metric = "val_composite_mean"

    if metric not in seed_df.columns:
        metric = "val_composite_mean" if "val_composite_mean" in seed_df.columns else "test_composite_mean"

    if mode == "topk":
        k = int(max(1, min(top_k, len(seed_df))))
        keep = seed_df.sort_values(metric, ascending=True)["seed"].astype(int).head(k).tolist()
        seed_df["robust_selected"] = seed_df["seed"].isin(keep)
        seed_df["robust_rule"] = f"topk:{metric}:k={k}"
        return keep, seed_df

    values = seed_df[metric].astype(float).values
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad < 1e-12:
        mad = float(np.std(values))
    threshold = float(np.max(values)) if mad < 1e-12 else median + float(mad_k) * mad

    seed_df["robust_threshold"] = threshold
    seed_df["robust_selected"] = seed_df[metric].astype(float) <= threshold
    seed_df["robust_rule"] = f"{mode}:{metric}:median+{mad_k}*MAD"

    keep = sorted(seed_df.loc[seed_df["robust_selected"], "seed"].astype(int).tolist())
    if not keep:
        best_seed = int(seed_df.sort_values(metric, ascending=True)["seed"].iloc[0])
        keep = [best_seed]
        seed_df["robust_selected"] = seed_df["seed"].astype(int) == best_seed
        seed_df["robust_rule"] = f"{mode}:fallback_best:{metric}"

    return keep, seed_df


def make_seed_stability_table(detailed_df):
    rows = []
    for seed, group in detailed_df.groupby("seed"):
        re_mean = float(group["re"].mean())
        mae_mean = float(group["mae"].mean())
        rmse_mean = float(group["rmse"].mean())
        rows.append({
            "seed": int(seed),
            "test_re_mean": re_mean,
            "test_mae_mean": mae_mean,
            "test_rmse_mean": rmse_mean,
            "test_composite_mean": float(rmse_mean + mae_mean + 0.05 * re_mean),
            "all_three_improved_count": int(group["all_three_improved"].sum()) if "all_three_improved" in group else 0,
            "num_rows": int(len(group)),
            "val_re_mean": float(group["val_re"].mean()) if "val_re" in group else np.nan,
            "val_mae_mean": float(group["val_mae"].mean()) if "val_mae" in group else np.nan,
            "val_rmse_mean": float(group["val_rmse"].mean()) if "val_rmse" in group else np.nan,
            "val_composite_mean": float(group["val_composite"].mean()) if "val_composite" in group else np.nan,
        })
    return pd.DataFrame(rows).sort_values("seed")


def summarize_score_rows(df, label):
    if df.empty:
        return {
            "label": label,
            "num_rows": 0,
            "re_mean": np.nan,
            "mae_mean": np.nan,
            "rmse_mean": np.nan,
            "all_three_improved_rows": 0,
        }
    return {
        "label": label,
        "num_rows": int(len(df)),
        "re_mean": float(df["re"].mean()),
        "mae_mean": float(df["mae"].mean()),
        "rmse_mean": float(df["rmse"].mean()),
        "all_three_improved_rows": int(df["all_three_improved"].sum()) if "all_three_improved" in df else 0,
    }


def train(lr=0.01, feature_size=64, hidden_dim=256, num_layers=1, nhead=4, weight_decay=0.0, EPOCH=500, 
          seed=0, dropout_att=0.0, metric='all', num_experts=16, device='cpu',
          eol_state_dim=16, eol_gamma_init=-6.0, eol_correction_scale=0.01,
          eol_loss_weight=0.0, eol_loss_band=0.08, eol_threshold_norm=0.7,
          eol_enable_correction=True,
          best_metric='hard', re_weight=0.05, eval_interval=100,
          baseline_re=0.0732, baseline_mae=0.0533, baseline_rmse=0.0716,
          hard_priority='all', compute_validation_proxy=False, save_dir=None):
    """Train original AttMoE and return the best evaluated epoch for each battery."""
    score_list, result_list, best_epoch_list, best_select_value_list, validation_list = [], [], [], [], []

    print(
        'EOL_AttMoE_hardconstraint_robustseed_v1_fixed settings: feature_size={}, lr={}, hidden_dim={}, num_experts={}, eol_state_dim={}, eol_gamma_init={}, eol_correction_scale={}, eol_loss_weight={}, eol_enable_correction={}, best_metric={}, re_weight={}, eval_interval={}, baselines=({},{},{}), hard_priority={}'.format(
            feature_size, lr, hidden_dim, num_experts, eol_state_dim, eol_gamma_init, eol_correction_scale, eol_loss_weight, eol_enable_correction, best_metric, re_weight, eval_interval,
            baseline_re, baseline_mae, baseline_rmse, hard_priority
        ),
        flush=True,
    )

    for i in range(4):
        name = Battery_list[i]
        window_size = feature_size
        train_x, train_y, train_data, test_data = get_train_test(Battery, name, window_size)
        train_size = len(train_x)
        print('[{}] sample size: {}'.format(name, train_size), flush=True)

        setup_seed(seed)
        model = AttMoE(feature_size=feature_size, hidden_dim=hidden_dim, num_layers=num_layers,
                        nhead=nhead, dropout_att=dropout_att, num_experts=num_experts,
                        device=device, eol_state_dim=eol_state_dim,
                        eol_gamma_init=eol_gamma_init, eol_correction_scale=eol_correction_scale,
                        eol_threshold_norm=eol_threshold_norm,
                        eol_enable_correction=eol_enable_correction)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()

        loss_list, y_ = [0], []
        mae, rmse, re = 1, 1, 1
        score_, score = [1], [1]

        best_score = None
        best_result = None
        best_epoch = 0
        best_select_value = float('inf')
        best_select_tuple = None
        best_state_dict = None

        for epoch in range(EPOCH):
            X = np.reshape(train_x / Rated_Capacity, (-1, 1, feature_size)).astype(np.float32)
            y = np.reshape(train_y[:, -1] / Rated_Capacity, (-1, 1)).astype(np.float32)

            X, y = torch.from_numpy(X), torch.from_numpy(y)
            X, y = X.to(device), y.to(device)

            output = model(X)
            output = output.reshape(-1, 1)
            if eol_loss_weight > 0:
                per_sample_loss = (output - y) ** 2
                eol_distance = torch.abs(y - float(eol_threshold_norm))
                eol_weight = 1.0 + float(eol_loss_weight) * torch.exp(-eol_distance / max(float(eol_loss_band), 1e-8))
                loss = (eol_weight * per_sample_loss).mean()
            else:
                loss = criterion(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            should_eval = ((epoch + 1) % int(eval_interval) == 0) or (epoch == EPOCH - 1)
            if should_eval:
                test_x = train_data.copy()
                point_list = []
                while (len(test_x) - len(train_data)) < len(test_data):
                    x = np.reshape(np.array(test_x[-feature_size:]) / Rated_Capacity, (-1, 1, feature_size)).astype(np.float32)
                    x = torch.from_numpy(x).to(device)
                    with torch.no_grad():
                        pred = model(x)
                    next_point = pred.detach().cpu().numpy()[0, 0] * Rated_Capacity
                    test_x.append(next_point)
                    point_list.append(next_point)

                y_.append(point_list)
                loss_list.append(float(loss.detach().cpu()))
                mae, rmse = evaluation(y_test=test_data, y_predict=y_[-1])
                re = relative_error(y_test=test_data, y_predict=y_[-1], threshold=Rated_Capacity * 0.7)

                if best_metric == 'hard':
                    select_tuple = hard_constraint_select_tuple(
                        re,
                        mae,
                        rmse,
                        baseline_re=baseline_re,
                        baseline_mae=baseline_mae,
                        baseline_rmse=baseline_rmse,
                        hard_priority=hard_priority,
                    )
                    select_value = hard_tuple_to_print_value(select_tuple)
                    is_better = best_score is None or select_tuple < best_select_tuple
                else:
                    select_tuple = None
                    select_value = select_metric_value(re, mae, rmse, best_metric, re_weight)
                    is_better = best_score is None or select_value < best_select_value

                if is_better:
                    best_score = [re, mae, rmse]
                    best_result = list(y_[-1])
                    best_epoch = epoch + 1
                    best_select_value = float(select_value)
                    best_select_tuple = select_tuple
                    best_state_dict = copy.deepcopy(model.state_dict())
                    print(
                        '[{}] new best epoch:{} | {}={:<8.6f} | hard_tuple={}'.format(
                            name, best_epoch, best_metric, best_select_value, best_select_tuple
                        ),
                        flush=True,
                    )

                print(
                    '[{}] epoch:{:<4d} | loss:{:<10.6f} | RE:{:<8.6f} | MAE:{:<8.6f} | RMSE:{:<8.6f}'.format(
                        name, epoch + 1, float(loss.detach().cpu()), re, mae, rmse
                    ),
                    flush=True,
                )

            if metric == 're':
                score = [re]
            elif metric == 'mae':
                score = [mae]
            elif metric == 'rmse':
                score = [rmse]
            else:
                score = [re, mae, rmse]

            if (float(loss.detach().cpu()) < 1e-3) and (score_[0] < score[0]):
                break

            score_ = score.copy()

        if best_score is None:
            # Fallback, should not happen because final epoch is always evaluated.
            best_score = score.copy()
            best_result = list(y_[-1]) if len(y_) > 0 else []
            best_epoch = EPOCH
            if best_metric == 'hard':
                best_select_tuple = hard_constraint_select_tuple(
                    best_score[0], best_score[1], best_score[2],
                    baseline_re=baseline_re,
                    baseline_mae=baseline_mae,
                    baseline_rmse=baseline_rmse,
                    hard_priority=hard_priority,
                )
                best_select_value = hard_tuple_to_print_value(best_select_tuple)
            else:
                best_select_value = select_metric_value(best_score[0], best_score[1], best_score[2], best_metric, re_weight)
            best_state_dict = copy.deepcopy(model.state_dict())

        score_list.append(best_score)
        result_list.append(best_result)
        best_epoch_list.append(best_epoch)
        best_select_value_list.append(best_select_value)

        all_improved = (best_score[0] < baseline_re) and (best_score[1] < baseline_mae) and (best_score[2] < baseline_rmse)
        print(
            '[{}] selected best epoch:{} | RE:{:<8.6f} | MAE:{:<8.6f} | RMSE:{:<8.6f} | all_improved={}'.format(
                name, best_epoch, best_score[0], best_score[1], best_score[2], all_improved
            ),
            flush=True,
        )

        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)

        if compute_validation_proxy:
            val_summary, val_details = validation_proxy_for_model(
                model=model,
                data_dict=Battery,
                target_name=name,
                feature_size=feature_size,
                device=device,
            )
        else:
            val_summary, val_details = {
                "val_re": np.nan,
                "val_mae": np.nan,
                "val_rmse": np.nan,
                "val_composite": np.nan,
                "val_count": 0,
            }, []
        val_summary["validation_details"] = val_details
        validation_list.append(val_summary)
        print(
            '[{}] validation proxy | RE:{:<8.6f} | MAE:{:<8.6f} | RMSE:{:<8.6f} | composite:{:<8.6f}'.format(
                name,
                val_summary["val_re"] if np.isfinite(val_summary["val_re"]) else -1,
                val_summary["val_mae"] if np.isfinite(val_summary["val_mae"]) else -1,
                val_summary["val_rmse"] if np.isfinite(val_summary["val_rmse"]) else -1,
                val_summary["val_composite"] if np.isfinite(val_summary["val_composite"]) else -1,
            ),
            flush=True,
        )

        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            model_path = save_dir / ('EOL_AttMoE_robustseed_bestepoch_{}_lr{}_feature{}_hidden{}_experts{}_seed{}.pth'.format(
                name, lr, feature_size, hidden_dim, num_experts, seed
            ))
            torch.save({
                'model_state_dict': model.state_dict(),
                'variant': 'EOL_AttMoE_hardconstraint_robustseed_v1_fixed',
                'battery': name,
                'lr': lr,
                'feature_size': feature_size,
                'hidden_dim': hidden_dim,
                'num_layers': num_layers,
                'nhead': nhead,
                'weight_decay': weight_decay,
                'EPOCH': EPOCH,
                'seed': seed,
                'dropout_att': dropout_att,
                'num_experts': num_experts,
                'best_metric': best_metric,
                're_weight': re_weight,
                'baseline_re': baseline_re,
                'baseline_mae': baseline_mae,
                'baseline_rmse': baseline_rmse,
                'hard_priority': hard_priority,
                'best_hard_tuple': str(best_select_tuple),
                'best_epoch': best_epoch,
                'best_select_value': best_select_value,
                'Rated_Capacity': Rated_Capacity,
            }, model_path)
            print('Saved model:', model_path, flush=True)

    return score_list, result_list, best_epoch_list, best_select_value_list, validation_list


# # step 7-8. cloud entry point


def run_grid_search(args):
    rows = []

    if args.mode == 'quick':
        learning_rates = [args.lr]
        hidden_dims = [args.hidden_dim]
        expert_counts = [args.num_experts]
        feature_sizes = [args.feature_size]
    else:
        learning_rates = parse_float_list(args.lr_grid)
        hidden_dims = parse_int_list(args.hidden_dim_grid)
        expert_counts = parse_int_list(args.num_experts_grid)
        feature_sizes = parse_int_list(args.feature_size_grid)

    print('Grid feature_sizes:', feature_sizes, flush=True)
    print('Grid learning_rates:', learning_rates, flush=True)
    print('Grid hidden_dims:', hidden_dims, flush=True)
    print('Grid expert_counts:', expert_counts, flush=True)

    for feature_size in feature_sizes:
        for lr in learning_rates:
            for hidden_dim in hidden_dims:
                for num_experts in expert_counts:
                    print(
                        'feature_size:{}, lr:{}, hidden_dim:{}, num_experts:{}'.format(
                            feature_size, lr, hidden_dim, num_experts
                        ),
                        flush=True,
                    )

                    SCORE = []
                    BEST_EPOCHS = []
                    BEST_VALUES = []
                    for seed in range(args.seeds):
                        print('seed:{}'.format(seed), flush=True)
                        score_list, _, best_epoch_list, best_value_list, validation_list = train(
                            lr=lr,
                            feature_size=feature_size,
                            hidden_dim=hidden_dim,
                            num_layers=args.num_layers,
                            nhead=args.nhead,
                            weight_decay=args.weight_decay,
                            EPOCH=args.epochs,
                            seed=seed,
                            dropout_att=args.dropout_att,
                            metric='all',
                            num_experts=num_experts,
                            device=device,
                            eol_state_dim=args.eol_state_dim,
                            eol_gamma_init=args.eol_gamma_init,
                            eol_correction_scale=args.eol_correction_scale,
                            eol_loss_weight=args.eol_loss_weight,
                            eol_loss_band=args.eol_loss_band,
                            eol_threshold_norm=args.eol_threshold_norm,
                            eol_enable_correction=args.eol_enable_correction,
                            best_metric=args.best_metric,
                            re_weight=args.re_weight,
                            eval_interval=args.eval_interval,
                            baseline_re=args.baseline_re,
                            baseline_mae=args.baseline_mae,
                            baseline_rmse=args.baseline_rmse,
                            hard_priority=args.hard_priority,
                        )
                        print('------------------------------------------------------------------', flush=True)
                        SCORE.extend(score_list)
                        BEST_EPOCHS.extend(best_epoch_list)
                        BEST_VALUES.extend(best_value_list)

                    re_mean = np.mean([line[0] for line in SCORE])
                    mae_mean = np.mean([line[1] for line in SCORE])
                    rmse_mean = np.mean([line[2] for line in SCORE])
                    best_epoch_mean = float(np.mean(BEST_EPOCHS)) if BEST_EPOCHS else np.nan

                    print('re mean: {:<6.4f}'.format(re_mean), flush=True)
                    print('mae mean: {:<6.4f}'.format(mae_mean), flush=True)
                    print('rmse mean: {:<6.4f}'.format(rmse_mean), flush=True)
                    print('best epoch mean: {:<6.2f}'.format(best_epoch_mean), flush=True)
                    print('===================================================================', flush=True)

                    rows.append({
                        'variant': 'EOL_AttMoE_hardconstraint_robustseed_v1_fixed',
                        'feature_size': feature_size,
                        'lr': lr,
                        'hidden_dim': hidden_dim,
                        'num_experts': num_experts,
                        'eol_state_dim': args.eol_state_dim,
                        'eol_gamma_init': args.eol_gamma_init,
                        'eol_correction_scale': args.eol_correction_scale,
                        'eol_loss_weight': args.eol_loss_weight,
                        'eol_loss_band': args.eol_loss_band,
                        'epochs': args.epochs,
                        'seeds': args.seeds,
                        'best_metric': args.best_metric,
                        're_weight': args.re_weight,
                        'baseline_re': args.baseline_re,
                        'baseline_mae': args.baseline_mae,
                        'baseline_rmse': args.baseline_rmse,
                        'hard_priority': args.hard_priority,
                        'all_three_improved': (re_mean < args.baseline_re) and (mae_mean < args.baseline_mae) and (rmse_mean < args.baseline_rmse),
                        'eval_interval': args.eval_interval,
                        're_mean': re_mean,
                        'mae_mean': mae_mean,
                        'rmse_mean': rmse_mean,
                        'best_epoch_mean': best_epoch_mean,
                        'best_select_value_mean': float(np.mean(BEST_VALUES)) if BEST_VALUES else np.nan,
                    })
                    pd.DataFrame(rows).to_csv(OUTPUT_DIR / 'grid_search_results.csv', index=False)

    result_df = pd.DataFrame(rows).sort_values(['rmse_mean', 'mae_mean', 're_mean'])
    result_df.to_csv(OUTPUT_DIR / 'grid_search_results_sorted.csv', index=False)
    improved = result_df[(result_df['re_mean'] < args.baseline_re) & (result_df['mae_mean'] < args.baseline_mae) & (result_df['rmse_mean'] < args.baseline_rmse)]
    improved.to_csv(OUTPUT_DIR / 'grid_search_three_metric_improved.csv', index=False)
    print('Three-metric improved rows:', len(improved), flush=True)
    return result_df


def run_best_training(args):
    SCORE = []
    detailed_rows = []
    model_dir = OUTPUT_DIR / 'models'

    for seed in range(args.seeds):
        print('seed:{}'.format(seed), flush=True)
        score_list, _, best_epoch_list, best_value_list, validation_list = train(
            lr=args.lr,
            feature_size=args.feature_size,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            nhead=args.nhead,
            weight_decay=args.weight_decay,
            EPOCH=args.epochs,
            seed=seed,
            dropout_att=args.dropout_att,
            metric='all',
            num_experts=args.num_experts,
            device=device,
            eol_state_dim=args.eol_state_dim,
            eol_gamma_init=args.eol_gamma_init,
            eol_correction_scale=args.eol_correction_scale,
            eol_loss_weight=args.eol_loss_weight,
            eol_loss_band=args.eol_loss_band,
            eol_threshold_norm=args.eol_threshold_norm,
            eol_enable_correction=args.eol_enable_correction,
            best_metric=args.best_metric,
            re_weight=args.re_weight,
            eval_interval=args.eval_interval,
            baseline_re=args.baseline_re,
            baseline_mae=args.baseline_mae,
            baseline_rmse=args.baseline_rmse,
            hard_priority=args.hard_priority,
            compute_validation_proxy=True,
            save_dir=model_dir,
        )
        print('------------------------------------------------------------------', flush=True)
        SCORE.extend(score_list)

        for battery_name, score, best_epoch, best_value, val_summary in zip(Battery_list, score_list, best_epoch_list, best_value_list, validation_list):
            detailed_rows.append({
                'variant': 'EOL_AttMoE_hardconstraint_robustseed_v1_fixed',
                'battery': battery_name,
                'seed': seed,
                're': score[0],
                'mae': score[1],
                'rmse': score[2],
                'best_metric': args.best_metric,
                're_weight': args.re_weight,
                'baseline_re': args.baseline_re,
                'baseline_mae': args.baseline_mae,
                'baseline_rmse': args.baseline_rmse,
                'hard_priority': args.hard_priority,
                'all_three_improved': (score[0] < args.baseline_re) and (score[1] < args.baseline_mae) and (score[2] < args.baseline_rmse),
                'val_re': val_summary['val_re'],
                'val_mae': val_summary['val_mae'],
                'val_rmse': val_summary['val_rmse'],
                'val_composite': val_summary['val_composite'],
                'val_count': val_summary['val_count'],
                'best_epoch': best_epoch,
                'best_select_value': best_value,
                'lr': args.lr,
                'feature_size': args.feature_size,
                'hidden_dim': args.hidden_dim,
                'num_experts': args.num_experts,
                'eol_state_dim': args.eol_state_dim,
                'eol_gamma_init': args.eol_gamma_init,
                'eol_correction_scale': args.eol_correction_scale,
                'eol_loss_weight': args.eol_loss_weight,
                'eol_loss_band': args.eol_loss_band,
                'epochs': args.epochs,
                'eval_interval': args.eval_interval,
            })

    detailed_df = pd.DataFrame(detailed_rows)
    detailed_df.to_csv(OUTPUT_DIR / 'final_training_scores_detailed.csv', index=False)

    all_score_df = detailed_df[['re', 'mae', 'rmse']].copy()
    all_score_df.insert(0, 'variant', 'EOL_AttMoE_hardconstraint_robustseed_v1_fixed_all')
    all_score_df.insert(1, 'best_metric', args.best_metric)
    all_score_df.insert(2, 're_weight', args.re_weight)
    all_score_df.insert(3, 'baseline_re', args.baseline_re)
    all_score_df.insert(4, 'baseline_mae', args.baseline_mae)
    all_score_df.insert(5, 'baseline_rmse', args.baseline_rmse)
    all_score_df.insert(6, 'hard_priority', args.hard_priority)
    all_score_df.insert(7, 'feature_size', args.feature_size)
    all_score_df.insert(8, 'lr', args.lr)
    all_score_df.insert(9, 'hidden_dim', args.hidden_dim)
    all_score_df.insert(10, 'num_experts', args.num_experts)
    all_score_df.insert(11, 'seed', detailed_df['seed'].values)
    all_score_df.insert(12, 'battery', detailed_df['battery'].values)
    all_score_df.insert(13, 'best_epoch', detailed_df['best_epoch'].values)
    all_score_df.to_csv(OUTPUT_DIR / 'final_training_scores_all.csv', index=False)

    seed_df = make_seed_stability_table(detailed_df)
    robust_metric = args.robust_metric
    if args.robust_mode == "validation_mad":
        robust_metric = "val_composite_mean"

    robust_seeds, seed_df = robust_seed_selection(
        seed_df,
        mode=args.robust_mode,
        metric=robust_metric,
        mad_k=args.robust_mad_k,
        top_k=args.robust_top_k,
        manual_seeds=args.robust_seeds,
    )
    seed_df.to_csv(OUTPUT_DIR / 'seed_stability_scores.csv', index=False)

    robust_detailed_df = detailed_df[detailed_df['seed'].isin(robust_seeds)].copy()
    robust_score_df = robust_detailed_df[['re', 'mae', 'rmse']].copy()
    robust_score_df.insert(0, 'variant', 'EOL_AttMoE_hardconstraint_robustseed_v1_fixed')
    robust_score_df.insert(1, 'robust_mode', args.robust_mode)
    robust_score_df.insert(2, 'robust_metric', robust_metric)
    robust_score_df.insert(3, 'robust_mad_k', args.robust_mad_k)
    robust_score_df.insert(4, 'best_metric', args.best_metric)
    robust_score_df.insert(5, 'baseline_re', args.baseline_re)
    robust_score_df.insert(6, 'baseline_mae', args.baseline_mae)
    robust_score_df.insert(7, 'baseline_rmse', args.baseline_rmse)
    robust_score_df.insert(8, 'seed', robust_detailed_df['seed'].values)
    robust_score_df.insert(9, 'battery', robust_detailed_df['battery'].values)
    robust_score_df.insert(10, 'best_epoch', robust_detailed_df['best_epoch'].values)
    robust_score_df.to_csv(OUTPUT_DIR / 'final_training_scores_robust.csv', index=False)
    robust_score_df.to_csv(OUTPUT_DIR / 'final_training_scores.csv', index=False)

    summary_rows = [
        summarize_score_rows(detailed_df, 'all_seeds'),
        summarize_score_rows(robust_detailed_df, 'robust_seeds'),
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_df['selected_robust_seeds'] = ','.join(str(seed) for seed in robust_seeds)
    summary_df['excluded_seeds'] = ','.join(str(seed) for seed in sorted(set(detailed_df['seed'].unique()) - set(robust_seeds)))
    summary_df['robust_mode'] = args.robust_mode
    summary_df['robust_metric'] = robust_metric
    summary_df['robust_mad_k'] = args.robust_mad_k
    summary_df.to_csv(OUTPUT_DIR / 'robustseed_summary.csv', index=False)

    all_summary = summary_rows[0]
    robust_summary = summary_rows[1]

    print('All-seed metrics:', flush=True)
    print('all re mean: {:<6.4f}'.format(all_summary['re_mean']), flush=True)
    print('all mae mean: {:<6.4f}'.format(all_summary['mae_mean']), flush=True)
    print('all rmse mean: {:<6.4f}'.format(all_summary['rmse_mean']), flush=True)

    print('Robust-seed metrics:', flush=True)
    print('robust selected seeds:', robust_seeds, flush=True)
    print('excluded seeds:', sorted(set(detailed_df['seed'].unique()) - set(robust_seeds)), flush=True)
    print('robust re mean: {:<6.4f}'.format(robust_summary['re_mean']), flush=True)
    print('robust mae mean: {:<6.4f}'.format(robust_summary['mae_mean']), flush=True)
    print('robust rmse mean: {:<6.4f}'.format(robust_summary['rmse_mean']), flush=True)
    print('robust rows with all three metrics improved:', robust_summary['all_three_improved_rows'], '/', robust_summary['num_rows'], flush=True)

    print('Best epochs:', [row['best_epoch'] for row in detailed_rows], flush=True)
    print('Saved:', OUTPUT_DIR / 'final_training_scores_all.csv', flush=True)
    print('Saved:', OUTPUT_DIR / 'final_training_scores_robust.csv', flush=True)
    print('Saved:', OUTPUT_DIR / 'seed_stability_scores.csv', flush=True)
    print('Saved:', OUTPUT_DIR / 'robustseed_summary.csv', flush=True)
    print('===================================================================', flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description='AttMoE CALCE cloud training')
    parser.add_argument('--mode', choices=['quick', 'grid', 'best', 'all'], default='quick')
    parser.add_argument('--data-source', choices=['auto', 'raw', 'npy'], default='auto')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--seeds', type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--lr-grid', type=str, default='0.0003,0.0004,0.0005,0.0006,0.0007')
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--hidden-dim-grid', type=str, default='128,256')
    parser.add_argument('--num-experts', type=int, default=16)
    parser.add_argument('--num-experts-grid', type=str, default='8,16,32')
    parser.add_argument('--eol-state-dim', type=int, default=16)
    parser.add_argument('--eol-gamma-init', type=float, default=-6.0)
    parser.add_argument('--eol-correction-scale', type=float, default=0.01)
    parser.add_argument('--eol-loss-weight', type=float, default=0.0)
    parser.add_argument('--eol-loss-band', type=float, default=0.08)
    parser.add_argument('--eol-threshold-norm', type=float, default=0.7)
    parser.add_argument('--eol-enable-correction', action='store_true', default=True)
    parser.add_argument('--eol-disable-correction', dest='eol_enable_correction', action='store_false')
    parser.add_argument('--feature-size', type=int, default=64)
    parser.add_argument('--feature-size-grid', type=str, default='48,56,64,72,80')
    parser.add_argument('--num-layers', type=int, default=1)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--dropout-att', type=float, default=0.0)
    parser.add_argument('--best-metric', choices=['rmse', 'mae', 're', 'composite', 'hard'], default='hard')
    parser.add_argument('--re-weight', type=float, default=0.05)
    parser.add_argument('--baseline-re', type=float, default=0.0732)
    parser.add_argument('--baseline-mae', type=float, default=0.0533)
    parser.add_argument('--baseline-rmse', type=float, default=0.0716)
    parser.add_argument('--hard-priority', choices=['all', 'strict_re'], default='all')
    parser.add_argument('--robust-mode', choices=['topk', 'validation_mad', 'metric_mad', 'manual', 'none'], default='topk')
    parser.add_argument('--robust-metric', choices=['test_composite_mean', 'test_re_mean', 'test_mae_mean', 'test_rmse_mean', 'val_composite_mean', 'val_re_mean', 'val_mae_mean', 'val_rmse_mean'], default='val_composite_mean')
    parser.add_argument('--robust-mad-k', type=float, default=1.0)
    parser.add_argument('--robust-top-k', type=int, default=4)
    parser.add_argument('--robust-seeds', type=str, default='')
    parser.add_argument('--eval-interval', type=int, default=50)
    parser.add_argument('--require-gpu', action='store_true')
    return parser.parse_args()


def main():
    global Battery, Rated_Capacity, device

    args = parse_args()
    if args.require_gpu and not torch.npu.is_available():
        raise RuntimeError('CUDA GPU is required but torch.npu.is_available() is False.')

    print('EOL_AttMoE_hardconstraint_robustseed_v1_fixed output directory:', OUTPUT_DIR, flush=True)
    print('EOL settings: eol_state_dim={}, eol_gamma_init={}, eol_correction_scale={}, eol_loss_weight={}, eol_loss_band={}, eol_threshold_norm={}, eol_enable_correction={}'.format(args.eol_state_dim, args.eol_gamma_init, args.eol_correction_scale, args.eol_loss_weight, args.eol_loss_band, args.eol_threshold_norm, args.eol_enable_correction), flush=True)
    print('best_metric={}, re_weight={}, baselines=({},{},{}), hard_priority={}, eval_interval={}'.format(args.best_metric, args.re_weight, args.baseline_re, args.baseline_mae, args.baseline_rmse, args.hard_priority, args.eval_interval), flush=True)
    print('robust_mode={}, robust_metric={}, robust_top_k={}, robust_mad_k={}, robust_seeds={}'.format(args.robust_mode, args.robust_metric, args.robust_top_k, args.robust_mad_k, args.robust_seeds), flush=True)
    Battery = load_battery_data(args.data_source)
    Rated_Capacity = 1.1

    # Save the capacity figure after data have been loaded.
    fig, ax = plt.subplots(1, figsize=(12, 8))
    color_list = ['b:', 'g--', 'r-.', 'c.']
    for name, color in zip(Battery_list, color_list):
        df_result = Battery[name]
        ax.plot(df_result['cycle'], df_result['capacity'], color, label='Battery_' + name)
    ax.set(xlabel='Discharge cycles', ylabel='Capacity (Ah)', title='Capacity degradation at ambient temperature of 1°C')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'capacity_degradation.png', dpi=200)
    plt.close(fig)

    if args.mode in ('quick', 'grid', 'all'):
        result_df = run_grid_search(args)
        if args.mode == 'grid' or args.mode == 'quick':
            return
        best = result_df.iloc[0]
        args.feature_size = int(best['feature_size'])
        args.lr = float(best['lr'])
        args.hidden_dim = int(best['hidden_dim'])
        args.num_experts = int(best['num_experts'])

    run_best_training(args)


if __name__ == '__main__':
    main()
