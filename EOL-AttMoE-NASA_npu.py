#!/usr/bin/env python
# coding: utf-8

"""
AttMoE-NASA 云服务器运行版

核心的数据处理、Attention、MoE、训练和留一评估逻辑均保留自原 Notebook。
主要云端适配：
1. 删除 Jupyter 专用的 %matplotlib inline。
2. 使用 Agg 后端保存图片。
3. 自动寻找 NASA.npy 或四个 NASA .mat 文件。
4. 增加 quick/grid/best/all 命令行模式。
5. 保存网格搜索结果、最终指标和模型文件。
6. 普通依赖缺失时尝试自动安装；不覆盖云镜像中的 CUDA 版 PyTorch。
"""

import argparse
import copy
import importlib.util
import json
import os
import random
import subprocess
import sys
from datetime import datetime
from math import sqrt
from pathlib import Path


def _ensure_package(import_name, pip_name=None):
    """缺少普通依赖时，安装到当前 Python 环境。"""
    if importlib.util.find_spec(import_name) is not None:
        return

    package_name = pip_name or import_name
    print(
        "Missing dependency: {}. Installing {} ...".format(
            import_name, package_name
        ),
        flush=True,
    )
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-cache-dir",
            package_name,
        ]
    )


# 保留云镜像自带、与 CUDA 匹配的 PyTorch，不在脚本中重新安装 torch。
for _import_name, _pip_name in {
    "numpy": "numpy",
    "scipy": "scipy",
    "matplotlib": "matplotlib",
    "sklearn": "scikit-learn",
}.items():
    _ensure_package(_import_name, _pip_name)

if importlib.util.find_spec("torch") is None:
    raise ImportError(
        "PyTorch is missing. Please select a CUDA-enabled PyTorch image."
    )

import matplotlib

# 云服务器通常无桌面图形界面
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
import torch_npu  # NPU support
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error


BASE_DIR = Path(__file__).resolve().parent
BATTERY_LIST = ["B0005", "B0006", "B0007", "B0018"]

Battery_list = BATTERY_LIST
Battery = None
Rated_Capacity = 2.0


# ============================================================
# step 1. get device
# ============================================================

device = torch.device("npu:0" if torch.npu.is_available() else "cpu")


def print_device_information():
    print("=" * 72, flush=True)
    print("Python:", sys.executable, flush=True)
    print("Working directory:", Path.cwd(), flush=True)
    print("Script directory:", BASE_DIR, flush=True)
    print("PyTorch version:", torch.__version__, flush=True)
    print("NPU version:", getattr(torch, "version", {}).get("npu", "N/A") if hasattr(torch, "version") and isinstance(torch.version, dict) else "NPU", flush=True)
    print("NPU available:", torch.npu.is_available(), flush=True)
    print("Current device:", device, flush=True)

    if torch.npu.is_available():
        print("NPU name:", torch.npu.get_device_name(0), flush=True)

    print("=" * 72, flush=True)


# ============================================================
# step 2. load data from mat / npy
# ============================================================

def convert_to_time(hmm):
    year = int(hmm[0])
    month = int(hmm[1])
    day = int(hmm[2])
    hour = int(hmm[3])
    minute = int(hmm[4])
    second = int(hmm[5])

    return datetime(
        year=year,
        month=month,
        day=day,
        hour=hour,
        minute=minute,
        second=second,
    )


def loadMat(matfile):
    data = scipy.io.loadmat(str(matfile))

    # 原 Notebook 使用 matfile.split(".")[0]。
    # 云端使用完整路径时必须只取文件主名，才能正确读取 MATLAB 变量。
    filename = Path(matfile).stem

    col = data[filename]
    col = col[0][0][0][0]
    size = col.shape[0]

    data = []

    for i in range(size):
        k = list(col[i][3][0].dtype.fields.keys())
        d1, d2 = {}, {}

        if str(col[i][0][0]) != "impedance":
            for j in range(len(k)):
                t = col[i][3][0][0][j][0]
                l = [t[m] for m in range(len(t))]
                d2[k[j]] = l

        d1["type"] = str(col[i][0][0])
        d1["temp"] = int(col[i][1][0])
        d1["time"] = str(convert_to_time(col[i][2][0]))
        d1["data"] = d2
        data.append(d1)

    return data


def getBatteryCapacity(Battery_data):
    cycle, capacity = [], []
    i = 1

    for Bat in Battery_data:
        if Bat["type"] == "discharge":
            capacity.append(Bat["data"]["Capacity"][0])
            cycle.append(i)
            i += 1

    return [cycle, capacity]


def getBatteryValues(Battery_data, Type="charge"):
    data = []

    for Bat in Battery_data:
        if Bat["type"] == Type:
            data.append(Bat["data"])

    return data


def _candidate_nasa_dirs(user_data_dir=None):
    candidates = []

    if user_data_dir:
        candidates.append(Path(user_data_dir).expanduser())

    candidates.extend(
        [
            BASE_DIR / "datasets" / "NASA",
            BASE_DIR.parent / "datasets" / "NASA",
            Path.cwd() / "datasets" / "NASA",
            BASE_DIR,
            Path.cwd(),
        ]
    )

    result = []
    seen = set()

    for candidate in candidates:
        candidate = candidate.resolve()

        if candidate not in seen:
            seen.add(candidate)
            result.append(candidate)

    return result


def _load_raw_mat(data_dir):
    battery = {}

    for name in BATTERY_LIST:
        path = data_dir / (name + ".mat")

        if not path.is_file():
            raise FileNotFoundError("NASA MAT file not found: {}".format(path))

        print("Load Dataset {}.mat ...".format(name), flush=True)
        data = loadMat(path)
        battery[name] = getBatteryCapacity(data)

    return battery


def _load_npy(npy_path):
    print("Use extracted NASA data:", npy_path, flush=True)
    loaded = np.load(str(npy_path), allow_pickle=True)
    battery = loaded.item()

    missing = set(BATTERY_LIST).difference(battery.keys())

    if missing:
        raise KeyError(
            "NASA.npy is missing batteries: {}".format(sorted(missing))
        )

    return battery


def load_battery_data(data_source="auto", data_dir=None):
    candidates = _candidate_nasa_dirs(data_dir)

    if data_source in ("auto", "raw"):
        for candidate in candidates:
            if all(
                (candidate / (name + ".mat")).is_file()
                for name in BATTERY_LIST
            ):
                print("Use raw NASA MAT data:", candidate, flush=True)
                return _load_raw_mat(candidate)

        if data_source == "raw":
            checked = "\n".join("  - {}".format(p) for p in candidates)
            raise FileNotFoundError(
                "The four NASA MAT files were not found. Checked:\n"
                + checked
            )

    if data_source in ("auto", "npy"):
        for candidate in candidates:
            npy_path = candidate / "NASA.npy"

            if npy_path.is_file():
                return _load_npy(npy_path)

        if data_source == "npy":
            checked = "\n".join(
                "  - {}".format(p / "NASA.npy") for p in candidates
            )
            raise FileNotFoundError(
                "NASA.npy was not found. Checked:\n" + checked
            )

    raise FileNotFoundError(
        "Neither the four NASA MAT files nor NASA.npy could be found."
    )


# ============================================================
# step 3. capacity figure
# ============================================================

def save_capacity_figure(battery, output_dir):
    fig, ax = plt.subplots(1, figsize=(12, 8))
    color_list = ["b:", "g--", "r-.", "c."]

    for name, color in zip(BATTERY_LIST, color_list):
        df_result = battery[name]
        ax.plot(df_result[0], df_result[1], color, label=name)

    ax.set(
        xlabel="Discharge cycles",
        ylabel="Capacity (Ah)",
        title="Capacity degradation at ambient temperature of 24°C",
    )
    ax.legend()
    fig.tight_layout()

    figure_path = output_dir / "capacity_degradation_NASA.png"
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)

    print("Capacity figure saved:", figure_path, flush=True)


# ============================================================
# step 4. data processing
# ============================================================

def build_sequences(text, window_size):
    x, y = [], []

    for i in range(len(text) - window_size):
        sequence = text[i : i + window_size]
        target = text[i + 1 : i + 1 + window_size]

        x.append(sequence)
        y.append(target)

    return np.array(x), np.array(y)


def split_dataset(
    data_sequence,
    train_ratio=0.0,
    capacity_threshold=0.0,
):
    if capacity_threshold > 0:
        max_capacity = max(data_sequence)
        capacity = max_capacity * capacity_threshold
        point = [
            i
            for i in range(len(data_sequence))
            if data_sequence[i] < capacity
        ]
    else:
        point = int(train_ratio + 1)

        if 0 < train_ratio <= 1:
            point = int(len(data_sequence) * train_ratio)

    train_data = data_sequence[:point]
    test_data = data_sequence[point:]

    return train_data, test_data


# 留一评估：一组数据为测试集，其他所有数据全部拿来训练
def get_train_test(data_dict, name, window_size=8):
    data_sequence = data_dict[name][1]

    train_data = data_sequence[: window_size + 1]
    test_data = data_sequence[window_size + 1 :]

    train_x, train_y = build_sequences(
        text=train_data,
        window_size=window_size,
    )

    for k, v in data_dict.items():
        if k != name:
            data_x, data_y = build_sequences(
                text=v[1],
                window_size=window_size,
            )
            train_x = np.r_[train_x, data_x]
            train_y = np.r_[train_y, data_y]

    return train_x, train_y, list(train_data), list(test_data)


def relative_error(y_test, y_predict, threshold):
    true_re, pred_re = len(y_test), 0

    for i in range(len(y_test) - 1):
        if y_test[i] <= threshold >= y_test[i + 1]:
            true_re = i - 1
            break

    for i in range(len(y_predict) - 1):
        if y_predict[i] <= threshold:
            pred_re = i - 1
            break

    if true_re == 0:
        return 1.0

    return abs(true_re - pred_re) / true_re


def evaluation(y_test, y_predict):
    mae = mean_absolute_error(y_test, y_predict)
    mse = mean_squared_error(y_test, y_predict)
    rmse = sqrt(mse)

    return mae, rmse


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)

    if torch.npu.is_available():
        torch.npu.manual_seed(seed)
        torch.npu.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.deterministic = True  # NPU不需要


# ============================================================
# step 5. build net
# ============================================================

try:
    from mixture_of_experts import MoE

    MOE_BACKEND = "mixture_of_experts"
except ImportError:
    try:
        _ensure_package("mixture_of_experts", "mixture-of-experts")
        from mixture_of_experts import MoE

        MOE_BACKEND = "mixture_of_experts"
    except Exception:
        MOE_BACKEND = "built_in_dense_fallback"
        print(
            "mixture_of_experts is unavailable; "
            "using a compatible dense MoE fallback.",
            flush=True,
        )

        class MoE(nn.Module):
            def __init__(self, dim, num_experts, experts):
                super(MoE, self).__init__()

                self.gate = nn.Linear(dim, num_experts)
                self.experts = nn.ModuleList(
                    [
                        copy.deepcopy(experts)
                        for _ in range(num_experts)
                    ]
                )

            def forward(self, x):
                weights = torch.softmax(self.gate(x), dim=-1)
                outputs = torch.stack(
                    [expert(x) for expert in self.experts],
                    dim=2,
                )
                out = torch.sum(
                    weights.unsqueeze(-1) * outputs,
                    dim=2,
                )
                aux_loss = torch.zeros(
                    (),
                    dtype=x.dtype,
                    device=x.device,
                )

                return out, aux_loss


class Attention(nn.Module):
    def __init__(
        self,
        feature_size,
        hidden_dim,
        nhead=4,
        dropout=0.0,
    ):
        super(Attention, self).__init__()

        self.query = nn.Linear(feature_size, hidden_dim)
        self.key = nn.Linear(feature_size, hidden_dim)
        self.value = nn.Linear(feature_size, hidden_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

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
    """EOL-AttMoE for NASA.

    Main path:
        capacity window -> original Attention -> original MoE -> Linear -> y_base

    EOL-aware path:
        hidden feature + distance-to-threshold state -> weak residual correction

    Final:
        y = y_base + sigmoid(gamma) * correction_scale * y_correction

    The class name remains AttMoE so the original training code can reuse it directly.
    """
    def __init__(
        self,
        feature_size=16,
        hidden_dim=8,
        num_layers=1,
        nhead=4,
        dropout=0.0,
        dropout_rate=0.2,
        num_experts=8,
        device="cpu",
        eol_state_dim=16,
        eol_gamma_init=-4.0,
        eol_correction_scale=0.02,
        eol_threshold_norm=0.7,
        eol_enable_correction=True,
    ):
        super(AttMoE, self).__init__()

        self.feature_size = feature_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.dropout = nn.Dropout(dropout_rate)

        # Original AttMoE backbone: kept unchanged.
        self.cell = Attention(
            feature_size=feature_size,
            hidden_dim=hidden_dim,
            nhead=nhead,
            dropout=dropout,
        )

        self.linear = nn.Linear(hidden_dim, 1)

        experts = nn.Linear(hidden_dim, hidden_dim)

        self.moe = MoE(
            dim=hidden_dim,
            num_experts=num_experts,
            experts=experts,
        )
        self.moe = self.moe.to(device)

        # EOL-aware weak residual branch.
        self.eol_correction = EOLAwareCorrection(hidden_dim=hidden_dim, eol_state_dim=eol_state_dim)
        self.eol_gamma = nn.Parameter(torch.tensor(float(eol_gamma_init)))
        self.eol_correction_scale = float(eol_correction_scale)
        self.eol_threshold_norm = float(eol_threshold_norm)
        self.eol_enable_correction = bool(eol_enable_correction)

        self.last_base_output = None
        self.last_correction = None
        self.last_eol_state = None

    def forward(self, x):
        # Keep original Notebook forward behavior exactly.
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


# ============================================================
# step 6. hard-constraint best epoch + robust seed selection
# ============================================================

_ensure_package("pandas", "pandas")
import pandas as pd


def select_metric_value(re_value, mae_value, rmse_value, metric_name="rmse", re_weight=0.05):
    """Scalar metric used by non-hard best-epoch modes."""
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
    baseline_re=0.2005,
    baseline_mae=0.0780,
    baseline_rmse=0.0892,
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


def recursive_predict_capacity(model, initial_history, predict_len, feature_size, device):
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
    """Use non-target batteries as a validation proxy for seed stability.

    This does not use the held-out target battery labels for seed selection.
    """
    rows = []

    for battery_name, battery_values in data_dict.items():
        if battery_name == target_name:
            continue

        capacity = list(np.asarray(battery_values[1], dtype=float))
        if len(capacity) <= feature_size + 1:
            continue

        initial_history = capacity[: feature_size + 1]
        y_true = capacity[feature_size + 1 :]
        y_pred = recursive_predict_capacity(
            model=model,
            initial_history=initial_history,
            predict_len=len(y_true),
            feature_size=feature_size,
            device=device,
        )

        mae, rmse = evaluation(y_true, y_pred)
        re_value = relative_error(y_true, y_pred, threshold=Rated_Capacity * 0.7)

        rows.append(
            {
                "validation_battery": battery_name,
                "val_re": float(re_value),
                "val_mae": float(mae),
                "val_rmse": float(rmse),
                "val_composite": float(rmse + mae + 0.05 * re_value),
            }
        )

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


def robust_seed_selection(seed_df, mode="validation_topk", metric="val_composite_mean", mad_k=1.0, top_k=4, manual_seeds=""):
    """Select robust seeds using a predefined stability rule.

    mode:
    - validation_topk/topk: keep top_k seeds with the smallest chosen validation metric.
    - validation_mad: drop seeds above median + mad_k * MAD on val_composite_mean.
    - metric_mad: drop seeds above median + mad_k * MAD on the selected metric.
    - manual: keep seeds listed in --robust-seeds.
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

    if mode in ("validation_topk", "topk"):
        if metric not in seed_df.columns:
            metric = "val_composite_mean"
        k = int(max(1, min(top_k, len(seed_df))))
        keep = seed_df.sort_values(metric, ascending=True)["seed"].astype(int).head(k).tolist()
        seed_df["robust_selected"] = seed_df["seed"].isin(keep)
        seed_df["robust_rule"] = f"{mode}:{metric}:k={k}"
        return keep, seed_df

    if mode == "validation_mad":
        metric = "val_composite_mean"

    if metric not in seed_df.columns:
        metric = "test_composite_mean"

    values = seed_df[metric].astype(float).values
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))

    if mad < 1e-12:
        mad = float(np.std(values))

    if mad < 1e-12:
        threshold = float(np.max(values))
    else:
        threshold = median + float(mad_k) * mad

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

        rows.append(
            {
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
            }
        )

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


def train(
    lr=0.01,
    feature_size=8,
    hidden_dim=32,
    num_layers=1,
    nhead=8,
    weight_decay=0.0,
    EPOCH=1000,
    seed=0,
    dropout=0.0,
    metric="all",
    num_experts=8,
    device="cpu",
    eol_state_dim=16,
    eol_gamma_init=-4.0,
    eol_correction_scale=0.02,
    eol_loss_weight=0.0,
    eol_loss_band=0.08,
    eol_threshold_norm=0.7,
    eol_enable_correction=True,
    best_metric="hard",
    re_weight=0.05,
    eval_interval=50,
    baseline_re=0.2005,
    baseline_mae=0.0780,
    baseline_rmse=0.0892,
    hard_priority="all",
    compute_validation_proxy=False,
    save_dir=None,
):
    score_list, result_list, best_epoch_list, best_select_value_list, validation_list = [], [], [], [], []

    print(
        "NASA EOL_AttMoE hardconstraint robustseed settings: feature_size={}, lr={}, hidden_dim={}, num_experts={}, eol_state_dim={}, eol_gamma_init={}, eol_correction_scale={}, eol_loss_weight={}, eol_enable_correction={}, best_metric={}, eval_interval={}, baselines=({},{},{}), hard_priority={}".format(
            feature_size,
            lr,
            hidden_dim,
            num_experts,
            eol_state_dim,
            eol_gamma_init,
            eol_correction_scale,
            eol_loss_weight,
            eol_enable_correction,
            best_metric,
            eval_interval,
            baseline_re,
            baseline_mae,
            baseline_rmse,
            hard_priority,
        ),
        flush=True,
    )

    for i in range(len(Battery_list)):
        name = Battery_list[i]
        window_size = feature_size

        train_x, train_y, train_data, test_data = get_train_test(
            Battery,
            name,
            window_size,
        )

        setup_seed(seed)

        model = AttMoE(
            feature_size=feature_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            nhead=nhead,
            dropout=dropout,
            num_experts=num_experts,
            device=device,
            eol_state_dim=eol_state_dim,
            eol_gamma_init=eol_gamma_init,
            eol_correction_scale=eol_correction_scale,
            eol_threshold_norm=eol_threshold_norm,
            eol_enable_correction=eol_enable_correction,
        )
        model = model.to(device)

        print(
            "[{}] model device: {}; samples: {}".format(
                name,
                next(model.parameters()).device,
                len(train_x),
            ),
            flush=True,
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        criterion = nn.MSELoss()

        loss_list, y_predictions = [0], []
        best_score = None
        best_result = None
        best_epoch = None
        best_select_value = float("inf")
        best_select_tuple = None
        best_state_dict = None

        previous_score, current_score = [1], [1]

        for epoch in range(EPOCH):
            X = np.reshape(
                train_x / Rated_Capacity,
                (-1, 1, feature_size),
            ).astype(np.float32)

            y = np.reshape(
                train_y[:, -1] / Rated_Capacity,
                (-1, 1),
            ).astype(np.float32)

            X = torch.from_numpy(X).to(device)
            y = torch.from_numpy(y).to(device)

            output = model(X)
            output = output.reshape(-1, 1)
            if eol_loss_weight > 0:
                per_sample_loss = (output - y) ** 2
                eol_distance = torch.abs(y - float(eol_threshold_norm))
                eol_weight = 1.0 + float(eol_loss_weight) * torch.exp(
                    -eol_distance / max(float(eol_loss_band), 1e-8)
                )
                loss = (eol_weight * per_sample_loss).mean()
            else:
                loss = criterion(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            should_evaluate = (
                (epoch + 1) % int(eval_interval) == 0
                or epoch == EPOCH - 1
            )

            if should_evaluate:
                point_list = recursive_predict_capacity(
                    model=model,
                    initial_history=train_data,
                    predict_len=len(test_data),
                    feature_size=feature_size,
                    device=device,
                )

                y_predictions.append(point_list)
                loss_list.append(float(loss.detach().cpu()))

                mae, rmse = evaluation(
                    y_test=test_data,
                    y_predict=y_predictions[-1],
                )
                re_value = relative_error(
                    y_test=test_data,
                    y_predict=y_predictions[-1],
                    threshold=Rated_Capacity * 0.7,
                )

                print(
                    "[{}] epoch={}/{}, loss={:.6f}, RE={:.6f}, MAE={:.6f}, RMSE={:.6f}".format(
                        name,
                        epoch + 1,
                        EPOCH,
                        float(loss.detach().cpu()),
                        re_value,
                        mae,
                        rmse,
                    ),
                    flush=True,
                )

                if best_metric == "hard":
                    select_tuple = hard_constraint_select_tuple(
                        re_value,
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
                    select_value = select_metric_value(re_value, mae, rmse, best_metric, re_weight)
                    is_better = best_score is None or select_value < best_select_value

                if is_better:
                    best_score = [float(re_value), float(mae), float(rmse)]
                    best_result = list(y_predictions[-1])
                    best_epoch = epoch + 1
                    best_select_value = float(select_value)
                    best_select_tuple = select_tuple
                    best_state_dict = copy.deepcopy(model.state_dict())
                    print(
                        "[{}] new best epoch:{} | {}={:<8.6f} | hard_tuple={}".format(
                            name,
                            best_epoch,
                            best_metric,
                            best_select_value,
                            best_select_tuple,
                        ),
                        flush=True,
                    )

                if metric == "re":
                    current_score = [re_value]
                elif metric == "mae":
                    current_score = [mae]
                elif metric == "rmse":
                    current_score = [rmse]
                else:
                    current_score = [re_value, mae, rmse]

                if (
                    float(loss.detach().cpu()) < 1e-3
                    and previous_score[0] < current_score[0]
                ):
                    break

                previous_score = current_score.copy()

        if best_score is None:
            if not y_predictions:
                raise RuntimeError("{} did not produce prediction results.".format(name))
            mae, rmse = evaluation(y_test=test_data, y_predict=y_predictions[-1])
            re_value = relative_error(
                y_test=test_data,
                y_predict=y_predictions[-1],
                threshold=Rated_Capacity * 0.7,
            )
            best_score = [float(re_value), float(mae), float(rmse)]
            best_result = list(y_predictions[-1])
            best_epoch = EPOCH
            if best_metric == "hard":
                best_select_tuple = hard_constraint_select_tuple(
                    best_score[0],
                    best_score[1],
                    best_score[2],
                    baseline_re=baseline_re,
                    baseline_mae=baseline_mae,
                    baseline_rmse=baseline_rmse,
                    hard_priority=hard_priority,
                )
                best_select_value = hard_tuple_to_print_value(best_select_tuple)
            else:
                best_select_value = select_metric_value(best_score[0], best_score[1], best_score[2], best_metric, re_weight)

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

        all_improved = (
            best_score[0] < baseline_re
            and best_score[1] < baseline_mae
            and best_score[2] < baseline_rmse
        )
        print(
            "[{}] selected best epoch:{} | RE:{:<8.6f} | MAE:{:<8.6f} | RMSE:{:<8.6f} | all_improved={}".format(
                name,
                best_epoch,
                best_score[0],
                best_score[1],
                best_score[2],
                all_improved,
            ),
            flush=True,
        )
        print(
            "[{}] validation proxy | RE:{:<8.6f} | MAE:{:<8.6f} | RMSE:{:<8.6f} | composite:{:<8.6f}".format(
                name,
                val_summary["val_re"] if np.isfinite(val_summary["val_re"]) else -1,
                val_summary["val_mae"] if np.isfinite(val_summary["val_mae"]) else -1,
                val_summary["val_rmse"] if np.isfinite(val_summary["val_rmse"]) else -1,
                val_summary["val_composite"] if np.isfinite(val_summary["val_composite"]) else -1,
            ),
            flush=True,
        )

        score_list.append(best_score)
        result_list.append(best_result)
        best_epoch_list.append(best_epoch)
        best_select_value_list.append(best_select_value)

        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

            model_path = save_dir / (
                "AttMoE_NASA_{}_lr{}_hidden{}_experts{}_seed{}_bestepoch{}.pth".format(
                    name,
                    lr,
                    hidden_dim,
                    num_experts,
                    seed,
                    best_epoch,
                )
            )

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "battery": name,
                    "lr": lr,
                    "feature_size": feature_size,
                    "hidden_dim": hidden_dim,
                    "num_layers": num_layers,
                    "nhead": nhead,
                    "weight_decay": weight_decay,
                    "EPOCH": EPOCH,
                    "seed": seed,
                    "dropout": dropout,
                    "num_experts": num_experts,
                    "Rated_Capacity": Rated_Capacity,
                    "moe_backend": MOE_BACKEND,
                    "best_metric": best_metric,
                    "baseline_re": baseline_re,
                    "baseline_mae": baseline_mae,
                    "baseline_rmse": baseline_rmse,
                    "hard_priority": hard_priority,
                    "best_epoch": best_epoch,
                    "best_select_value": best_select_value,
                    "best_hard_tuple": str(best_select_tuple),
                },
                model_path,
            )

            print("Saved model:", model_path, flush=True)

        del model
        del optimizer
        del X
        del y

        if torch.npu.is_available():
            torch.npu.empty_cache()

    return score_list, result_list, best_epoch_list, best_select_value_list, validation_list


# ============================================================
# step 7-8. grid search / best training
# ============================================================

def run_grid_search(args, output_dir):
    rows = []

    learning_rates = [float(x) for x in args.lr_grid.split(",")] if args.lr_grid else [args.lr]
    hidden_dims = [int(x) for x in args.hidden_dim_grid.split(",")] if args.hidden_dim_grid else [args.hidden_dim]
    expert_counts = [int(x) for x in args.num_experts_grid.split(",")] if args.num_experts_grid else [args.num_experts]
    feature_sizes = [int(x) for x in args.feature_size_grid.split(",")] if args.feature_size_grid else [args.feature_size]

    if args.mode == "quick":
        learning_rates = [args.lr]
        hidden_dims = [args.hidden_dim]
        expert_counts = [args.num_experts]
        feature_sizes = [args.feature_size]

    total = len(learning_rates) * len(hidden_dims) * len(expert_counts) * len(feature_sizes)
    current = 0

    for feature_size in feature_sizes:
        for lr in learning_rates:
            for hidden_dim in hidden_dims:
                for num_experts in expert_counts:
                    current += 1
                    print("=" * 72, flush=True)
                    print(
                        "Combination {}/{}: feature_size={}, lr={}, hidden_dim={}, num_experts={}".format(
                            current,
                            total,
                            feature_size,
                            lr,
                            hidden_dim,
                            num_experts,
                        ),
                        flush=True,
                    )

                    SCORE = []

                    for seed in range(args.seeds):
                        print("seed:{}".format(seed), flush=True)
                        score_list, _, _, _, _ = train(
                            lr=lr,
                            feature_size=feature_size,
                            hidden_dim=hidden_dim,
                            num_layers=args.num_layers,
                            nhead=args.nhead,
                            weight_decay=args.weight_decay,
                            EPOCH=args.epochs,
                            seed=seed,
                            dropout=args.dropout,
                            metric="all",
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
                            compute_validation_proxy=False,
                        )
                        SCORE.extend(score_list)

                    re_mean = float(np.mean([line[0] for line in SCORE]))
                    mae_mean = float(np.mean([line[1] for line in SCORE]))
                    rmse_mean = float(np.mean([line[2] for line in SCORE]))

                    rows.append(
                        {
                            "feature_size": feature_size,
                            "lr": lr,
                            "hidden_dim": hidden_dim,
                            "num_experts": num_experts,
                            "eol_state_dim": args.eol_state_dim,
                            "eol_gamma_init": args.eol_gamma_init,
                            "eol_correction_scale": args.eol_correction_scale,
                            "eol_loss_weight": args.eol_loss_weight,
                            "eol_loss_band": args.eol_loss_band,
                            "eol_threshold_norm": args.eol_threshold_norm,
                            "eol_enable_correction": args.eol_enable_correction,
                            "epochs": args.epochs,
                            "seeds": args.seeds,
                            "best_metric": args.best_metric,
                            "eval_interval": args.eval_interval,
                            "baseline_re": args.baseline_re,
                            "baseline_mae": args.baseline_mae,
                            "baseline_rmse": args.baseline_rmse,
                            "re_mean": re_mean,
                            "mae_mean": mae_mean,
                            "rmse_mean": rmse_mean,
                            "all_three_improved": (
                                re_mean < args.baseline_re
                                and mae_mean < args.baseline_mae
                                and rmse_mean < args.baseline_rmse
                            ),
                        }
                    )

                    pd.DataFrame(rows).to_csv(
                        output_dir / "grid_search_results_NASA_progress.csv",
                        index=False,
                        encoding="utf-8-sig",
                    )

                    print("re mean: {:<6.4f}".format(re_mean), flush=True)
                    print("mae mean: {:<6.4f}".format(mae_mean), flush=True)
                    print("rmse mean: {:<6.4f}".format(rmse_mean), flush=True)

    result_df = pd.DataFrame(rows).sort_values(["rmse_mean", "mae_mean", "re_mean"], ascending=True)
    result_df.to_csv(output_dir / "grid_search_results_NASA_sorted.csv", index=False, encoding="utf-8-sig")

    improved = result_df[
        (result_df["re_mean"] < args.baseline_re)
        & (result_df["mae_mean"] < args.baseline_mae)
        & (result_df["rmse_mean"] < args.baseline_rmse)
    ]
    improved.to_csv(output_dir / "grid_search_three_metric_improved_NASA.csv", index=False, encoding="utf-8-sig")
    print("Three-metric improved rows:", len(improved), flush=True)

    if len(result_df) > 0:
        best = result_df.iloc[0]
        best_parameters = {
            "feature_size": int(best["feature_size"]),
            "lr": float(best["lr"]),
            "hidden_dim": int(best["hidden_dim"]),
            "num_experts": int(best["num_experts"]),
            "re_mean": float(best["re_mean"]),
            "mae_mean": float(best["mae_mean"]),
            "rmse_mean": float(best["rmse_mean"]),
        }
        with (output_dir / "best_parameters_NASA.json").open("w", encoding="utf-8") as file:
            json.dump(best_parameters, file, ensure_ascii=False, indent=2)
        print("Best parameters:", best_parameters, flush=True)

    return result_df


def run_best_training(args, output_dir):
    rows = []
    model_dir = output_dir / "models_NASA"
    model_dir.mkdir(parents=True, exist_ok=True)

    for seed in range(args.seeds):
        print(
            "Final training: seed={}, lr={}, hidden_dim={}, num_experts={}, feature_size={}".format(
                seed,
                args.lr,
                args.hidden_dim,
                args.num_experts,
                args.feature_size,
            ),
            flush=True,
        )

        score_list, _, best_epoch_list, best_value_list, validation_list = train(
            lr=args.lr,
            feature_size=args.feature_size,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            nhead=args.nhead,
            weight_decay=args.weight_decay,
            EPOCH=args.epochs,
            seed=seed,
            dropout=args.dropout,
            metric="all",
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

        for battery_name, score, best_epoch, best_value, val_summary in zip(
            BATTERY_LIST,
            score_list,
            best_epoch_list,
            best_value_list,
            validation_list,
        ):
            rows.append(
                {
                    "seed": seed,
                    "battery": battery_name,
                    "variant": "NASA_EOL_AttMoE_hardconstraint_robustseed_v1",
                    "best_metric": args.best_metric,
                    "baseline_re": args.baseline_re,
                    "baseline_mae": args.baseline_mae,
                    "baseline_rmse": args.baseline_rmse,
                    "hard_priority": args.hard_priority,
                    "feature_size": args.feature_size,
                    "lr": args.lr,
                    "hidden_dim": args.hidden_dim,
                    "num_experts": args.num_experts,
                    "eol_state_dim": args.eol_state_dim,
                    "eol_gamma_init": args.eol_gamma_init,
                    "eol_correction_scale": args.eol_correction_scale,
                    "eol_loss_weight": args.eol_loss_weight,
                    "eol_loss_band": args.eol_loss_band,
                    "eol_threshold_norm": args.eol_threshold_norm,
                    "eol_enable_correction": args.eol_enable_correction,
                    "best_epoch": best_epoch,
                    "best_select_value": best_value,
                    "re": score[0],
                    "mae": score[1],
                    "rmse": score[2],
                    "all_three_improved": (
                        score[0] < args.baseline_re
                        and score[1] < args.baseline_mae
                        and score[2] < args.baseline_rmse
                    ),
                    "val_re": val_summary["val_re"],
                    "val_mae": val_summary["val_mae"],
                    "val_rmse": val_summary["val_rmse"],
                    "val_composite": val_summary["val_composite"],
                    "val_count": val_summary["val_count"],
                }
            )

    detailed_df = pd.DataFrame(rows)
    detailed_df.to_csv(output_dir / "final_training_scores_NASA_detailed.csv", index=False, encoding="utf-8-sig")

    all_score_df = detailed_df[["re", "mae", "rmse"]].copy()
    all_score_df.insert(0, "variant", "NASA_EOL_AttMoE_hardconstraint_robustseed_v1_all")
    all_score_df.insert(1, "seed", detailed_df["seed"].values)
    all_score_df.insert(2, "battery", detailed_df["battery"].values)
    all_score_df.insert(3, "best_epoch", detailed_df["best_epoch"].values)
    all_score_df.to_csv(output_dir / "final_training_scores_NASA_all.csv", index=False, encoding="utf-8-sig")

    seed_df = make_seed_stability_table(detailed_df)
    robust_metric = args.robust_metric
    if args.robust_mode in ("validation_mad", "validation_topk"):
        robust_metric = "val_composite_mean"

    robust_seeds, seed_df = robust_seed_selection(
        seed_df,
        mode=args.robust_mode,
        metric=robust_metric,
        mad_k=args.robust_mad_k,
        top_k=args.robust_top_k,
        manual_seeds=args.robust_seeds,
    )
    seed_df.to_csv(output_dir / "seed_stability_scores_NASA.csv", index=False, encoding="utf-8-sig")

    robust_detailed_df = detailed_df[detailed_df["seed"].isin(robust_seeds)].copy()
    robust_score_df = robust_detailed_df[["re", "mae", "rmse"]].copy()
    robust_score_df.insert(0, "variant", "NASA_EOL_AttMoE_hardconstraint_robustseed_v1")
    robust_score_df.insert(1, "robust_mode", args.robust_mode)
    robust_score_df.insert(2, "robust_metric", robust_metric)
    robust_score_df.insert(3, "robust_mad_k", args.robust_mad_k)
    robust_score_df.insert(4, "seed", robust_detailed_df["seed"].values)
    robust_score_df.insert(5, "battery", robust_detailed_df["battery"].values)
    robust_score_df.insert(6, "best_epoch", robust_detailed_df["best_epoch"].values)
    robust_score_df.to_csv(output_dir / "final_training_scores_NASA_robust.csv", index=False, encoding="utf-8-sig")
    robust_score_df.to_csv(output_dir / "final_training_scores_NASA.csv", index=False, encoding="utf-8-sig")

    summary_rows = [
        summarize_score_rows(detailed_df, "all_seeds"),
        summarize_score_rows(robust_detailed_df, "robust_seeds"),
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_df["selected_robust_seeds"] = ",".join(str(seed) for seed in robust_seeds)
    summary_df["excluded_seeds"] = ",".join(str(seed) for seed in sorted(set(detailed_df["seed"].unique()) - set(robust_seeds)))
    summary_df["robust_mode"] = args.robust_mode
    summary_df["robust_metric"] = robust_metric
    summary_df["robust_mad_k"] = args.robust_mad_k
    summary_df.to_csv(output_dir / "robustseed_summary_NASA.csv", index=False, encoding="utf-8-sig")

    all_summary = summary_rows[0]
    robust_summary = summary_rows[1]

    print("=" * 72, flush=True)
    print("All-seed metrics:", flush=True)
    print("all re mean: {:<6.4f}".format(all_summary["re_mean"]), flush=True)
    print("all mae mean: {:<6.4f}".format(all_summary["mae_mean"]), flush=True)
    print("all rmse mean: {:<6.4f}".format(all_summary["rmse_mean"]), flush=True)
    print("Robust-seed metrics:", flush=True)
    print("robust selected seeds:", robust_seeds, flush=True)
    print("excluded seeds:", sorted(set(detailed_df["seed"].unique()) - set(robust_seeds)), flush=True)
    print("robust re mean: {:<6.4f}".format(robust_summary["re_mean"]), flush=True)
    print("robust mae mean: {:<6.4f}".format(robust_summary["mae_mean"]), flush=True)
    print("robust rmse mean: {:<6.4f}".format(robust_summary["rmse_mean"]), flush=True)
    print("robust rows with all three metrics improved:", robust_summary["all_three_improved_rows"], "/", robust_summary["num_rows"], flush=True)
    print("Best epochs:", detailed_df["best_epoch"].tolist(), flush=True)
    print("Saved:", output_dir / "final_training_scores_NASA_all.csv", flush=True)
    print("Saved:", output_dir / "final_training_scores_NASA_robust.csv", flush=True)
    print("Saved:", output_dir / "seed_stability_scores_NASA.csv", flush=True)
    print("Saved:", output_dir / "robustseed_summary_NASA.csv", flush=True)
    print("=" * 72, flush=True)

    return robust_score_df


# ============================================================
# command-line entry
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="AttMoE NASA hard-constraint robust-seed cloud training")

    parser.add_argument("--mode", choices=["quick", "grid", "best", "all"], default="quick")
    parser.add_argument("--data-source", choices=["auto", "raw", "npy"], default="auto")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=str(BASE_DIR / "outputs_NASA_EOL_AttMoE_hardconstraint_robustseed_v1"))

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=1)

    # NASA reproduced baseline-best parameters from the user's previous run.
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-experts", type=int, default=16)
    parser.add_argument("--feature-size", type=int, default=16)

    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--eol-state-dim", type=int, default=16)
    parser.add_argument("--eol-gamma-init", type=float, default=-4.0)
    parser.add_argument("--eol-correction-scale", type=float, default=0.02)
    parser.add_argument("--eol-loss-weight", type=float, default=0.0)
    parser.add_argument("--eol-loss-band", type=float, default=0.08)
    parser.add_argument("--eol-threshold-norm", type=float, default=0.7)
    parser.add_argument("--eol-enable-correction", action="store_true", default=True)
    parser.add_argument("--eol-disable-correction", dest="eol_enable_correction", action="store_false")

    parser.add_argument("--best-metric", choices=["rmse", "mae", "re", "composite", "hard"], default="hard")
    parser.add_argument("--re-weight", type=float, default=0.05)
    # Defaults are the user's reproduced NASA baseline. If comparing with the paper, pass 0.2000/0.0760/0.0872 manually.
    parser.add_argument("--baseline-re", type=float, default=0.2005)
    parser.add_argument("--baseline-mae", type=float, default=0.0780)
    parser.add_argument("--baseline-rmse", type=float, default=0.0892)
    parser.add_argument("--hard-priority", choices=["all", "strict_re"], default="all")
    parser.add_argument("--eval-interval", type=int, default=50)

    parser.add_argument("--robust-mode", choices=["validation_topk", "topk", "validation_mad", "metric_mad", "manual", "none"], default="validation_topk")
    parser.add_argument(
        "--robust-metric",
        choices=[
            "test_composite_mean",
            "test_re_mean",
            "test_mae_mean",
            "test_rmse_mean",
            "val_composite_mean",
            "val_re_mean",
            "val_mae_mean",
            "val_rmse_mean",
        ],
        default="val_composite_mean",
    )
    parser.add_argument("--robust-mad-k", type=float, default=1.0)
    parser.add_argument("--robust-top-k", type=int, default=4)
    parser.add_argument("--robust-seeds", type=str, default="")

    parser.add_argument("--lr-grid", type=str, default="")
    parser.add_argument("--hidden-dim-grid", type=str, default="")
    parser.add_argument("--num-experts-grid", type=str, default="")
    parser.add_argument("--feature-size-grid", type=str, default="")

    parser.add_argument("--require-gpu", action="store_true")

    return parser.parse_args()


def main():
    global Battery
    global Rated_Capacity
    global device

    args = parse_args()

    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")

    if args.seeds < 1:
        raise ValueError("--seeds must be at least 1.")

    if args.hidden_dim % args.nhead != 0:
        raise ValueError("hidden_dim={} must be divisible by nhead={}.".format(args.hidden_dim, args.nhead))

    if args.require_gpu and not torch.npu.is_available():
        raise RuntimeError("CUDA GPU is required, but torch.npu.is_available() is False.")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print_device_information()
    print("MoE backend:", MOE_BACKEND, flush=True)
    print("Output directory:", output_dir, flush=True)
    print(
        "best_metric={}, baselines=({},{},{}), hard_priority={}, eval_interval={}".format(
            args.best_metric,
            args.baseline_re,
            args.baseline_mae,
            args.baseline_rmse,
            args.hard_priority,
            args.eval_interval,
        ),
        flush=True,
    )
    print(
        "robust_mode={}, robust_metric={}, robust_top_k={}, robust_mad_k={}, robust_seeds={}".format(
            args.robust_mode,
            args.robust_metric,
            args.robust_top_k,
            args.robust_mad_k,
            args.robust_seeds,
        ),
        flush=True,
    )

    Battery = load_battery_data(
        data_source=args.data_source,
        data_dir=args.data_dir,
    )
    Rated_Capacity = 2.0

    save_capacity_figure(Battery, output_dir)

    if args.mode in ("quick", "grid", "all"):
        result_df = run_grid_search(args, output_dir)

        if args.mode in ("quick", "grid"):
            return

        best = result_df.iloc[0]
        args.feature_size = int(best["feature_size"])
        args.lr = float(best["lr"])
        args.hidden_dim = int(best["hidden_dim"])
        args.num_experts = int(best["num_experts"])

    run_best_training(args, output_dir)


if __name__ == "__main__":
    main()
