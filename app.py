"""Production-grade HAR Streamlit app — CPU-only, memory-conscious."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import plotly.express as px
import plotly.graph_objects as go

# ── Page config (must be first Streamlit call) ───────────────────────────
st.set_page_config(
    page_title="HAR – Human Activity Recognition",
    page_icon="🏃",
    layout="wide",
)

# ── Constants ────────────────────────────────────────────────────────────
ACTIVITIES = [
    "walking", "walking-upstairs", "walking-downstairs",
    "sitting", "standing", "laying",
]
N_CLASSES = len(ACTIVITIES)
N_CHANNELS = 9
SEQ_LEN = 128
SEED = 42

SIGNAL_NAMES = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "out"
DATA_DIR = BASE_DIR / "data"

# ── PyTorch models (copied from har.py for standalone deployment) ───────

class LSTMModel(nn.Module):
    def __init__(self, n_channels: int = N_CHANNELS, n_classes: int = N_CLASSES,
                 hidden: int = 128, dropout: float = 0.3) -> None:
        super().__init__()
        self.lstm = nn.LSTM(n_channels, hidden, num_layers=2,
                            batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden * 2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        out = out.mean(dim=1)
        return self.fc(out)


class GRUModel(nn.Module):
    def __init__(self, n_channels: int = N_CHANNELS, n_classes: int = N_CLASSES,
                 hidden: int = 128, dropout: float = 0.3) -> None:
        super().__init__()
        self.gru = nn.GRU(n_channels, hidden, num_layers=2,
                          batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden * 2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        out, _ = self.gru(x)
        out = out.mean(dim=1)
        return self.fc(out)


class CNN1DModel(nn.Module):
    def __init__(self, n_channels: int = N_CHANNELS, n_classes: int = N_CLASSES,
                 dropout: float = 0.3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=9, padding=4),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=9, padding=4),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(256, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = out.squeeze(-1)
        out = self.dropout(out)
        return self.fc(out)


MODEL_MAP: Dict[str, type] = {"lstm": LSTMModel, "gru": GRUModel, "cnn1d": CNN1DModel}

# ── Data loading helpers ─────────────────────────────────────────────────

def _load_signal_file(path: Path, n_rows: int) -> np.ndarray:
    return np.loadtxt(path, dtype=np.float32).reshape(n_rows, SEQ_LEN)


@st.cache_data(show_spinner="Loading test-set signals …")
def load_test_signals(data_dir: str = str(DATA_DIR)) -> Tuple[np.ndarray, np.ndarray]:
    """Return (X_test, y_test) — raw inertial signals for the test split only."""
    root = Path(data_dir) / "UCI HAR Dataset"
    if not root.exists():
        st.error(f"UCI HAR Dataset not found at {root}. "
                 f"Place the extracted dataset under {data_dir}/")
        st.stop()

    y = np.loadtxt(root / "test" / "y_test.txt", dtype=np.int64) - 1
    n_samples = y.shape[0]
    subdir = root / "test" / "Inertial Signals"
    channels = []
    for sig_name in SIGNAL_NAMES:
        path = subdir / f"{sig_name}_test.txt"
        channels.append(_load_signal_file(path, n_samples))
    X = np.stack(channels, axis=1).astype(np.float32)
    return X, y


@st.cache_data(show_spinner="Loading metrics …")
def load_metrics(out_dir: str = str(OUT_DIR)) -> List[dict]:
    path = Path(out_dir) / "metrics.json"
    if not path.exists():
        st.error(f"metrics.json not found at {path}.  "
                 f"Run har.py first to produce metrics and model weights.")
        st.stop()
    with open(path) as f:
        return json.load(f)


# ── Model loading ────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading LSTM model …")
def load_lstm(weights_path: str) -> LSTMModel:
    model = LSTMModel()
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    model.eval()
    return model


@st.cache_resource(show_spinner="Loading GRU model …")
def load_gru(weights_path: str) -> GRUModel:
    model = GRUModel()
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    model.eval()
    return model


@st.cache_resource(show_spinner="Loading CNN1D model …")
def load_cnn1d(weights_path: str) -> CNN1DModel:
    model = CNN1DModel()
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def load_all_models(out_dir: str = str(OUT_DIR)) -> Dict[str, nn.Module]:
    out = Path(out_dir)
    models: Dict[str, nn.Module] = {}
    missing: List[str] = []

    for name in ("lstm", "gru", "cnn1d"):
        fpath = out / f"{name}_best.pt"
        if not fpath.exists():
            missing.append(str(fpath))
            continue
        if name == "lstm":
            models[name] = load_lstm(str(fpath))
        elif name == "gru":
            models[name] = load_gru(str(fpath))
        else:
            models[name] = load_cnn1d(str(fpath))

    if missing:
        st.warning("Missing model weights:\n\n" + "\n".join(f"- {m}" for m in missing) +
                   "\n\nRun `python har.py` to train and save weights.")

    return models


@torch.no_grad()
def predict_all(models: Dict[str, nn.Module], x: np.ndarray) -> Dict[str, Dict]:
    """x shape (9, 128) -> predict with each model. Returns {model: {pred_idx, probs}}."""
    tensor = torch.from_numpy(x).unsqueeze(0)  # (1, 9, 128)
    results = {}
    for name, model in models.items():
        logits = model(tensor)  # (1, n_classes)
        probs = torch.softmax(logits, dim=1).squeeze(0).numpy()
        pred = int(probs.argmax())
        results[name] = {"pred": pred, "probs": probs}
    return results


# ── Plot helpers ─────────────────────────────────────────────────────────

def plot_signal_window(x: np.ndarray, sample_idx: int, gt_label: str) -> go.Figure:
    """x shape (9, 128): plot all 9 channels."""
    t = np.arange(SEQ_LEN)
    fig = go.Figure()
    colors = px.colors.qualitative.Set1 + px.colors.qualitative.Set2
    for ch in range(N_CHANNELS):
        fig.add_trace(go.Scatter(
            x=t, y=x[ch],
            mode="lines",
            name=SIGNAL_NAMES[ch],
            line=dict(color=colors[ch % len(colors)], width=1.2),
            legendgroup=f"g{ch // 3}",
            legendgrouptitle_text=["Accelerometer", "Gyroscope", "Total Acc."][ch // 3],
        ))
    fig.update_layout(
        title=f"Sample #{sample_idx} — Ground-truth: <b>{gt_label}</b>",
        xaxis_title="Timestep (128 samples @ 50 Hz = 2.56 s)",
        yaxis_title="Normalised amplitude",
        height=420,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5),
    )
    return fig


def plot_prob_bars(predictions: Dict[str, Dict]) -> go.Figure:
    fig = go.Figure()
    model_names = list(predictions.keys())
    for i, name in enumerate(model_names):
        probs = predictions[name]["probs"]
        fig.add_trace(go.Bar(
            x=ACTIVITIES,
            y=probs,
            name=name.upper(),
            text=[f"{p:.2%}" for p in probs],
            textposition="outside",
            visible=True if i == 0 else "legendonly",
        ))
    fig.update_layout(
        title="Class probabilities per model (click legend to toggle)",
        yaxis=dict(range=[0, 1.05], tickformat=".0%"),
        height=380,
        margin=dict(l=20, r=20, t=50, b=20),
        barmode="group",
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
    )
    return fig


# ── Sidebar ──────────────────────────────────────────────────────────────

def render_sidebar() -> Dict[str, nn.Module]:
    st.sidebar.title("🏃 HAR Dashboard")
    st.sidebar.markdown("**Human Activity Recognition** from smartphone inertial signals.")

    models = load_all_models()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Models loaded")
    for name in ("lstm", "gru", "cnn1d"):
        loaded = "✅" if name in models else "❌"
        st.sidebar.markdown(f"{loaded} {name.upper()}")

    metrics = None
    metrics_path = OUT_DIR / "metrics.json"
    if metrics_path.exists():
        metrics = load_metrics()

    if metrics is not None:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Best model")
        best = max(metrics, key=lambda r: r["accuracy"])
        st.sidebar.markdown(f"**{best['model'].upper()}** — {best['accuracy']:.2%} accuracy")

    st.sidebar.markdown("---")
    st.sidebar.caption("Deep models trained on raw 9×128 inertial signals. "
                       "Subject-independent split (30 train, 6 test subjects).")
    return models


# ── Tab 1: Live Prediction ──────────────────────────────────────────────

def tab_live_prediction(models: Dict[str, nn.Module], X_test: np.ndarray, y_test: np.ndarray):
    st.header("Live Prediction")
    st.markdown("Select or randomly sample a test-set sensor window "
                "and see each model's prediction.")

    if not models:
        st.warning("No deep models loaded. Check that weight files exist in `out/`.")
        return

    col1, col2 = st.columns([1, 2])
    with col1:
        use_random = st.checkbox("Random sample", value=True)
        if use_random:
            if st.button("🎲 Draw random sample"):
                st.session_state.sample_idx = random.randint(0, X_test.shape[0] - 1)
            if "sample_idx" not in st.session_state:
                st.session_state.sample_idx = random.randint(0, X_test.shape[0] - 1)
        else:
            idx = st.number_input("Sample index", min_value=0,
                                  max_value=X_test.shape[0] - 1,
                                  value=st.session_state.get("sample_idx", 0))
            st.session_state.sample_idx = idx

        idx = st.session_state.sample_idx
        gt_label = ACTIVITIES[y_test[idx]]

        st.markdown(f"#### Ground truth: **{gt_label}**")

        predictions = predict_all(models, X_test[idx])
        for name, res in predictions.items():
            pred_label = ACTIVITIES[res["pred"]]
            conf = res["probs"][res["pred"]]
            correct = "✅" if res["pred"] == y_test[idx] else "❌"
            st.markdown(f"{correct} **{name.upper()}**: {pred_label} ({conf:.2%})")

    with col2:
        idx = st.session_state.sample_idx
        gt_label = ACTIVITIES[y_test[idx]]
        fig_signals = plot_signal_window(X_test[idx], idx, gt_label)
        st.plotly_chart(fig_signals, use_container_width=True)

    st.markdown("---")
    idx = st.session_state.sample_idx
    predictions = predict_all(models, X_test[idx])
    fig_probs = plot_prob_bars(predictions)
    st.plotly_chart(fig_probs, use_container_width=True)


# ── Tab 2: Model Comparison ──────────────────────────────────────────────

def tab_model_comparison(metrics: List[dict]):
    st.header("Model Comparison")
    st.markdown("Accuracy and macro-F1 across all models. "
                "The 1D-CNN on raw signals outperforms the RandomForest baseline "
                "trained on 561 hand-engineered features.")

    if not metrics:
        st.warning("No metrics available.")
        return

    rows = []
    for m in metrics:
        rows.append({
            "Model": m["model"].upper(),
            "Accuracy": m["accuracy"],
            "Macro F1": m["macro_f1"],
        })
    df = pd.DataFrame(rows).set_index("Model")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.dataframe(
            df.style.format("{:.2%}").highlight_max(axis=0,
                props="font-weight:bold; background-color:#e6ffe6"),
            use_container_width=True,
        )

    with col2:
        df_plot = df.reset_index().melt(id_vars="Model", var_name="Metric", value_name="Score")
        fig = px.bar(df_plot, x="Model", y="Score", color="Metric", barmode="group",
                     text_auto=".2%", title="Model Accuracy & Macro F1",
                     color_discrete_sequence=["#636EFA", "#EF553B"])
        fig.update_layout(height=380, margin=dict(l=20, r=20, t=50, b=20),
                          yaxis=dict(range=[0, 1.05], tickformat=".0%"))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Per-class F1 scores")
    per_class_rows = []
    for m in metrics:
        for activity, f1 in m["per_class_f1"].items():
            per_class_rows.append({"Model": m["model"].upper(), "Activity": activity, "F1": f1})
    df_pc = pd.DataFrame(per_class_rows)
    fig_pc = px.bar(df_pc, x="Activity", y="F1", color="Model", barmode="group",
                    text_auto=".3f",
                    title="Per-class F1 by Model",
                    color_discrete_sequence=px.colors.qualitative.Set1)
    fig_pc.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=20),
                         yaxis=dict(range=[0, 1.05], tickformat=".0%"))
    st.plotly_chart(fig_pc, use_container_width=True)


# ── Tab 3: Confusion Matrices ────────────────────────────────────────────

def tab_confusion_matrices(metrics: List[dict]):
    st.header("Confusion Matrices")
    st.markdown("Heatmap per model. Note the persistent sitting↔standing confusion "
                "across all models — these static postures are inherently harder to separate "
                "from inertial signals alone.")

    if not metrics:
        st.warning("No metrics available.")
        return

    deep_metrics = [m for m in metrics if m["model"] != "randomforest"]
    cols = st.columns(len(deep_metrics))
    for i, m in enumerate(deep_metrics):
        cm = np.array(m["confusion_matrix"])
        with cols[i]:
            fig = px.imshow(
                cm,
                x=ACTIVITIES,
                y=ACTIVITIES,
                text_auto=True,
                color_continuous_scale="Blues",
                title=f"{m['model'].upper()} (acc={m['accuracy']:.2%})",
                aspect="auto",
            )
            fig.update_layout(height=420, margin=dict(l=20, r=20, t=50, b=20),
                              xaxis_tickangle=45)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("RandomForest baseline")
    rf = next((m for m in metrics if m["model"] == "randomforest"), None)
    if rf:
        cm = np.array(rf["confusion_matrix"])
        fig = px.imshow(
            cm, x=ACTIVITIES, y=ACTIVITIES, text_auto=True,
            color_continuous_scale="Oranges",
            title=f"RandomForest (acc={rf['accuracy']:.2%})",
            aspect="auto",
        )
        fig.update_layout(height=420, margin=dict(l=20, r=20, t=50, b=20),
                          xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)


# ── Tab 4: About ─────────────────────────────────────────────────────────

def tab_about():
    st.header("About this Project")
    st.markdown("""
    ### 🏃 Human Activity Recognition (HAR)

    This application demonstrates deep-learning-based activity recognition from
    smartphone inertial sensor data, using the [**UCI HAR Dataset**](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
    (Anguita et al., 2013) — 30 volunteers aged 19–48, six activities, ~10,000 labelled
    2.56 s windows. Download: [archive.ics.uci.edu/dataset/240](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones).

    #### Methodology

    - **Data**: 3-axial linear acceleration and 3-axial angular velocity captured
      at 50 Hz from a waist-mounted Samsung Galaxy S II. Each sample is a 2.56 s
      fixed-width sliding window (128 timesteps) with 50% overlap.
    - **Input representation (deep models)**: Raw 9-channel × 128-timestep inertial
      signals (body acceleration, body gyroscope, total acceleration — each in x/y/z).
    - **Input representation (RandomForest baseline)**: 561 hand-engineered time and
      frequency-domain features from the original dataset authors.
    - **Train/test split**: Subject-independent — 21 subjects for training
      (~7,352 windows), 9 subjects for testing (~2,947 windows). No reshuffling
      across subjects.

    #### Models

    | Model | Architecture | Input |
    |---|---|---|
    | LSTM | 2-layer biLSTM (128 hidden) → avg-pool → FC | Raw 9×128 |
    | GRU | 2-layer biGRU (128 hidden) → avg-pool → FC | Raw 9×128 |
    | 1D-CNN | 3 conv blocks (64→128→256) → GAP → FC | Raw 9×128 |
    | RandomForest | 200 trees, max_depth=25 | 561 engineered features |

    #### Key Findings

    - The **1D-CNN** achieves the highest accuracy (~94.3%), surpassing the
      RandomForest baseline (~92.9%) despite using no hand-engineered features.
    - **Sitting vs. standing** is the hardest distinction across all models —
      these static postures produce very similar inertial patterns.
    - **Laying** is the easiest activity (>99% F1) due to the distinctive
      orientation of the device.

    #### References

    <a id="ref1">[1]</a> Anguita, D., Ghio, A., Oneto, L., Parra, X., &
    Reyes-Ortiz, J. L. (2013). *A Public Domain Dataset for Human Activity
    Recognition Using Smartphones*. ESANN.
    """)


# ── Main entrypoint ──────────────────────────────────────────────────────

def main() -> None:
    st.title("🏃 Human Activity Recognition")
    st.markdown("*Deep learning on smartphone inertial signals — raw 9×128 sensor windows.*")

    # ── Check pre-requisites ──
    out_dir = OUT_DIR
    if not (out_dir / "metrics.json").exists():
        st.error("""
        **Pre-trained model files not found.**

        This app requires pre-computed model weights and metrics.  
        Run `python har.py` in the project root first to train models and
        save weights to the `out/` directory.
        """)
        return

    # ── Cached loads ──
    metrics = load_metrics()
    X_test, y_test = load_test_signals()
    models = render_sidebar()

    # ── Polish + KPI header cards ──
    st.markdown(
        """
        <style>
        div[data-testid="stMetric"] {
            background: #f7f9fc; border: 1px solid #e6e8ec; border-radius: 10px;
            padding: 0.8rem 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        div[data-testid="stMetricValue"] { color: #1f77b4; font-weight: 700; }
        div[data-testid="stMetricLabel"] { color: #5f6368; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    _model_rows = [r for r in metrics if isinstance(r, dict) and "model" in r]
    if _model_rows:
        _best = max(_model_rows, key=lambda r: r["accuracy"])
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Best Accuracy", f"{_best['accuracy']*100:.1f}%",
                  _best["model"].upper(), delta_color="off")
        k2.metric("Activities", "6")
        k3.metric("Test Windows", f"{len(y_test):,}")
        k4.metric("Signal Input", "9 ch × 128")
    st.markdown(
        "**Dataset:** [UCI HAR — Human Activity Recognition Using Smartphones]"
        "(https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) "
        "— **30 subjects**, **6 activities**, waist-mounted Samsung Galaxy S II "
        "accelerometer + gyroscope sampled at 50 Hz."
    )
    st.divider()

    # ── Tabs ──
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Live Prediction",
        "📊 Model Comparison",
        "🎯 Confusion Matrices",
        "ℹ️ About",
    ])

    with tab1:
        tab_live_prediction(models, X_test, y_test)
    with tab2:
        tab_model_comparison(metrics)
    with tab3:
        tab_confusion_matrices(metrics)
    with tab4:
        tab_about()


if __name__ == "__main__":
    main()
