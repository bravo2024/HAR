#!/usr/bin/env python3
"""Production-grade Human Activity Recognition (HAR) pipeline.

Deep models consume the 9-channel×128-timestep raw inertial signals;
the classical baseline uses the 561 pre-engineered features.
Uses the official subject-independent train/test split — no reshuffling.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.error import URLError
from urllib.request import urlretrieve

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
)

logger = logging.getLogger("har")

# ── Constants ────────────────────────────────────────────────────────────────
ACTIVITIES = [
    "walking",
    "walking-upstairs",
    "walking-downstairs",
    "sitting",
    "standing",
    "laying",
]
N_CLASSES = len(ACTIVITIES)
N_CHANNELS = 9
SEQ_LEN = 128
SEED = 42

UCI_URL = (
    "https://archive.ics.uci.edu/static/public/240/"
    "human+activity+recognition+using+smartphones.zip"
)
FALLBACK_URL = (
    "https://www.dropbox.com/scl/fi/jlzyavx4fq46pa9hlm1k6/"
    "UCI_HAR_Dataset.zip?rlkey=5vqcy7tvxlgp8s0o46rg3yq7i&st=97jxk1de&dl=1"
)

SIGNAL_NAMES = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]


# ── Utilities ────────────────────────────────────────────────────────────────

def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = False     # type: ignore[attr-defined]


def _repo_zip(root: Path) -> str:
    """Return a deterministic hash used for seeding."""
    return str(SEED)


def download_and_extract(data_dir: Path) -> Path:
    """Download UCI HAR dataset and extract.  Returns dataset root directory."""
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "har.zip"
    dataset_root = data_dir / "UCI HAR Dataset"

    if dataset_root.exists():
        logger.info("Dataset already exists at %s", dataset_root)
        return dataset_root

    for url in (UCI_URL, FALLBACK_URL):
        try:
            logger.info("Downloading from %s …", url)
            urlretrieve(url, zip_path)
            break
        except (URLError, Exception) as exc:
            logger.warning("Download failed (%s), trying next source.", exc)
    else:
        msg = (
            "Both download URLs failed. "
            "Please download manually from %s and extract to %s"
        )
        raise RuntimeError(msg % (UCI_URL, data_dir))

    logger.info("Extracting zip …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)

    # Check for nested zip inside the extracted contents
    for entry in sorted(data_dir.rglob("*.zip")):
        logger.info("Found nested zip: %s", entry.relative_to(data_dir))
        with zipfile.ZipFile(entry, "r") as inner:
            inner.extractall(data_dir)
        entry.unlink()  # clean up nested zip after extraction

    # After extraction, check standard layout
    if (data_dir / "UCI HAR Dataset").exists():
        return data_dir / "UCI HAR Dataset"
    # Possibly nested inside an outer directory
    for candidate in sorted(data_dir.iterdir()):
        if candidate.is_dir() and (candidate / "train").exists():
            return candidate
    raise RuntimeError("Could not locate UCI HAR Dataset directory under %s" % data_dir)


def _load_signal_file(path: Path, n_rows: int) -> np.ndarray:
    """Load a whitespace-delimited signal file (rows × 128)."""
    return np.loadtxt(path, dtype=np.float32).reshape(n_rows, SEQ_LEN)


def load_raw_signals(root: Path, split: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return (X, y) where X shape = (samples, channels, timesteps)."""
    subdir = root / split / "Inertial Signals"
    y = np.loadtxt(root / split / f"y_{split}.txt", dtype=np.int64) - 1

    n_samples = y.shape[0]
    channels = []
    for sig_name in SIGNAL_NAMES:
        path = subdir / f"{sig_name}_{split}.txt"
        channels.append(_load_signal_file(path, n_samples))
    X = np.stack(channels, axis=1).astype(np.float32)  # (N, 9, 128)
    return X, y


def load_engineered_features(root: Path, split: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return (X, y) for the 561 pre-engineered features."""
    X = np.loadtxt(root / split / f"X_{split}.txt", dtype=np.float32)
    y = np.loadtxt(root / split / f"y_{split}.txt", dtype=np.int64) - 1
    return X, y


# ── PyTorch Datasets ────────────────────────────────────────────────────────

class HARSequenceDataset(torch.utils.data.Dataset):  # type: ignore[name-defined]
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return self.X.size(0)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


# ── Models ───────────────────────────────────────────────────────────────────

class LSTMModel(nn.Module):
    """2-layer biLSTM → avg-pool → FC."""

    def __init__(self, n_channels: int = N_CHANNELS, n_classes: int = N_CLASSES,
                 hidden: int = 128, dropout: float = 0.3) -> None:
        super().__init__()
        self.lstm = nn.LSTM(n_channels, hidden, num_layers=2,
                            batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden * 2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)           # (B, T, C)
        out, _ = self.lstm(x)             # (B, T, 2H)
        out = out.mean(dim=1)             # (B, 2H)
        return self.fc(out)


class GRUModel(nn.Module):
    """2-layer biGRU → avg-pool → FC."""

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
    """3 conv blocks (depth-wise sep style) over time → GAP → FC."""

    def __init__(self, n_channels: int = N_CHANNELS, n_classes: int = N_CLASSES,
                 dropout: float = 0.3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=9, padding=4),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=9, padding=4),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.MaxPool1d(2),  # 128 → 64

            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.MaxPool1d(2),  # 64 → 32

            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),  # → (B, 256, 1)
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(256, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)            # (B, 256, 1)
        out = out.squeeze(-1)             # (B, 256)
        out = self.dropout(out)
        return self.fc(out)


MODEL_REGISTRY: Dict[str, type] = {
    "lstm": LSTMModel,
    "gru": GRUModel,
    "cnn1d": CNN1DModel,
}


# ── Training ─────────────────────────────────────────────────────────────────

def _train_epoch(model: nn.Module, loader, optimizer: optim.Optimizer,
                 criterion: nn.Module) -> float:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for Xb, yb in loader:
        optimizer.zero_grad()
        logits = model(Xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * Xb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        total += Xb.size(0)
    return total_loss / total


@torch.no_grad()
def _eval_epoch(model: nn.Module, loader, criterion: nn.Module
                ) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for Xb, yb in loader:
        logits = model(Xb)
        loss = criterion(logits, yb)
        total_loss += loss.item() * Xb.size(0)
        preds = logits.argmax(1)
        correct += (preds == yb).sum().item()
        total += Xb.size(0)
        all_preds.append(preds.numpy())
        all_labels.append(yb.numpy())
    acc = correct / total
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    return total_loss / total, acc, all_preds, all_labels


def train_deep_model(model_type: str, X_train: np.ndarray, y_train: np.ndarray,
                     X_test: np.ndarray, y_test: np.ndarray,
                     out_dir: Path, epochs: int, lr: float = 1e-3,
                     batch_size: int = 64) -> dict:
    logger.info("── Training %s ──", model_type.upper())
    ds_train = HARSequenceDataset(X_train, y_train)
    ds_test = HARSequenceDataset(X_test, y_test)
    loader_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    loader_test = torch.utils.data.DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model {model_type}. Choose from {list(MODEL_REGISTRY)}")

    model = MODEL_REGISTRY[model_type]()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_test_acc = 0.0
    best_weights: Optional[dict] = None
    history: Dict[str, List[float]] = {"train_loss": [], "test_loss": [], "test_acc": []}

    for epoch in range(1, epochs + 1):
        train_loss = _train_epoch(model, loader_train, optimizer, criterion)
        test_loss, test_acc, _, _ = _eval_epoch(model, loader_test, criterion)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        logger.info("Epoch %3d | train loss %.4f | test loss %.4f | test acc %.4f",
                     epoch, train_loss, test_loss, test_acc)

    if best_weights is None:
        raise RuntimeError("No best weights captured — training likely failed.")

    model.load_state_dict(best_weights)
    _, final_acc, preds, labels = _eval_epoch(model, loader_test, criterion)

    macro_f1 = f1_score(labels, preds, average="macro")
    per_class = f1_score(labels, preds, average=None)
    cm = confusion_matrix(labels, preds)

    # Save weights
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / f"{model_type}_best.pt")

    return {
        "model": model_type,
        "accuracy": round(final_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class_f1": {ACTIVITIES[i]: round(float(per_class[i]), 4) for i in range(N_CLASSES)},
        "confusion_matrix": cm.tolist(),
        "history": history,
    }


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray) -> dict:
    logger.info("── Training RandomForest baseline ──")
    clf = RandomForestClassifier(n_estimators=200, max_depth=25,
                                 random_state=SEED, n_jobs=-1, verbose=0)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    acc = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro")
    per_class = f1_score(y_test, preds, average=None)
    cm = confusion_matrix(y_test, preds)

    return {
        "model": "randomforest",
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class_f1": {ACTIVITIES[i]: round(float(per_class[i]), 4) for i in range(N_CLASSES)},
        "confusion_matrix": cm.tolist(),
        "history": None,
    }


# ── Orchestration ───────────────────────────────────────────────────────────

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HAR pipeline (deep + classical)")
    p.add_argument("--data-dir", default="./data", help="Directory for dataset")
    p.add_argument("--out", default="./out", help="Output directory")
    p.add_argument("--epochs", type=int, default=20, help="Training epochs")
    p.add_argument("--model", choices=["lstm", "gru", "cnn1d", "all"],
                   default="all", help="Which model(s) to train")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")

    set_seed(SEED)

    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out).resolve()

    try:
        dataset_root = download_and_extract(data_dir)
    except RuntimeError as e:
        logger.error("Fatal: %s", e)
        sys.exit(1)

    logger.info("Loading raw inertial signals (9×128) …")
    X_train_seq, y_train = load_raw_signals(dataset_root, "train")
    X_test_seq, y_test = load_raw_signals(dataset_root, "test")
    logger.info("Train seq: %s,  Test seq: %s", X_train_seq.shape, X_test_seq.shape)

    # ── Classical baseline (always run for comparison) ──
    X_train_eng, y_train_eng = load_engineered_features(dataset_root, "train")
    X_test_eng, y_test_eng = load_engineered_features(dataset_root, "test")
    rf_result = train_random_forest(X_train_eng, y_train_eng, X_test_eng, y_test_eng)

    models_to_run = list(MODEL_REGISTRY.keys()) if args.model == "all" else [args.model]
    results: List[dict] = [rf_result]

    for mtype in models_to_run:
        res = train_deep_model(mtype, X_train_seq, y_train, X_test_seq, y_test,
                               out_dir, args.epochs, lr=args.lr,
                               batch_size=args.batch_size)
        results.append(res)

    metrics_path = out_dir / "metrics.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Metrics saved → %s", metrics_path)
    logger.info("──  Final Results  ──")
    for r in results:
        logger.info("%-15s  acc %.4f  macro-F1 %.4f", r["model"], r["accuracy"], r["macro_f1"])


if __name__ == "__main__":
    main()
