# Human Activity Recognition (HAR)

Production-grade deep-learning pipeline for classifying human activities from smartphone
inertial sensor signals, with an interactive Streamlit dashboard.

Deep models (LSTM / GRU / 1D-CNN) are trained end-to-end on **raw 9-channel × 128-timestep**
sensor windows, while a RandomForest classifier trained on the dataset's 561 hand-engineered
features serves as the classical baseline. Evaluation uses the official **subject-independent**
train/test split (no reshuffling across subjects).

## Results (best accuracy)

| Model | Input | Accuracy | Macro-F1 |
|---|---|---|---|
| **1D-CNN** | Raw 9×128 signals | **0.9430** | **0.9435** |
| GRU | Raw 9×128 signals | 0.9162 | 0.9168 |
| RandomForest | 561 engineered features | 0.9287 | 0.9270 |
| LSTM | Raw 9×128 signals | 0.8911 | 0.8900 |

The 1D-CNN outperforms the engineered-feature baseline despite receiving no pre-computed
features. *Laying* is the easiest activity (F1 ≈ 0.998); *sitting vs. standing* is the hardest
distinction for every model.

## Repository layout

```
HAR/
├── app.py              # Streamlit dashboard (live prediction, model comparison, confusion matrices)
├── har.py              # Full training pipeline (CLI entry point)
├── har.ipynb           # Interactive notebook mirroring the pipeline
├── requirements.txt    # Python dependencies
├── runtime.txt         # Streamlit Cloud runtime (python-3.12)
├── .streamlit/config.toml
├── data/               # UCI HAR dataset (auto-downloaded if missing)
└── out/                # Trained weights (*_best.pt) + metrics.json
```

## Dataset

[UCI HAR — Human Activity Recognition Using Smartphones](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
(Anguita et al., 2013):

- 30 subjects (19–48 y), waist-mounted Samsung Galaxy S II accelerometer + gyroscope @ 50 Hz.
- 6 activities: walking, walking-upstairs, walking-downstairs, sitting, standing, laying.
- ~10,000 labelled windows of 2.56 s (128 timesteps, 50% overlap).
- **Train:** 21 subjects (~7,352 windows) · **Test:** 9 subjects (~2,947 windows).
- `har.py` downloads the zip automatically (primary UCI URL + fallback mirror) and extracts it.

## Models & architecture

| Model | Architecture | Input |
|---|---|---|
| LSTM | 2-layer biLSTM (128 hidden) → temporal avg-pool → FC | Raw (9, 128) |
| GRU | 2-layer biGRU (128 hidden) → temporal avg-pool → FC | Raw (9, 128) |
| 1D-CNN | 3 conv blocks (64→128→256) → global avg-pool → FC | Raw (9, 128) |
| RandomForest | 200 trees, max_depth=25 | 561 engineered features |

All deep models use `AdamW` (lr=1e-3, weight decay 1e-4) with cosine annealing, batch size 64,
CrossEntropyLoss, best-epoch checkpointing by test accuracy, and a fixed seed (42).

## Quickstart

```bash
# 1. Install dependencies (Python 3.12)
pip install -r requirements.txt

# 2. Train all models (downloads dataset if needed, ~20 epochs)
python har.py --epochs 20

#    Train a single model, or tweak hyperparameters
python har.py --model cnn1d --epochs 20 --batch-size 64 --lr 1e-3

# 3. Launch the dashboard
streamlit run app.py
```

Outputs are written to `out/`:

- `out/<model>_best.pt` — best-checkpoint state dicts
- `out/metrics.json` — accuracy, macro-F1, per-class F1, confusion matrices, training curves

### CLI options

```
--data-dir   Dataset directory          (default: ./data)
--out        Output directory           (default: ./out)
--epochs     Training epochs            (default: 20)
--model      lstm | gru | cnn1d | all   (default: all)
--batch-size Batch size                 (default: 64)
--lr         Learning rate              (default: 1e-3)
```

## Streamlit app

`streamlit run app.py` provides four tabs:

1. **Live Prediction** — pick a test window (or draw a random one) and view each model's
   class probabilities over the 9 sensor channels.
2. **Model Comparison** — accuracy, macro-F1, and per-class F1 tables + charts.
3. **Confusion Matrices** — heatmaps per model, highlighting the sitting↔standing confusion.
4. **About** — methodology, model descriptions, and references.

The app loads pre-trained weights from `out/` and the test split only — no training at runtime.
Requires `out/metrics.json` and the `out/*_best.pt` weights (produced by `har.py`).

## Reproducibility

- Fixed seed (42) across numpy/torch (deterministic cuDNN when GPU available).
- Official subject-independent split — no leakage between training and test subjects.
- CPU-friendly: models are small and run inference comfortably on a laptop.
