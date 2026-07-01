"""
Model Accuracy Evaluation & Graph Generation
=============================================
Evaluates both Stage 1 (Beat CNN) and Stage 2 (Arrhythmia Ensemble)
on the MIT-BIH DS2 test set and saves all graphs to evaluation_results/
"""

import os, pickle, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import wfdb
import tensorflow as tf
from collections import Counter
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH   = 'Dataset/mit-bih-arrhythmia-database-1.0.0/'
MODEL_PATH  = 'best_ecg_model.keras'
CLF_PATH    = 'arrhythmia_classifier.pkl'
OUT_DIR     = 'evaluation_results'
os.makedirs(OUT_DIR, exist_ok=True)

CLASSES     = ['N', 'S', 'V', 'F', 'Q']
CLASS_NAMES = {
    'N': 'Normal', 'S': 'Supraventricular',
    'V': 'Ventricular', 'F': 'Fusion', 'Q': 'Unknown/Paced'
}
CLASS_COLORS = {
    'N': '#22c55e', 'S': '#f59e0b', 'V': '#ef4444',
    'F': '#8b5cf6', 'Q': '#94a3b8'
}

# LabelEncoder.fit(['N','S','V','F','Q']) sorts alphabetically → F=0,N=1,Q=2,S=3,V=4
# Model output index → class label
MODEL_IDX_TO_CLASS = ['F', 'N', 'Q', 'S', 'V']

AAMI_MAP = {
    'N':'N','L':'N','R':'N','e':'N','j':'N',
    'A':'S','a':'S','J':'S','S':'S',
    'V':'V','E':'V','!':'V',
    'F':'F',
    '/':'Q','f':'Q','Q':'Q',
}

TEST_RECORDS = [
    '100','103','105','111','113','117','121','123',
    '200','202','210','212','213','214','219','221',
    '222','228','231','232','233','234'
]

BEFORE, AFTER, FS = 90, 110, 360
STYLE = {
    'figure.facecolor': 'white',
    'axes.facecolor':   '#f8fafc',
    'axes.grid':        True,
    'grid.color':       '#e2e8f0',
    'grid.linewidth':   0.8,
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'font.family':      'DejaVu Sans',
}
plt.rcParams.update(STYLE)

# ── Helpers ───────────────────────────────────────────────────────────────────
def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved → {path}")

# ── Load models ───────────────────────────────────────────────────────────────
print("Loading models...")
model = tf.keras.models.load_model(MODEL_PATH)
with open(CLF_PATH, 'rb') as f:
    _arr = pickle.load(f)
arr_gbm     = _arr.get('gbm')
arr_rf      = _arr.get('rf')
arr_clf     = _arr.get('model')
arr_encoder = _arr['encoder']
print("  Models loaded")

# ── Extract test beats ────────────────────────────────────────────────────────
print("\nExtracting test beats from DS2...")
X_test, y_test, beat_signals = [], [], []

for rec in TEST_RECORDS:
    try:
        signal, _ = wfdb.rdsamp(DATA_PATH + rec)
        ann       = wfdb.rdann(DATA_PATH + rec, 'atr')
    except Exception as e:
        print(f"  Skipping {rec}: {e}")
        continue
    ecg = signal[:, 0].astype(np.float64)
    for peak, sym in zip(ann.sample, ann.symbol):
        cls = AAMI_MAP.get(sym)
        if cls is None:
            continue
        start, end = peak - BEFORE, peak + AFTER
        if start < 0 or end > len(ecg):
            continue
        beat = ecg[start:end].astype(np.float32)
        std  = beat.std()
        if std < 1e-6:
            continue
        beat = (beat - beat.mean()) / (std + 1e-8)
        X_test.append(beat)
        y_test.append(cls)
        beat_signals.append(beat)

X_test = np.array(X_test, dtype=np.float32)[..., np.newaxis]
y_test = np.array(y_test)
print(f"  {len(y_test)} test beats | Distribution: {dict(Counter(y_test))}")

# ── Stage 1 predictions ───────────────────────────────────────────────────────
print("\nRunning Stage 1 (Beat CNN) predictions...")
probs_all = model.predict(X_test, batch_size=512, verbose=1)
# LabelEncoder.fit(['N','S','V','F','Q']) sorts alphabetically: F=0,N=1,Q=2,S=3,V=4
MODEL_IDX_TO_CLASS = ['F', 'N', 'Q', 'S', 'V']
y_pred = np.array([MODEL_IDX_TO_CLASS[np.argmax(p)] for p in probs_all])
y_conf = probs_all.max(axis=1)
# Reorder prob columns to CLASSES=[N,S,V,F,Q] order for ROC
col_map = {cls: MODEL_IDX_TO_CLASS.index(cls) for cls in CLASSES}
probs_reordered = np.column_stack([probs_all[:, col_map[c]] for c in CLASSES])

# ── Stage 1 Metrics ───────────────────────────────────────────────────────────
acc  = accuracy_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)
prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)

print(f"\n{'='*55}")
print(f"  STAGE 1 — BEAT CLASSIFIER (CNN)")
print(f"{'='*55}")
print(f"  Accuracy  : {acc*100:.2f}%")
print(f"  Precision : {prec*100:.2f}%")
print(f"  Recall    : {rec*100:.2f}%")
print(f"  F1 Score  : {f1*100:.2f}%")
print(f"\n{classification_report(y_test, y_pred, labels=CLASSES, target_names=[CLASS_NAMES[c] for c in CLASSES], zero_division=0)}")

# ─────────────────────────────────────────────────────────────────────────────
# GRAPH 1 — Confusion Matrix
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating graphs...")
cm = confusion_matrix(y_test, y_pred, labels=CLASSES)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Stage 1 — Beat Classifier Confusion Matrix', fontsize=14, fontweight='bold', y=1.02)

for ax, data, title, fmt in zip(
    axes,
    [cm, cm_norm],
    ['Raw Counts', 'Normalized (Recall)'],
    ['d', '.2f']
):
    sns.heatmap(data, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES,
                ax=ax, linewidths=0.5, linecolor='#e2e8f0',
                cbar_kws={'shrink': 0.8})
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    ax.set_xlabel('Predicted', fontsize=10)
    ax.set_ylabel('True', fontsize=10)
    ax.tick_params(labelsize=10)

plt.tight_layout()
save(fig, '01_confusion_matrix.png')

# ─────────────────────────────────────────────────────────────────────────────
# GRAPH 2 — Per-class Precision / Recall / F1
# ─────────────────────────────────────────────────────────────────────────────
report = classification_report(y_test, y_pred, labels=CLASSES,
                                target_names=CLASSES,
                                output_dict=True, zero_division=0)
metrics_per_class = {
    cls: {
        'Precision': report[cls]['precision'],
        'Recall':    report[cls]['recall'],
        'F1':        report[cls]['f1-score'],
    }
    for cls in CLASSES if cls in report
}

x     = np.arange(len(CLASSES))
width = 0.25
fig, ax = plt.subplots(figsize=(12, 6))
colors  = ['#3b82f6', '#22c55e', '#f59e0b']
labels  = ['Precision', 'Recall', 'F1']

for i, (metric, color) in enumerate(zip(labels, colors)):
    vals = [metrics_per_class.get(c, {}).get(metric, 0) for c in CLASSES]
    bars = ax.bar(x + i*width, vals, width, label=metric, color=color, alpha=0.85)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{v:.2f}', ha='center', va='bottom', fontsize=8.5, fontweight='600')

ax.set_xticks(x + width)
ax.set_xticklabels([f'{c}\n{CLASS_NAMES[c]}' for c in CLASSES], fontsize=10)
ax.set_ylim(0, 1.12)
ax.set_ylabel('Score', fontsize=11)
ax.set_title('Stage 1 — Per-Class Precision / Recall / F1', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.axhline(0.9, color='#ef4444', linestyle='--', linewidth=1, alpha=0.5, label='0.90 threshold')
plt.tight_layout()
save(fig, '02_per_class_metrics.png')

# ─────────────────────────────────────────────────────────────────────────────
# GRAPH 3 — ROC Curves (one-vs-rest)
# ─────────────────────────────────────────────────────────────────────────────
y_bin = label_binarize(y_test, classes=CLASSES)
fig, ax = plt.subplots(figsize=(9, 7))

for i, cls in enumerate(CLASSES):
    if y_bin[:, i].sum() == 0:
        continue
    fpr, tpr, _ = roc_curve(y_bin[:, i], probs_reordered[:, i])
    roc_auc     = auc(fpr, tpr)
    ax.plot(fpr, tpr, linewidth=2, color=CLASS_COLORS[cls],
            label=f'{cls} — {CLASS_NAMES[cls]} (AUC = {roc_auc:.3f})')

ax.plot([0,1],[0,1], 'k--', linewidth=1, alpha=0.4, label='Random (AUC = 0.500)')
ax.set_xlabel('False Positive Rate', fontsize=11)
ax.set_ylabel('True Positive Rate', fontsize=11)
ax.set_title('Stage 1 — ROC Curves (One-vs-Rest)', fontsize=13, fontweight='bold')
ax.legend(fontsize=9, loc='lower right')
ax.set_xlim([-0.01, 1.01])
ax.set_ylim([-0.01, 1.05])
plt.tight_layout()
save(fig, '03_roc_curves.png')

# ─────────────────────────────────────────────────────────────────────────────
# GRAPH 4 — Confidence Distribution
# ─────────────────────────────────────────────────────────────────────────────
correct   = y_pred == y_test
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: histogram of confidence for correct vs wrong
axes[0].hist(y_conf[correct],  bins=40, alpha=0.7, color='#22c55e', label='Correct')
axes[0].hist(y_conf[~correct], bins=40, alpha=0.7, color='#ef4444', label='Incorrect')
axes[0].set_xlabel('Confidence Score', fontsize=11)
axes[0].set_ylabel('Count', fontsize=11)
axes[0].set_title('Prediction Confidence Distribution', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=10)

# Right: accuracy vs confidence threshold
thresholds = np.linspace(0.3, 0.99, 50)
accs, coverages = [], []
for t in thresholds:
    mask = y_conf >= t
    if mask.sum() == 0:
        break
    accs.append(accuracy_score(y_test[mask], y_pred[mask]))
    coverages.append(mask.mean())

ax2 = axes[1].twinx()
axes[1].plot(thresholds[:len(accs)], [a*100 for a in accs],
             color='#3b82f6', linewidth=2.5, label='Accuracy (%)')
ax2.plot(thresholds[:len(coverages)], [c*100 for c in coverages],
         color='#f59e0b', linewidth=2, linestyle='--', label='Coverage (%)')
axes[1].set_xlabel('Confidence Threshold', fontsize=11)
axes[1].set_ylabel('Accuracy (%)', fontsize=11, color='#3b82f6')
ax2.set_ylabel('Coverage (%)', fontsize=11, color='#f59e0b')
axes[1].set_title('Accuracy vs Confidence Threshold', fontsize=12, fontweight='bold')
lines1, labels1 = axes[1].get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
axes[1].legend(lines1+lines2, labels1+labels2, fontsize=9)

plt.tight_layout()
save(fig, '04_confidence_distribution.png')

# ─────────────────────────────────────────────────────────────────────────────
# GRAPH 5 — Beat Distribution (Test Set)
# ─────────────────────────────────────────────────────────────────────────────
counts_true = Counter(y_test)
counts_pred = Counter(y_pred)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, counts, title, alpha in zip(
    axes,
    [counts_true, counts_pred],
    ['True Label Distribution', 'Predicted Label Distribution'],
    [0.85, 0.85]
):
    vals  = [counts.get(c, 0) for c in CLASSES]
    bars  = ax.bar(CLASSES, vals,
                   color=[CLASS_COLORS[c] for c in CLASSES], alpha=alpha)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{v:,}', ha='center', va='bottom', fontsize=9, fontweight='600')
    ax.set_xlabel('Beat Class', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xticklabels([f'{c}\n{CLASS_NAMES[c]}' for c in CLASSES], fontsize=9)

plt.tight_layout()
save(fig, '05_beat_distribution.png')

# ─────────────────────────────────────────────────────────────────────────────
# GRAPH 6 — Sample Beat Waveforms per Class
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, len(CLASSES), figsize=(18, 4))
fig.suptitle('Stage 1 — Representative Beat Waveforms per Class',
             fontsize=13, fontweight='bold')
x_ms = np.linspace(-250, 305, 200)

for ax, cls in zip(axes, CLASSES):
    idxs = np.where(y_test == cls)[0]
    if len(idxs) == 0:
        ax.set_title(f'{cls}\n(no samples)')
        continue
    # Plot up to 30 beats lightly, then mean
    sample_idxs = idxs[:min(30, len(idxs))]
    beats = beat_signals[:len(y_test)]
    for i in sample_idxs:
        ax.plot(x_ms, beats[i], color=CLASS_COLORS[cls], alpha=0.2, linewidth=0.8)
    mean_beat = np.mean([beats[i] for i in idxs[:200]], axis=0)
    ax.plot(x_ms, mean_beat, color=CLASS_COLORS[cls], linewidth=2.5, label='Mean')
    ax.axvline(0, color='#94a3b8', linewidth=0.8, linestyle='--')
    ax.set_title(f'{cls} — {CLASS_NAMES[cls]}\n({len(idxs):,} beats)',
                 fontsize=9, fontweight='bold')
    ax.set_xlabel('ms', fontsize=8)
    if ax == axes[0]:
        ax.set_ylabel('Amplitude (σ)', fontsize=9)
    ax.tick_params(labelsize=7)

plt.tight_layout()
save(fig, '06_beat_waveforms.png')

# ─────────────────────────────────────────────────────────────────────────────
# GRAPH 7 — Summary Dashboard
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
fig.suptitle('CardioAI — Model Evaluation Summary', fontsize=16,
             fontweight='bold', y=0.98)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# Top-left: Overall metrics bar
ax1 = fig.add_subplot(gs[0, 0])
metric_names  = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
metric_values = [acc, prec, rec, f1]
metric_colors = ['#3b82f6', '#22c55e', '#f59e0b', '#8b5cf6']
bars = ax1.barh(metric_names, [v*100 for v in metric_values],
                color=metric_colors, alpha=0.85)
for bar, v in zip(bars, metric_values):
    ax1.text(v*100 + 0.3, bar.get_y() + bar.get_height()/2,
             f'{v*100:.1f}%', va='center', fontsize=10, fontweight='700')
ax1.set_xlim(0, 108)
ax1.set_xlabel('Score (%)', fontsize=10)
ax1.set_title('Overall Metrics', fontsize=11, fontweight='bold')
ax1.tick_params(labelsize=10)

# Top-middle: Normalized confusion matrix (small)
ax2 = fig.add_subplot(gs[0, 1])
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=CLASSES, yticklabels=CLASSES,
            ax=ax2, linewidths=0.5, cbar=False, annot_kws={'size': 9})
ax2.set_title('Confusion Matrix\n(Normalized)', fontsize=11, fontweight='bold')
ax2.set_xlabel('Predicted', fontsize=9)
ax2.set_ylabel('True', fontsize=9)
ax2.tick_params(labelsize=9)

# Top-right: Per-class F1
ax3 = fig.add_subplot(gs[0, 2])
f1_vals = [report.get(c, {}).get('f1-score', 0) for c in CLASSES]
bars = ax3.bar(CLASSES, f1_vals,
               color=[CLASS_COLORS[c] for c in CLASSES], alpha=0.85)
for bar, v in zip(bars, f1_vals):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{v:.2f}', ha='center', va='bottom', fontsize=9, fontweight='600')
ax3.set_ylim(0, 1.12)
ax3.set_ylabel('F1 Score', fontsize=10)
ax3.set_title('Per-Class F1 Score', fontsize=11, fontweight='bold')
ax3.axhline(0.9, color='#ef4444', linestyle='--', linewidth=1, alpha=0.6)
ax3.tick_params(labelsize=9)

# Bottom-left: True distribution pie
ax4 = fig.add_subplot(gs[1, 0])
pie_vals   = [counts_true.get(c, 0) for c in CLASSES]
pie_colors = [CLASS_COLORS[c] for c in CLASSES]
wedges, texts, autotexts = ax4.pie(
    pie_vals, labels=CLASSES, colors=pie_colors,
    autopct='%1.1f%%', startangle=90,
    wedgeprops=dict(width=0.55), pctdistance=0.75)
for at in autotexts:
    at.set_fontsize(8)
ax4.set_title('Test Set Distribution', fontsize=11, fontweight='bold')

# Bottom-middle: ROC (compact)
ax5 = fig.add_subplot(gs[1, 1])
for i, cls in enumerate(CLASSES):
    if y_bin[:, i].sum() == 0:
        continue
    fpr, tpr, _ = roc_curve(y_bin[:, i], probs_reordered[:, i])
    roc_auc     = auc(fpr, tpr)
    ax5.plot(fpr, tpr, linewidth=2, color=CLASS_COLORS[cls],
             label=f'{cls} ({roc_auc:.2f})')
ax5.plot([0,1],[0,1],'k--',linewidth=1,alpha=0.4)
ax5.set_xlabel('FPR', fontsize=9)
ax5.set_ylabel('TPR', fontsize=9)
ax5.set_title('ROC Curves', fontsize=11, fontweight='bold')
ax5.legend(fontsize=8, loc='lower right')
ax5.tick_params(labelsize=8)

# Bottom-right: Confidence histogram
ax6 = fig.add_subplot(gs[1, 2])
ax6.hist(y_conf[correct],  bins=30, alpha=0.75, color='#22c55e', label='Correct')
ax6.hist(y_conf[~correct], bins=30, alpha=0.75, color='#ef4444', label='Incorrect')
ax6.set_xlabel('Confidence', fontsize=10)
ax6.set_ylabel('Count', fontsize=10)
ax6.set_title('Confidence Distribution', fontsize=11, fontweight='bold')
ax6.legend(fontsize=9)
ax6.tick_params(labelsize=9)

save(fig, '07_summary_dashboard.png')

# ─────────────────────────────────────────────────────────────────────────────
# GRAPH 8 — Training History (if available)
# ─────────────────────────────────────────────────────────────────────────────
history_path = 'training_history.pkl'
if os.path.exists(history_path):
    with open(history_path, 'rb') as f:
        history = pickle.load(f)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Stage 1 — Training History', fontsize=13, fontweight='bold')

    for ax, metric, title in zip(
        axes,
        [('accuracy','val_accuracy'), ('loss','val_loss')],
        ['Accuracy', 'Loss']
    ):
        train_key, val_key = metric
        if train_key in history:
            ax.plot(history[train_key], color='#3b82f6', linewidth=2, label='Train')
        if val_key in history:
            ax.plot(history[val_key], color='#f59e0b', linewidth=2, label='Validation')
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(f'{title} over Epochs', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)

    plt.tight_layout()
    save(fig, '08_training_history.png')
else:
    print("  (training_history.pkl not found — skipping training history graph)")

# ─────────────────────────────────────────────────────────────────────────────
# Print final summary
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  EVALUATION COMPLETE")
print(f"{'='*55}")
print(f"  Test beats evaluated : {len(y_test):,}")
print(f"  Overall Accuracy     : {acc*100:.2f}%")
print(f"  Weighted Precision   : {prec*100:.2f}%")
print(f"  Weighted Recall      : {rec*100:.2f}%")
print(f"  Weighted F1          : {f1*100:.2f}%")
print(f"\n  Per-class F1:")
for cls in CLASSES:
    if cls in report:
        print(f"    {cls} ({CLASS_NAMES[cls]:20}) : {report[cls]['f1-score']*100:.1f}%  "
              f"(support: {int(report[cls]['support']):,})")
print(f"\n  Graphs saved to: {OUT_DIR}/")
print(f"    01_confusion_matrix.png")
print(f"    02_per_class_metrics.png")
print(f"    03_roc_curves.png")
print(f"    04_confidence_distribution.png")
print(f"    05_beat_distribution.png")
print(f"    06_beat_waveforms.png")
print(f"    07_summary_dashboard.png")
