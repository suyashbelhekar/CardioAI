"""
ECG Arrhythmia Detection — Stage 1 Beat Classifier
====================================================
Data separation follows the AAMI/PhysioNet inter-patient protocol:
  DS1 (Train): 22 records — no patient overlap with test set
  DS2 (Test) : 22 records — completely held out, never seen during training

This prevents data leakage that would inflate accuracy when the same
patient appears in both train and test sets.
"""

import wfdb
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DATA_PATH = 'Dataset/mit-bih-arrhythmia-database-1.0.0/'

# Standard inter-patient split (DS1 / DS2) — no patient overlap
TRAIN_RECORDS = [
    '101', '106', '108', '109', '112', '114', '115', '116',
    '118', '119', '122', '124', '201', '203', '205', '207',
    '208', '209', '215', '220', '223', '230'
]
TEST_RECORDS = [
    '100', '103', '105', '111', '113', '117', '121', '123',
    '200', '202', '210', '212', '213', '214', '219', '221',
    '222', '228', '231', '232', '233', '234'
]

# AAMI 5-class mapping
AAMI_MAP = {
    'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',
    'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',
    'V': 'V', 'E': 'V', '!': 'V',
    'F': 'F',
    '/': 'Q', 'f': 'Q', 'Q': 'Q'
}

CLASSES    = ['N', 'S', 'V', 'F', 'Q']
BEFORE     = 90    # samples before R-peak
AFTER      = 110   # samples after  R-peak  → 200-sample window @ 360 Hz
FS         = 360
np.random.seed(42)
tf.random.set_seed(42)

# ─── AUGMENTATION ─────────────────────────────────────────────────────────────
def augment_beat(beat):
    aug = beat.copy()
    aug *= np.random.uniform(0.85, 1.15)
    aug += np.random.normal(0, 0.03, len(aug))
    t    = np.linspace(0, 1, len(aug))
    aug += np.random.uniform(-0.05, 0.05) * np.sin(
           2 * np.pi * np.random.uniform(0.5, 2) * t)
    shift = np.random.randint(-3, 4)
    if shift > 0:
        aug = np.concatenate([aug[shift:], np.zeros(shift)])
    elif shift < 0:
        aug = np.concatenate([np.zeros(-shift), aug[:shift]])
    return aug[:200].astype(np.float32)  # ensure fixed length

# ─── DATA LOADING ─────────────────────────────────────────────────────────────
def extract_beats(records, split_name=''):
    X, y = [], []
    for rec in records:
        try:
            signal, _ = wfdb.rdsamp(DATA_PATH + rec)
            ann       = wfdb.rdann(DATA_PATH + rec, 'atr')
        except Exception as e:
            print(f"  Skipping {rec}: {e}")
            continue
        ecg = signal[:, 0]
        for peak, sym in zip(ann.sample, ann.symbol):
            label = AAMI_MAP.get(sym)
            if label is None:
                continue
            s, e = peak - BEFORE, peak + AFTER
            if s < 0 or e > len(ecg):
                continue
            beat = ecg[s:e].astype(np.float32)
            std  = beat.std()
            if std < 1e-6:
                continue
            beat = (beat - beat.mean()) / (std + 1e-8)
            X.append(beat)
            y.append(label)
    print(f"  {split_name}: {len(X)} beats  {dict(sorted(Counter(y).items()))}")
    return np.array(X, dtype=np.float32), np.array(y)

print("Loading data (inter-patient split)...")
print("Train records (DS1):")
X_train_raw, y_train_raw = extract_beats(TRAIN_RECORDS, 'DS1')
print("Test records (DS2):")
X_test,  y_test  = extract_beats(TEST_RECORDS,  'DS2')

# ─── CLASS BALANCING ──────────────────────────────────────────────────────────
# Cap N at 3× second-largest class; oversample minorities with augmentation
def balance(X, y, augment=True):
    counts = Counter(y)
    sorted_vals = sorted(counts.values(), reverse=True)
    second_max  = sorted_vals[1] if len(sorted_vals) > 1 else sorted_vals[0]
    target_N    = min(counts.get('N', 0), second_max * 3)
    target_min  = max(second_max, 1500)

    X_out, y_out = [], []
    for cls in CLASSES:
        idx = np.where(y == cls)[0]
        if len(idx) == 0:
            continue
        Xc, yc = X[idx], y[idx]
        n       = len(idx)
        target  = target_N if cls == 'N' else target_min

        if n >= target:
            chosen = np.random.choice(n, target, replace=False)
            X_out.append(Xc[chosen]);  y_out.append(yc[chosen])
        else:
            X_out.append(Xc);  y_out.append(yc)
            needed  = target - n
            aug_idx = np.random.choice(n, needed, replace=True)
            Xaug    = np.array([augment_beat(Xc[i]) for i in aug_idx]) if augment else Xc[aug_idx]
            stds    = Xaug.std(axis=1, keepdims=True) + 1e-8
            Xaug    = (Xaug - Xaug.mean(axis=1, keepdims=True)) / stds
            X_out.append(Xaug.astype(np.float32))
            y_out.append(np.array([cls] * needed))

    Xb = np.vstack(X_out)
    yb = np.concatenate(y_out)
    # Shuffle
    idx = np.random.permutation(len(Xb))
    return Xb[idx], yb[idx]

X_train_bal, y_train_bal = balance(X_train_raw, y_train_raw, augment=True)
print(f"\nBalanced train: {X_train_bal.shape}")
print(f"Distribution:   {dict(sorted(Counter(y_train_bal).items()))}")

# ─── LABEL ENCODING ───────────────────────────────────────────────────────────
le = LabelEncoder()
le.fit(CLASSES)

y_train_enc = le.transform(y_train_bal)
y_test_enc  = le.transform(y_test)

# Validation set: 10% of ORIGINAL unbalanced train data (real distribution)
val_n   = int(0.10 * len(X_train_raw))
val_idx = np.random.choice(len(X_train_raw), val_n, replace=False)
X_val   = X_train_raw[val_idx][..., np.newaxis]
y_val   = tf.keras.utils.to_categorical(le.transform(y_train_raw[val_idx]), len(CLASSES))

# Class weights — penalise minority misclassifications more
counts_bal = Counter(y_train_bal)
total_bal  = sum(counts_bal.values())
class_weight = {
    le.transform([c])[0]: total_bal / (len(CLASSES) * counts_bal[c])
    for c in CLASSES if c in counts_bal
}
print(f"\nClass weights: { {le.classes_[k]: round(v, 2) for k, v in class_weight.items()} }")

y_train_oh = tf.keras.utils.to_categorical(y_train_enc, len(CLASSES))
y_test_oh  = tf.keras.utils.to_categorical(y_test_enc,  len(CLASSES))

X_train_cnn = X_train_bal[..., np.newaxis]
X_test_cnn  = X_test[..., np.newaxis]

# ─── MODEL ────────────────────────────────────────────────────────────────────
def build_model(input_shape, num_classes):
    inp = layers.Input(shape=input_shape)

    x = layers.Conv1D(32, 5, padding='same', activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv1D(64, 5, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv1D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Bidirectional(layers.LSTM(64, return_sequences=False))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

model = build_model(X_train_cnn.shape[1:], len(CLASSES))
model.summary()

# ─── TRAINING ─────────────────────────────────────────────────────────────────
cb_list = [
    callbacks.EarlyStopping(monitor='val_loss', patience=12,
                             restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                 patience=5, min_lr=1e-6, verbose=1),
    callbacks.ModelCheckpoint('best_ecg_model.keras', monitor='val_accuracy',
                               save_best_only=True, verbose=1),
]

print("\nTraining...")
history = model.fit(
    X_train_cnn, y_train_oh,
    validation_data=(X_val, y_val),
    epochs=60,
    batch_size=128,
    class_weight=class_weight,
    callbacks=cb_list,
    verbose=1
)

# ─── EVALUATION ───────────────────────────────────────────────────────────────
loss, acc = model.evaluate(X_test_cnn, y_test_oh, verbose=0)
print(f"\nTest Accuracy : {acc*100:.2f}%  |  Loss: {loss:.4f}")

y_pred = np.argmax(model.predict(X_test_cnn, verbose=0), axis=1)
print("\nClassification Report:")
print(classification_report(y_test_enc, y_pred, target_names=le.classes_))

cm = confusion_matrix(y_test_enc, y_pred)
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
ConfusionMatrixDisplay(cm, display_labels=le.classes_).plot(ax=axes[0], colorbar=False)
axes[0].set_title('Confusion Matrix — DS2 Test Set')
axes[1].plot(history.history['accuracy'],     label='Train Acc')
axes[1].plot(history.history['val_accuracy'], label='Val Acc')
axes[1].plot(history.history['loss'],         label='Train Loss', linestyle='--')
axes[1].plot(history.history['val_loss'],     label='Val Loss',   linestyle='--')
axes[1].set_title('Training History')
axes[1].set_xlabel('Epoch')
axes[1].legend()
plt.tight_layout()
plt.savefig('ecg_results.png', dpi=150)
plt.show()
print("Saved ecg_results.png  |  Best model: best_ecg_model.keras")
