import numpy as np
import tensorflow as tf
import json
import os, glob
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from model import cnn

#Load preprocessed data

path = "/home/ilemleisher/data/"
pattern = os.path.join(path, "cont_I4_D*_T*_F*.npz")
files = sorted(glob.glob(pattern))
print(f"Found {len(files)} files")
for fp in files:
    with np.load(fp) as d:
        freqs_list = d['freqs_list']
        asd_list = d['asd_list']

t, n_bins, k = len(freqs_list), len(freqs_list[0]), 6

X = np.log10(asd_list + 1e-12).astype(np.float32)

y_win = np.random.randint(0, 2, size=(t,), dtype=np.int32)  #placeholder labels

# Create patches
X_patch = np.array([X[i:i+k] for i in range(t-k+1)])  # (N, k, F)
y_patch = np.array([int(np.any(y_win[i:i+k] == 1)) for i in range(t - k + 1)], dtype=np.int32)

# Split
N = len(X_patch)
n_train = int(0.7 * N)
n_val = int(0.15 * N)

X_train, y_train = X_patch[:n_train], y_patch[:n_train]
X_val, y_val = X_patch[n_train:n_train+n_val], y_patch[n_train:n_train+n_val]
X_test, y_test = X_patch[n_train+n_val:], y_patch[n_train+n_val:]

# Create unique training directory
training_dir = os.path.join("training", datetime.now().strftime("%Y%m%d_%H%M%S"))
os.makedirs(training_dir, exist_ok=True)

mu = X_train.mean(axis=(0,1), keepdims=True)     # (1,1,F,1)
sigma = X_train.std(axis=(0,1), keepdims=True) + 1e-8
X_train = (X_train - mu) / sigma
X_val   = (X_val - mu) / sigma
X_test  = (X_test - mu) / sigma

n_epochs = 40

model = cnn(k=k, n_bins=n_bins)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),                        #adapts LR, parameter is starting LR
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),       #uses numerically stable sigmoid + BCE formulation
    metrics=[
        tf.keras.metrics.BinaryAccuracy(threshold=0.0, name="acc"),  #logits threshold
        tf.keras.metrics.AUC(from_logits=True, name="auc"),          #area under ROC curve
    ],
)

model.summary()

classes = np.unique(y_train)
weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)  #assigns larger weight to rare classes
class_weight = {int(c): float(w) for c, w in zip(classes, weights)}

#Callbacks

callbacks = [
    tf.keras.callbacks.EarlyStopping(                             #stop if no improvement in patience epochs
        monitor="val_auc", mode="max", patience=8, restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(                                             #saves model when improves
        os.path.join(training_dir, "best_asd_cnn.keras"), monitor="val_auc", mode="max", save_best_only=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(                               #reduces LR by factor if no loss decrease
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
    )
]

#Train

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=n_epochs,
    batch_size=16,                   #model sees n patches and computes average loss, does one optimizer
    class_weight=class_weight,      
    callbacks=callbacks,
    verbose=1,                              #amount of training output printed per fit()
)

#Evaluation

best_model = tf.keras.models.load_model(os.path.join(training_dir, "best_asd_cnn.keras"))

val_results = best_model.evaluate(X_val, y_val, verbose=0)
test_results = best_model.evaluate(X_test, y_test, verbose=0)

val_metrics = dict(zip(best_model.metrics_names, val_results))
test_metrics = dict(zip(best_model.metrics_names, test_results))

print("Validation metrics:", val_metrics)
print("Test metrics:", test_metrics)

np.save(os.path.join(training_dir, "mu.npy"), mu)
np.save(os.path.join(training_dir, "sigma.npy"), sigma)

with open(os.path.join(training_dir, "history.json"), "w") as f:
    json.dump(history.history, f, indent=2)

summary = {
    "val": val_metrics,
    "test": test_metrics,
}
with open(os.path.join(training_dir, "metrics.json"), "w") as f:
    json.dump(summary, f, indent=2)

with open(os.path.join(training_dir, "threshold.json"), "w") as f:
    json.dump({"threshold": 0.5}, f, indent=2)
