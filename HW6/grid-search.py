#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 20:31:18 2026

@author: dcastro
"""

"""
    - Hyperparameter Grid Search Optimization for MNIST MLP
    - Optimizes n_iter, d_hidden, and eta to achieve >95% accuracy
    - Automatically trains the best setup and saves performance plots
"""

import os
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import metrics as skMetric

# Custom Multi-Layer Perceptron
from src.NeuralNetMLP import NeuralNetMLP as MLP


# CONFIGURATION
SEED = 1234

DATA_DIR = "data"
FILE_IN = "mnist_scaled.npz"
OUTPUT_DIR = "fig"

VALID_SIZE = 0.05
MINIBATCH_SIZE = 100
L2_REG = 0.1

os.makedirs(OUTPUT_DIR, exist_ok=True)

np.random.seed(SEED)


# LOAD DATA

print("=" * 70)
print("Loading MNIST data...")
print("=" * 70)

data_path = os.path.join(DATA_DIR, FILE_IN)

with np.load(data_path) as data:
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]

print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples:     {X_test.shape[0]}")
print(f"Input dimensions: {X_train.shape[1]}")


# TRAIN / VALIDATION SPLIT
X_train_split, X_valid_split, y_train_split, y_valid_split = train_test_split(
    X_train,
    y_train,
    test_size=VALID_SIZE,
    shuffle=True,
    stratify=y_train,
    random_state=SEED
)

print("\nData split:")
print(f"Training subset:   {X_train_split.shape[0]}")
print(f"Validation subset: {X_valid_split.shape[0]}")


# HYPERPARAMETER GRID SEARCH

print("\n" + "=" * 70)
print("Starting Hyperparameter Optimization")
print("=" * 70)

search_space = {
    "n_iter": [150, 200],
    "d_hidden": [100, 150],
    "eta": [0.001, 0.002],
}

# Generate all parameter combinations
param_combinations = list(
    itertools.product(
        search_space["n_iter"],
        search_space["d_hidden"],
        search_space["eta"]
    )
)

print(f"\nTotal configurations to test: {len(param_combinations)}")

best_valid_acc = -np.inf
best_params = None
grid_results = []

for config_number, (n_iter, d_hidden, eta) in enumerate(
    param_combinations, start=1
):

    print("\n" + "-" * 70)
    print(f"Configuration {config_number}/{len(param_combinations)}")
    print(
        f"n_iter={n_iter}, "
        f"d_hidden={d_hidden}, "
        f"eta={eta}"
    )

    model = MLP(
        d_hidden=d_hidden,
        L2=L2_REG,
        n_iter=n_iter,
        eta=eta,
        shuffle=True,
        minibatch_size=MINIBATCH_SIZE,
        seed=SEED
    )

    model.fit(
        X_train=X_train_split,
        y_train=y_train_split,
        X_valid=X_valid_split,
        y_valid=y_valid_split
    )

    # Use the final validation accuracy
    valid_acc = float(model.dEval["acc_valid"][-1])

    print(f"Validation accuracy: {valid_acc * 100:.2f}%")

    grid_results.append({
        "n_iter": n_iter,
        "d_hidden": d_hidden,
        "eta": eta,
        "valid_accuracy": valid_acc * 100
    })

    if valid_acc > best_valid_acc:
        best_valid_acc = valid_acc

        best_params = {
            "n_iter": n_iter,
            "d_hidden": d_hidden,
            "eta": eta
        }


# GRID SEARCH RESULTS

results_df = pd.DataFrame(grid_results)

results_df = results_df.sort_values(
    by="valid_accuracy",
    ascending=False
).reset_index(drop=True)

print("\n" + "=" * 70)
print("GRID SEARCH RESULTS")
print("=" * 70)

print(results_df.to_string(index=False))

print("\n" + "=" * 70)
print("BEST CONFIGURATION")
print("=" * 70)

print(f"n_iter:     {best_params['n_iter']}")
print(f"d_hidden:   {best_params['d_hidden']}")
print(f"eta:        {best_params['eta']}")
print(f"Validation: {best_valid_acc * 100:.2f}%")
print("=" * 70)


# Save grid-search results
results_df.to_csv(
    os.path.join(OUTPUT_DIR, "grid_search_results.csv"),
    index=False
)


# FINAL MODEL

print("\nTraining final model on the complete training dataset...")

nn_clf = MLP(
    d_hidden=best_params["d_hidden"],
    L2=L2_REG,
    n_iter=best_params["n_iter"],
    eta=best_params["eta"],
    shuffle=True,
    minibatch_size=MINIBATCH_SIZE,
    seed=SEED
)


nn_clf.fit(
    X_train=X_train,
    y_train=y_train,
    X_valid=X_valid_split,
    y_valid=y_valid_split
)


# TRAINING PERFORMANCE

y_train_pred = nn_clf.predict(X_train)

train_accuracy = (
    skMetric.accuracy_score(y_train, y_train_pred) * 100
)

print(
    f"\nFinal Training Accuracy: "
    f"{train_accuracy:.2f}%"
)


# TEST PERFORMANCE

y_test_pred = nn_clf.predict(X_test)

test_accuracy = (
    skMetric.accuracy_score(y_test, y_test_pred) * 100
)

test_precision = (
    skMetric.precision_score(
        y_test,
        y_test_pred,
        average="macro",
        zero_division=0
    ) * 100
)

test_recall = (
    skMetric.recall_score(
        y_test,
        y_test_pred,
        average="macro",
        zero_division=0
    ) * 100
)

test_f1 = (
    skMetric.f1_score(
        y_test,
        y_test_pred,
        average="macro",
        zero_division=0
    ) * 100
)

print("\n" + "=" * 70)
print("FINAL TEST PERFORMANCE")
print("=" * 70)

print(f"Accuracy:  {test_accuracy:.2f}%")
print(f"Precision: {test_precision:.2f}%")
print(f"Recall:    {test_recall:.2f}%")
print(f"F1-score:  {test_f1:.2f}%")

if test_accuracy >= 95.0:
    print("Target >95% accuracy: ACHIEVED")
else:
    print("Target >95% accuracy: NOT ACHIEVED")

print("=" * 70)




# CONVERGENCE PLOTS

cost = np.asarray(nn_clf.dEval["cost"])
acc_train = np.asarray(nn_clf.dEval["acc_train"])
acc_valid = np.asarray(nn_clf.dEval["acc_valid"])

epochs_cost = np.arange(1, len(cost) + 1)
epochs_acc = np.arange(1, len(acc_train) + 1)

fig1, ax = plt.subplots(
    2,
    1,
    figsize=(9, 10)
)

# Cost
ax[0].plot(
    epochs_cost,
    cost * 1e-3,
    "ko-",
    ms=4,
    lw=2,
    mfc="none",
    label="Cost"
)

ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Cost × 10⁻³")
ax[0].set_title("Optimized Cost Curve")
ax[0].legend(loc="upper right")
ax[0].grid(alpha=0.3)

# Accuracy
ax[1].plot(
    epochs_acc,
    acc_train,
    "bo-",
    ms=4,
    lw=2,
    mfc="none",
    label="Training"
)

ax[1].plot(
    epochs_acc,
    acc_valid,
    "ro-",
    ms=4,
    lw=2,
    mfc="none",
    label="Validation"
)

ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Accuracy")
ax[1].set_title("Training and Validation Accuracy")
ax[1].legend(loc="lower right")
ax[1].grid(alpha=0.3)

plt.tight_layout()

performance_path = os.path.join(
    OUTPUT_DIR,
    "optimized_performance.png"
)

plt.savefig(
    performance_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close(fig1)


# CONFUSION MATRIX

fig2, ax_cm = plt.subplots(
    figsize=(9, 9)
)

skMetric.ConfusionMatrixDisplay.from_predictions(
    y_true=y_test,
    y_pred=y_test_pred,
    cmap="viridis",
    ax=ax_cm
)

ax_cm.set_title(
    "Optimized Confusion Matrix — Test Data\n"
    f"Accuracy: {test_accuracy:.2f}% | "
    f"Precision: {test_precision:.2f}% | "
    f"Recall: {test_recall:.2f}%",
    fontsize=13,
    pad=15
)

plt.tight_layout()

confusion_path = os.path.join(
    OUTPUT_DIR,
    "optimized_confusion_matrix.png"
)

plt.savefig(
    confusion_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close(fig2)

