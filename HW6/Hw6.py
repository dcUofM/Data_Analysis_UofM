#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 13:17:37 2026

@author: dcastro
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.metrics as skMetric
#from sklearn.neural_network import MLPClassifier
from src.NeuralNetMLP import NeuralNetMLP as MLP

np.random.seed(1234)

#=======================================================================================
#                   1. Download and Prep Data
#=======================================================================================
data_dir = 'data'
file_in  = 'mnist_scaled.npz'
nx, ny   = 28, 28

#--------hyper parameters---------------
n_iter   = 100
d_hidden = 100 # number of hidden units
L2_reg   = 0.1 #factor for regularization term
eta      = 0.001 # learning rate
#----------training and testing sizes---
n_minibatch = 100

OUTPUT="fig"
dData = np.load(f"{data_dir}/{file_in}")
X_train, y_train = dData['X_train'], dData['y_train']
X_test, y_test   = dData['X_test'], dData['y_test']


X_train, X_valid, y_train, y_valid = train_test_split( X_train, y_train,
                                                        shuffle=True,
                                                        train_size=0.10,
                                                        stratify=y_train)

print(f"Subsampled Train Split Size: {X_train.shape[0]} rows")


nn_clf = MLP( d_hidden=d_hidden, L2= L2_reg, n_iter=n_iter, 
               eta=eta, shuffle = True,
              minibatch_size=n_minibatch,
              seed=1)

# Train on the full original training dataset to reach max performance 
nn_clf.fit( X_train = X_train, y_train=y_train,
          X_valid = X_valid, y_valid=y_valid)

y_train_pred = nn_clf.predict( X_train)
fAcc_train = skMetric.accuracy_score( y_train_pred, y_train)*100


y_train_pred = nn_clf.predict(X_train)
y_test_pred = nn_clf.predict(X_test)

fAcc_train = skMetric.accuracy_score(y_train, y_train_pred) * 100
fAcc_test = skMetric.accuracy_score(y_test, y_test_pred) * 100

print('\n--- Optimized Model Results ---')
print(f"Training Accuracy: {fAcc_train:.2f}%")
print(f"Testing Accuracy: {fAcc_test:.2f}%")

#=======================================================================================
#            3. For Loop: Training vs. Validation Size Trade-off (10% Data)
#=======================================================================================

performance_records = []
fig1, ax = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Loop from 5% to 95% in 10% increments
for curr_valid_size in np.arange(5, 95 + 10, 10):
    val_fraction = curr_valid_size / 100.0
    
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, 
        test_size=val_fraction, 
        random_state=1, 
        stratify=y_train 
    )
    
    remaining_train_samples = X_tr.shape[0]
    batch_size = 100 if remaining_train_samples >= 100 else int(remaining_train_samples)
    
    if batch_size < 2:
        continue 

    nn_clf = MLP(d_hidden=d_hidden, L2=L2_reg, n_iter=n_iter, 
                 eta=eta, shuffle=True,
                 minibatch_size=batch_size, 
                 seed=1)
    
    nn_clf.fit(X_train=X_tr, y_train=y_tr,
               X_valid=X_val, y_valid=y_val)
    
    y_tr_pred   = nn_clf.predict(X_tr)    
    y_test_pred = nn_clf.predict(X_test)   
    
    fAcc_train = round(skMetric.accuracy_score(y_train, y_train_pred) * 100, 2)
    fAcc_test  = round(skMetric.accuracy_score(y_test, y_test_pred) * 100, 2)
    f_prec     = round(skMetric.precision_score(y_test, y_test_pred, average='macro') * 100, 2)
    f_recall   = round(skMetric.recall_score(y_test, y_test_pred, average='macro') * 100, 2)

    
    metrics_dict = {
    "val_split_pct": curr_valid_size,
    "train_split_pct": 100 - curr_valid_size,
    "train_accuracy": fAcc_train,
    "test_accuracy": fAcc_test,
    "precision_macro": f_prec,
    "recall_macro": f_recall
    }
    performance_records.append(metrics_dict)

    label_text = f"Val Split: {curr_valid_size}% (Train: {100-curr_valid_size}%)"
    ax[0].plot(np.arange(n_iter) + 1, 
               np.asarray(nn_clf.dEval['cost']) * 1e-3, 
               'o-', ms=4, lw=1.5, label=label_text)
    
    ax[1].plot(np.arange(n_iter) + 1, nn_clf.dEval['acc_valid'], 
               'o-', ms=4, lw=1.5, label=f"Val {curr_valid_size}%")

df_performance = pd.DataFrame(performance_records)
df_performance.to_csv(f"{OUTPUT}/model_metrics_loop.csv", index=False)

print("\n--- Summary Table Created ---")
print(df_performance.to_string(index=False))

# --- Format the Cost Subplot (ax[0]) ---
ax[0].set_title("Loss Curves Across Changing Validation Fractions (Using 10% Subsample Data)", fontsize=14)
ax[0].set_ylabel("Cost * 1E3", fontsize=12)
ax[0].grid(True, linestyle='--', alpha=0.6)
ax[0].legend(bbox_to_anchor=(1.02, 1), loc='upper left', title="Splits")

# --- Format the Accuracy Subplot (ax[1]) ---
ax[1].set_ylabel("Validation Accuracy", fontsize=12)
ax[1].set_xlabel("Epochs / Iterations", fontsize=12)
ax[1].grid(True, linestyle='--', alpha=0.6)
ax[1].legend(bbox_to_anchor=(1.02, 1), loc='upper left', title="Accuracy")

plt.tight_layout()
plt.savefig(f"{OUTPUT}/performance_for_loop.png", bbox_inches='tight')
plt.show()


