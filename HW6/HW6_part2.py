#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 18:06:32 2026

@author: dcastro
"""

import matplotlib.pyplot as plt
import pandas as pd


OUTPUT="fig"
df_performance = pd.read_csv(f"{OUTPUT}/model_metrics_loop.csv")
#=======================================================================================
#               4. Final Performance Metrics Comparison Plot
# =======================================================================================
plt.figure(figsize=(12, 6))

# Extract metrics from your performance data frame
x_ticks = df_performance["val_split_pct"].astype(str) + "%"

# Plot each evaluation score across the changing configurations
plt.plot(df_performance["train_split_pct"], df_performance["test_accuracy"], 
         'o-', color='navy', linewidth=2.5, markersize=8, label='Test Accuracy')

plt.plot(df_performance["train_split_pct"], df_performance["precision_macro"], 
         's--', color='darkgreen', linewidth=2, markersize=7, label='Precision')

plt.plot(df_performance["train_split_pct"], df_performance["recall_macro"], 
         '*:', color='crimson', linewidth=2, markersize=7, label='Recall')

# Reverse the x-axis direction so it matches a reading timeline (from 5% training up to 95%)
plt.gca().invert_xaxis()

# Add contextual styling elements
plt.title("Overall Model Performance Across Decreassing Data Training", fontsize=14, pad=15)
plt.xlabel("Percentage of Data out of Training ", fontsize=12)
plt.ylabel("Performance (%)", fontsize=12)
plt.xticks(df_performance["train_split_pct"], x_ticks)
plt.ylim(min(df_performance["recall_macro"]) - 5, 102) # Dynamically scales the floor safely
plt.grid(True, linestyle=':', alpha=0.5)
plt.legend(loc='lower left', fontsize=11, frameon=True, shadow=True)


plt.tight_layout()
plt.savefig(f"{OUTPUT}/final_metrics_comparison.png", dpi=300)
plt.show()
