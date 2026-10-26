#!/usr/bin/env python3
"""
Create visualizations for High Fog YOLO Training Results
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Paths
results_csv = "/mnt/Data/as19197/new_project/yolo_training/runs/wunet_yolo_beta0.02/results.csv"
output_dir = Path("/home/as19197/.gemini/antigravity/brain/521be7ca-380d-4b05-9a55-7a6ab4119914")
output_dir.mkdir(exist_ok=True, parents=True)

# Read results
df = pd.read_csv(results_csv)
df.columns = df.columns.str.strip()  # Remove whitespace from column names

print("Columns available:", df.columns.tolist())
print(f"\nTotal epochs: {len(df)}")

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('High Fog YOLO Training Results (Beta=0.02)', fontsize=16, fontweight='bold')

# Plot 1: Loss curves
ax1 = axes[0, 0]
ax1.plot(df['epoch'], df['train/box_loss'], label='Box Loss', linewidth=2)
ax1.plot(df['epoch'], df['train/cls_loss'], label='Class Loss', linewidth=2)
ax1.plot(df['epoch'], df['train/dfl_loss'], label='DFL Loss', linewidth=2)
ax1.set_xlabel('Epoch', fontsize=11)
ax1.set_ylabel('Loss', fontsize=11)
ax1.set_title('Training Losses', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: mAP metrics
ax2 = axes[0, 1]
ax2.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@50', linewidth=2, color='green')
ax2.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@50-95', linewidth=2, color='blue')
ax2.set_xlabel('Epoch', fontsize=11)
ax2.set_ylabel('mAP', fontsize=11)
ax2.set_title('Validation mAP', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 1])

# Plot 3: Precision & Recall
ax3 = axes[1, 0]
ax3.plot(df['epoch'], df['metrics/precision(B)'], label='Precision', linewidth=2, color='orange')
ax3.plot(df['epoch'], df['metrics/recall(B)'], label='Recall', linewidth=2, color='red')
ax3.set_xlabel('Epoch', fontsize=11)
ax3.set_ylabel('Score', fontsize=11)
ax3.set_title('Precision & Recall', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_ylim([0, 1])

# Plot 4: Summary metrics
ax4 = axes[1, 1]
final_metrics = {
    'mAP@50': df['metrics/mAP50(B)'].iloc[-1],
    'mAP@50-95': df['metrics/mAP50-95(B)'].iloc[-1],
    'Precision': df['metrics/precision(B)'].iloc[-1],
    'Recall': df['metrics/recall(B)'].iloc[-1]
}

bars = ax4.bar(final_metrics.keys(), final_metrics.values(), 
               color=['#2ecc71', '#3498db', '#f39c12', '#e74c3c'])
ax4.set_ylabel('Score', fontsize=11)
ax4.set_title('Final Metrics (Epoch 100)', fontsize=12, fontweight='bold')
ax4.set_ylim([0, 1])
ax4.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'high_fog_yolo_training_curves.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Saved training curves to {output_dir / 'high_fog_yolo_training_curves.png'}")

# Print summary
print("\n" + "="*70)
print("TRAINING SUMMARY")
print("="*70)
print(f"Best mAP@50: {df['metrics/mAP50(B)'].max():.4f} (Epoch {df['metrics/mAP50(B)'].idxmax() + 1})")
print(f"Best mAP@50-95: {df['metrics/mAP50-95(B)'].max():.4f} (Epoch {df['metrics/mAP50-95(B)'].idxmax() + 1})")
print(f"\nFinal Epoch ({len(df)}):")
print(f"  mAP@50: {final_metrics['mAP@50']:.4f}")
print(f"  mAP@50-95: {final_metrics['mAP@50-95']:.4f}")
print(f"  Precision: {final_metrics['Precision']:.4f}")
print(f"  Recall: {final_metrics['Recall']:.4f}")
print("="*70)

plt.close()
