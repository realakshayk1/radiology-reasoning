"""
metrics.py — Calibration and performance metrics for the RadReason project.
"""

import numpy as np
import torch
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt

def calculate_brier_score(targets, probs):
    """
    Calculate the Brier score for multi-label predictions.
    targets: (N, num_classes) numpy array
    probs: (N, num_classes) numpy array
    """
    num_classes = targets.shape[1]
    scores = []
    for i in range(num_classes):
        scores.append(brier_score_loss(targets[:, i], probs[:, i]))
    return np.mean(scores), scores

def get_reliability_data(targets, probs, n_bins=10):
    """
    Generate data for a reliability diagram.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    # Flatten across classes for a macro-level reliability diagram
    # or keep per class. Here we do macro.
    y_true = targets.flatten()
    y_prob = probs.flatten()
    
    accuracies = []
    confidences = []
    counts = []
    
    for lower, upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > lower) & (y_prob <= upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracies.append(np.mean(y_true[in_bin]))
            confidences.append(np.mean(y_prob[in_bin]))
            counts.append(np.sum(in_bin))
        else:
            accuracies.append(0.0)
            confidences.append((lower + upper) / 2)
            counts.append(0)
            
    return np.array(accuracies), np.array(confidences), np.array(counts)

def plot_reliability_diagram(targets, probs, title="Reliability Diagram", save_path=None):
    """
    Compute and plot a reliability diagram.
    """
    acc, conf, counts = get_reliability_data(targets, probs)
    ece = np.sum(np.abs(acc - conf) * counts) / np.sum(counts)
    
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    plt.plot(conf, acc, "s-", label=f"Model (ECE={ece:.4f})")
    
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend(loc="upper left")
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

class TemperatureScaler(torch.nn.Module):
    """
    A thin wrapper to learn the temperature T for temperature scaling.
    """
    def __init__(self):
        super().__init__()
        self.temperature = torch.nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature
