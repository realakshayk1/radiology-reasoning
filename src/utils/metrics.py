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

def get_reliability_data(targets, probs, n_bins=10, adaptive=False, min_bin_size=None):
    """
    Generate data for a reliability diagram.
    adaptive: If True, uses equal-mass bins (quantiles). If False, uses equal-width bins.
    min_bin_size: Minimum number of samples per bin (only for adaptive).
    """
    # Flatten across classes for a macro-level reliability diagram
    y_true = targets.flatten()
    y_prob = probs.flatten()
    
    if adaptive:
        # Equal-mass bins using quantiles
        quantiles = np.linspace(0, 1, n_bins + 1)
        bin_boundaries = np.unique(np.quantile(y_prob, quantiles))
        
        # Ensure min_bin_size by merging bins from right to left if needed
        if min_bin_size is not None:
            new_boundaries = [bin_boundaries[0]]
            i = 1
            while i < len(bin_boundaries):
                current_upper = bin_boundaries[i]
                # Count samples in the proposed bin
                count = np.sum((y_prob > new_boundaries[-1]) & (y_prob <= current_upper))
                
                # If it's the last boundary and still too small, merge with previous
                if i == len(bin_boundaries) - 1 and count < min_bin_size and len(new_boundaries) > 1:
                    new_boundaries[-1] = current_upper
                elif count >= min_bin_size or i == len(bin_boundaries) - 1:
                    new_boundaries.append(current_upper)
                else:
                    # Look ahead to see if we should merge
                    # We merge by skipping this boundary and checking next one in next iteration
                    pass
                i += 1
            bin_boundaries = np.array(new_boundaries)

        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
    else:
        # Equal-width bins
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
    
    accuracies = []
    confidences = []
    counts = []
    
    for lower, upper in zip(bin_lowers, bin_uppers):
        if lower == bin_lowers[0]:
            in_bin = (y_prob >= lower) & (y_prob <= upper)
        else:
            in_bin = (y_prob > lower) & (y_prob <= upper)
            
        if np.any(in_bin):
            accuracies.append(np.mean(y_true[in_bin]))
            confidences.append(np.mean(y_prob[in_bin]))
            counts.append(np.sum(in_bin))
        else:
            # For plotting continuity, but ECE should skip empty bins
            pass
            
    return np.array(accuracies), np.array(confidences), np.array(counts)

def calculate_ece(targets, probs, n_bins=10, adaptive=False, min_bin_size=None):
    """
    Calculate Expected Calibration Error (ECE).
    """
    acc, conf, counts = get_reliability_data(targets, probs, n_bins=n_bins, adaptive=adaptive, min_bin_size=min_bin_size)
    total_count = np.sum(counts)
    if total_count == 0:
        return 0.0
    ece = np.sum(np.abs(acc - conf) * counts) / total_count
    return ece

def plot_combined_reliability_diagram(targets, probs_dict, n_bins=10, adaptive=True, min_bin_size=None, title="Reliability Diagram", save_path=None):
    """
    Plot multiple reliability curves on the same axes.
    probs_dict: mapping of label -> probabilities array
    """
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfectly calibrated")
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(probs_dict)))
    
    for (label, probs), color in zip(probs_dict.items(), colors):
        acc, conf, counts = get_reliability_data(targets, probs, n_bins=n_bins, adaptive=adaptive, min_bin_size=min_bin_size)
        ece = calculate_ece(targets, probs, n_bins=n_bins, adaptive=adaptive, min_bin_size=min_bin_size)
        plt.plot(conf, acc, "s-", color=color, label=f"{label} (ECE={ece:.4f})")
    
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(title + (" (Adaptive Binning)" if adaptive else ""))
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

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
