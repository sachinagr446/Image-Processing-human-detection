"""
Compare Results and Generate Plots
Generates DET curves, ROC curves, bar charts, and sample visualizations.

BITS F311 Image Processing Project
"""

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatter
import cv2

from utils import (apply_clahe, gamma_correction, extract_hog_opencv,
                   extract_hog_scharr, extract_lbp_features)


# Color scheme for plots
COLORS = {
    "Baseline HOG": "#e74c3c",                    # Red
    "CLAHE + HOG": "#3498db",                      # Blue
    "Scharr HOG": "#2ecc71",                       # Green
    "Combined (CLAHE+Scharr+LBP)": "#9b59b6",     # Purple
}

MARKERS = {
    "Baseline HOG": "o",
    "CLAHE + HOG": "s",
    "Scharr HOG": "^",
    "Combined (CLAHE+Scharr+LBP)": "D",
}


def load_results(results_dir="results"):
    """Load all saved results."""
    results_file = os.path.join(results_dir, "all_results.pkl")
    if os.path.exists(results_file):
        with open(results_file, "rb") as f:
            return pickle.load(f)
    
    # Try loading individual files
    results = {}
    for fname in os.listdir(results_dir):
        if fname.endswith("_results.pkl") and fname != "all_results.pkl":
            with open(os.path.join(results_dir, fname), "rb") as f:
                r = pickle.load(f)
                results[r["method"]] = r
    return results


def plot_det_curves(all_results, save_path="results/det_curves.png"):
    """
    Plot Detection Error Tradeoff (DET) curves.
    This is the primary evaluation metric used in the paper.
    x-axis: False Positive Rate (FPPW), y-axis: Miss Rate
    Both on log scale.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for method, results in all_results.items():
        fpr = results["fpr"]
        miss_rate = results["miss_rate"]
        
        # Filter out zeros for log scale
        mask = (fpr > 0) & (miss_rate > 0)
        if np.sum(mask) > 1:
            ax.plot(fpr[mask], miss_rate[mask],
                    color=COLORS.get(method, "#333"),
                    linewidth=2.5,
                    label=f'{method} (AUC={results["auc"]:.4f})')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('False Positive Rate (FPPW)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Miss Rate', fontsize=14, fontweight='bold')
    ax.set_title('Detection Error Tradeoff (DET) Curves\nLower is Better', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim([1e-4, 1])
    ax.set_ylim([1e-3, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved DET curves to {save_path}")


def plot_roc_curves(all_results, save_path="results/roc_curves.png"):
    """Plot ROC curves for all methods."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for method, results in all_results.items():
        ax.plot(results["fpr"], results["tpr"],
                color=COLORS.get(method, "#333"),
                linewidth=2.5,
                label=f'{method} (AUC={results["auc"]:.4f})')
    
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
    ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    ax.set_title('ROC Curves Comparison', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved ROC curves to {save_path}")


def plot_metrics_comparison(all_results, save_path="results/metrics_comparison.png"):
    """Bar chart comparing accuracy, precision, recall, F1 across methods."""
    metrics = ["accuracy", "precision", "recall", "f1"]
    labels = ["Accuracy", "Precision", "Recall", "F1 Score"]
    methods = list(all_results.keys())
    
    x = np.arange(len(metrics))
    width = 0.18
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for i, method in enumerate(methods):
        values = [all_results[method][m] for m in metrics]
        offset = (i - len(methods)/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width,
                      label=method,
                      color=COLORS.get(method, "#333"),
                      edgecolor='white',
                      linewidth=0.5)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Metric', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score', fontsize=14, fontweight='bold')
    ax.set_title('Classification Metrics Comparison', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.legend(fontsize=11, loc='lower right')
    ax.set_ylim([0, 1.12])
    ax.grid(True, alpha=0.2, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved metrics comparison to {save_path}")


def plot_timing_comparison(all_results, save_path="results/timing_comparison.png"):
    """Bar chart comparing processing time per image."""
    methods = list(all_results.keys())
    times = [all_results[m]["avg_time_per_image"] * 1000 for m in methods]  # ms
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.barh(range(len(methods)), times,
                   color=[COLORS.get(m, "#333") for m in methods],
                   edgecolor='white', linewidth=0.5, height=0.6)
    
    for bar, t in zip(bars, times):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2.,
                f'{t:.1f} ms', ha='left', va='center', fontsize=12, fontweight='bold')
    
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=12)
    ax.set_xlabel('Average Time per Image (ms)', fontsize=14, fontweight='bold')
    ax.set_title('Processing Time Comparison', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.2, axis='x')
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved timing comparison to {save_path}")


def plot_feature_dim_comparison(all_results, save_path="results/feature_dimensions.png"):
    """Compare feature vector dimensionality."""
    methods = list(all_results.keys())
    dims = [all_results[m]["feature_dim"] for m in methods]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(range(len(methods)), dims,
                  color=[COLORS.get(m, "#333") for m in methods],
                  edgecolor='white', linewidth=0.5, width=0.6)
    
    for bar, d in zip(bars, dims):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 50,
                str(d), ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=10, rotation=15, ha='right')
    ax.set_ylabel('Feature Vector Dimension', fontsize=14, fontweight='bold')
    ax.set_title('Feature Vector Dimensionality Comparison', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.2, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved feature dimensions to {save_path}")


def plot_preprocessing_samples(save_path="results/preprocessing_samples.png"):
    """
    Visualize preprocessing effects on sample images.
    Shows: Original, Gamma Corrected, CLAHE, Scharr Gradients.
    """
    # Create a synthetic sample or use a test image
    from utils import find_dataset_root
    
    try:
        root, fmt = find_dataset_root("data")
        import glob
        if fmt == "voc":
            pos_dir = os.path.join(root, "Test", "JPEGImages")
        else:
            pos_dir = os.path.join(root, "Test", "pos")
        images = sorted(glob.glob(os.path.join(pos_dir, "*.*")))[:3]
        
        if not images:
            print("  No sample images found, skipping preprocessing visualization.")
            return
        
        fig, axes = plt.subplots(3, 5, figsize=(20, 12))
        titles = ["Original", "Gamma (√)", "CLAHE", "Simple Gradient", "Scharr Gradient"]
        
        for row, img_path in enumerate(images):
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (64, 128))
            
            # Original
            axes[row, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            # Gamma corrected
            gamma_img = gamma_correction(img)
            axes[row, 1].imshow(cv2.cvtColor(gamma_img, cv2.COLOR_BGR2RGB))
            
            # CLAHE
            clahe_img = apply_clahe(img)
            axes[row, 2].imshow(cv2.cvtColor(clahe_img, cv2.COLOR_BGR2RGB))
            
            # Simple gradient magnitude
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gx_s = cv2.filter2D(gray, cv2.CV_64F, np.array([[-1, 0, 1]]))
            gy_s = cv2.filter2D(gray, cv2.CV_64F, np.array([[-1], [0], [1]]))
            mag_simple = np.sqrt(gx_s**2 + gy_s**2)
            axes[row, 3].imshow(mag_simple, cmap='hot')
            
            # Scharr gradient magnitude
            gx_sch = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
            gy_sch = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
            mag_scharr = np.sqrt(gx_sch**2 + gy_sch**2)
            axes[row, 4].imshow(mag_scharr, cmap='hot')
        
        for j, title in enumerate(titles):
            axes[0, j].set_title(title, fontsize=14, fontweight='bold')
        
        for ax in axes.flat:
            ax.axis('off')
        
        fig.suptitle('Preprocessing Comparison on Sample Images', fontsize=18, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  Saved preprocessing samples to {save_path}")
        
    except FileNotFoundError:
        print("  Dataset not found, skipping preprocessing visualization.")


def generate_summary_table(all_results, save_path="results/summary_table.txt"):
    """Generate a text summary table."""
    with open(save_path, "w") as f:
        f.write("=" * 90 + "\n")
        f.write("  HOG Pedestrian Detection - Results Summary\n")
        f.write("  Paper: Dalal & Triggs, CVPR 2005\n")
        f.write("  Dataset: INRIA Person\n")
        f.write("=" * 90 + "\n\n")
        
        f.write(f"{'Method':<35} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7} {'Dim':>6} {'ms/img':>7}\n")
        f.write("-" * 90 + "\n")
        
        for name, r in all_results.items():
            f.write(f"{name:<35} {r['accuracy']:>7.4f} {r['precision']:>7.4f} "
                    f"{r['recall']:>7.4f} {r['f1']:>7.4f} {r['auc']:>7.4f} "
                    f"{r['feature_dim']:>6d} {r['avg_time_per_image']*1000:>7.1f}\n")
        
        f.write("-" * 90 + "\n")
        
        # Find best method
        best = max(all_results.items(), key=lambda x: x[1]["f1"])
        baseline = all_results.get("Baseline HOG", {})
        f.write(f"\nBest method: {best[0]}\n")
        if baseline:
            improvement = best[1]["f1"] - baseline["f1"]
            f.write(f"F1 improvement over baseline: +{improvement:.4f} ({improvement/baseline['f1']*100:.1f}%)\n")
    
    print(f"  Saved summary table to {save_path}")


def main():
    """Generate all comparison plots."""
    print("=" * 60)
    print("  Generating Comparison Plots")
    print("=" * 60)
    
    os.makedirs("results", exist_ok=True)
    
    all_results = load_results()
    
    if not all_results:
        print("\nNo results found! Run baseline_hog.py first.")
        return
    
    print(f"\nFound results for {len(all_results)} methods:")
    for name in all_results:
        print(f"  - {name}")
    
    print("\nGenerating plots...")
    plot_det_curves(all_results)
    plot_roc_curves(all_results)
    plot_metrics_comparison(all_results)
    plot_timing_comparison(all_results)
    plot_feature_dim_comparison(all_results)
    plot_preprocessing_samples()
    generate_summary_table(all_results)
    
    print("\n" + "=" * 60)
    print("  All plots generated successfully!")
    print("  Check the results/ directory.")
    print("=" * 60)


if __name__ == "__main__":
    main()
