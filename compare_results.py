"""
Compare Results and Generate Plots
Generates DET curves, ROC curves, bar charts, and dual-condition comparisons.

BITS F311 Image Processing Project
"""

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2

from utils import (apply_clahe, gamma_correction, apply_bilateral_filter,
                   extract_hog_opencv, extract_hog_scharr, extract_lbp_features)

# Color scheme for plots
COLORS = {
    "Baseline HOG": "#e74c3c",
    "CLAHE + HOG": "#3498db",
    "Bilateral + HOG": "#e67e22",
    "Scharr HOG": "#2ecc71",
    "Combined (CLAHE+Bilateral+Scharr+LBP)": "#9b59b6",
}

MARKERS = {
    "Baseline HOG": "o",
    "CLAHE + HOG": "s",
    "Bilateral + HOG": "p",
    "Scharr HOG": "^",
    "Combined (CLAHE+Bilateral+Scharr+LBP)": "D",
}


def load_results(results_dir="results"):
    results_file = os.path.join(results_dir, "all_results.pkl")
    if os.path.exists(results_file):
        with open(results_file, "rb") as f:
            return pickle.load(f)
    results = {}
    for fname in os.listdir(results_dir):
        if fname.endswith("_results.pkl") and fname != "all_results.pkl":
            with open(os.path.join(results_dir, fname), "rb") as f:
                r = pickle.load(f)
                results[r["method"]] = r
    return results


def plot_det_curves(all_results, save_path="results/det_curves.png"):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    for ax, suffix, title in [(axes[0], "", "Original"), (axes[1], "_chal", "Challenging")]:
        fpr_key = "fpr" + suffix
        mr_key = "miss_rate" + suffix
        auc_key = "auc" + suffix
        for method, results in all_results.items():
            if fpr_key not in results:
                continue
            fpr = results[fpr_key]
            miss_rate = results[mr_key]
            mask = (fpr > 0) & (miss_rate > 0)
            if np.sum(mask) > 1:
                ax.plot(fpr[mask], miss_rate[mask], color=COLORS.get(method, "#333"),
                        linewidth=2.5, label=f'{method} (AUC={results[auc_key]:.4f})')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('False Positive Rate', fontsize=13, fontweight='bold')
        ax.set_ylabel('Miss Rate', fontsize=13, fontweight='bold')
        ax.set_title(f'DET Curve - {title} Condition', fontsize=15, fontweight='bold')
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3, which='both')
        ax.set_xlim([1e-4, 1]); ax.set_ylim([1e-3, 1])
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight'); plt.close()
    print(f"  Saved DET curves to {save_path}")


def plot_roc_curves(all_results, save_path="results/roc_curves.png"):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    for ax, suffix, title in [(axes[0], "", "Original"), (axes[1], "_chal", "Challenging")]:
        fpr_key = "fpr" + suffix
        tpr_key = "tpr" + suffix
        auc_key = "auc" + suffix
        for method, results in all_results.items():
            if fpr_key not in results:
                continue
            ax.plot(results[fpr_key], results[tpr_key], color=COLORS.get(method, "#333"),
                    linewidth=2.5, label=f'{method} (AUC={results[auc_key]:.4f})')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.set_xlabel('False Positive Rate', fontsize=13, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=13, fontweight='bold')
        ax.set_title(f'ROC Curve - {title} Condition', fontsize=15, fontweight='bold')
        ax.legend(fontsize=9, loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight'); plt.close()
    print(f"  Saved ROC curves to {save_path}")


def plot_metrics_comparison(all_results, save_path="results/metrics_comparison.png"):
    metrics = ["accuracy", "precision", "recall", "f1"]
    labels = ["Accuracy", "Precision", "Recall", "F1 Score"]
    methods = list(all_results.keys())
    x = np.arange(len(metrics))
    width = 0.15
    fig, ax = plt.subplots(figsize=(14, 8))
    for i, method in enumerate(methods):
        values = [all_results[method][m] for m in metrics]
        offset = (i - len(methods)/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=method,
                      color=COLORS.get(method, "#333"), edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.set_xlabel('Metric', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score', fontsize=14, fontweight='bold')
    ax.set_title('Classification Metrics - Original Condition', fontsize=16, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=12)
    ax.legend(fontsize=9, loc='lower right')
    ax.set_ylim([0, 1.15]); ax.grid(True, alpha=0.2, axis='y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight'); plt.close()
    print(f"  Saved metrics comparison to {save_path}")


def plot_dual_condition_comparison(all_results, save_path="results/dual_condition_comparison.png"):
    """Side-by-side bar chart: Original vs Challenging F1 per method."""
    methods = list(all_results.keys())
    has_chal = any("f1_chal" in r for r in all_results.values())
    if not has_chal:
        print("  No challenging results found, skipping dual comparison.")
        return
    f1_orig = [all_results[m].get("f1", 0) for m in methods]
    f1_chal = [all_results[m].get("f1_chal", 0) for m in methods]
    x = np.arange(len(methods))
    width = 0.35
    fig, ax = plt.subplots(figsize=(14, 8))
    bars1 = ax.bar(x - width/2, f1_orig, width, label='Original', color='#2ecc71',
                   edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width/2, f1_chal, width, label='Challenging', color='#e74c3c',
                   edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars1, f1_orig):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.008,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar, val in zip(bars2, f1_chal):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.008,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylabel('F1 Score', fontsize=14, fontweight='bold')
    ax.set_title('F1 Score: Original vs Challenging Conditions', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    short = [m.replace("Combined (CLAHE+Bilateral+Scharr+LBP)", "Combined") for m in methods]
    ax.set_xticklabels(short, fontsize=10, rotation=10, ha='right')
    ax.legend(fontsize=12); ax.set_ylim([0, 1.15]); ax.grid(True, alpha=0.2, axis='y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight'); plt.close()
    print(f"  Saved dual condition comparison to {save_path}")


def plot_performance_drop(all_results, save_path="results/performance_drop.png"):
    """Show how much each method degrades under challenging conditions."""
    methods = list(all_results.keys())
    has_chal = any("f1_chal" in r for r in all_results.values())
    if not has_chal:
        return
    drops = []
    for m in methods:
        f1_o = all_results[m].get("f1", 0)
        f1_c = all_results[m].get("f1_chal", 0)
        drop_pct = ((f1_c - f1_o) / f1_o * 100) if f1_o > 0 else 0
        drops.append(drop_pct)
    fig, ax = plt.subplots(figsize=(12, 7))
    colors_bar = ['#e74c3c' if d < -10 else '#f39c12' if d < -5 else '#2ecc71' for d in drops]
    short = [m.replace("Combined (CLAHE+Bilateral+Scharr+LBP)", "Combined") for m in methods]
    bars = ax.barh(range(len(methods)), drops, color=colors_bar, edgecolor='white', height=0.6)
    for bar, d in zip(bars, drops):
        ax.text(bar.get_width() - 1 if d < 0 else bar.get_width() + 0.5,
                bar.get_y() + bar.get_height()/2., f'{d:.1f}%',
                ha='right' if d < 0 else 'left', va='center', fontsize=12, fontweight='bold')
    ax.set_yticks(range(len(methods))); ax.set_yticklabels(short, fontsize=11)
    ax.set_xlabel('F1 Score Change (%)', fontsize=14, fontweight='bold')
    ax.set_title('Performance Drop: Original → Challenging\n(Less negative = more robust)',
                 fontsize=15, fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.grid(True, alpha=0.2, axis='x'); ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight'); plt.close()
    print(f"  Saved performance drop chart to {save_path}")


def plot_timing_comparison(all_results, save_path="results/timing_comparison.png"):
    methods = list(all_results.keys())
    times = [all_results[m]["avg_time_per_image"] * 1000 for m in methods]
    fig, ax = plt.subplots(figsize=(10, 6))
    short = [m.replace("Combined (CLAHE+Bilateral+Scharr+LBP)", "Combined") for m in methods]
    bars = ax.barh(range(len(methods)), times,
                   color=[COLORS.get(m, "#333") for m in methods],
                   edgecolor='white', linewidth=0.5, height=0.6)
    for bar, t in zip(bars, times):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2.,
                f'{t:.1f} ms', ha='left', va='center', fontsize=12, fontweight='bold')
    ax.set_yticks(range(len(methods))); ax.set_yticklabels(short, fontsize=11)
    ax.set_xlabel('Average Time per Image (ms)', fontsize=14, fontweight='bold')
    ax.set_title('Processing Time Comparison', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.2, axis='x'); ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight'); plt.close()
    print(f"  Saved timing comparison to {save_path}")


def plot_feature_dim_comparison(all_results, save_path="results/feature_dimensions.png"):
    methods = list(all_results.keys())
    dims = [all_results[m]["feature_dim"] for m in methods]
    fig, ax = plt.subplots(figsize=(10, 6))
    short = [m.replace("Combined (CLAHE+Bilateral+Scharr+LBP)", "Combined") for m in methods]
    bars = ax.bar(range(len(methods)), dims,
                  color=[COLORS.get(m, "#333") for m in methods],
                  edgecolor='white', linewidth=0.5, width=0.6)
    for bar, d in zip(bars, dims):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 50,
                str(d), ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(short, fontsize=10, rotation=15, ha='right')
    ax.set_ylabel('Feature Vector Dimension', fontsize=14, fontweight='bold')
    ax.set_title('Feature Vector Dimensionality Comparison', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.2, axis='y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight'); plt.close()
    print(f"  Saved feature dimensions to {save_path}")


def plot_preprocessing_samples(save_path="results/preprocessing_samples.png"):
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
            print("  No sample images found, skipping."); return

        fig, axes = plt.subplots(3, 6, figsize=(24, 12))
        titles = ["Original", "Gamma (√)", "CLAHE", "Bilateral", "Simple Grad", "Scharr Grad"]
        for row, img_path in enumerate(images):
            img = cv2.imread(img_path)
            if img is None: continue
            img = cv2.resize(img, (64, 128))
            axes[row, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes[row, 1].imshow(cv2.cvtColor(gamma_correction(img), cv2.COLOR_BGR2RGB))
            axes[row, 2].imshow(cv2.cvtColor(apply_clahe(img), cv2.COLOR_BGR2RGB))
            axes[row, 3].imshow(cv2.cvtColor(apply_bilateral_filter(img), cv2.COLOR_BGR2RGB))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gx_s = cv2.filter2D(gray, cv2.CV_64F, np.array([[-1, 0, 1]]))
            gy_s = cv2.filter2D(gray, cv2.CV_64F, np.array([[-1], [0], [1]]))
            axes[row, 4].imshow(np.sqrt(gx_s**2 + gy_s**2), cmap='hot')
            gx_sch = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
            gy_sch = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
            axes[row, 5].imshow(np.sqrt(gx_sch**2 + gy_sch**2), cmap='hot')
        for j, title in enumerate(titles):
            axes[0, j].set_title(title, fontsize=14, fontweight='bold')
        for ax in axes.flat:
            ax.axis('off')
        fig.suptitle('Preprocessing Comparison (incl. Bilateral Filter)', fontsize=18,
                     fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight'); plt.close()
        print(f"  Saved preprocessing samples to {save_path}")
    except FileNotFoundError:
        print("  Dataset not found, skipping preprocessing visualization.")


def generate_summary_table(all_results, save_path="results/summary_table.txt"):
    has_chal = any("f1_chal" in r for r in all_results.values())
    with open(save_path, "w") as f:
        f.write("=" * 110 + "\n")
        f.write("  HOG Pedestrian Detection - Results Summary\n")
        f.write("  Paper: Dalal & Triggs, CVPR 2005\n")
        f.write("  Dataset: INRIA Person (Original + Challenging)\n")
        f.write("=" * 110 + "\n\n")
        if has_chal:
            f.write(f"{'Method':<42} {'Orig F1':>8} {'Chal F1':>8} {'Drop%':>7} "
                    f"{'Orig AUC':>9} {'Chal AUC':>9} {'Dim':>5} {'ms/img':>7}\n")
            f.write("-" * 110 + "\n")
            for name, r in all_results.items():
                f1_drop = ((r.get('f1_chal',0) - r['f1']) / r['f1'] * 100) if r['f1'] > 0 else 0
                f.write(f"{name:<42} {r['f1']:>8.4f} {r.get('f1_chal',0):>8.4f} "
                        f"{f1_drop:>+7.1f} {r['auc']:>9.4f} {r.get('auc_chal',0):>9.4f} "
                        f"{r['feature_dim']:>5d} {r['avg_time_per_image']*1000:>7.1f}\n")
        else:
            f.write(f"{'Method':<42} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7}\n")
            f.write("-" * 80 + "\n")
            for name, r in all_results.items():
                f.write(f"{name:<42} {r['accuracy']:>7.4f} {r['precision']:>7.4f} "
                        f"{r['recall']:>7.4f} {r['f1']:>7.4f} {r['auc']:>7.4f}\n")
        f.write("-" * 110 + "\n")
        best = max(all_results.items(), key=lambda x: x[1]["f1"])
        baseline = all_results.get("Baseline HOG", {})
        f.write(f"\nBest method (original): {best[0]}\n")
        if baseline:
            imp = best[1]["f1"] - baseline["f1"]
            f.write(f"F1 improvement over baseline: +{imp:.4f} ({imp/baseline['f1']*100:.1f}%)\n")
        if has_chal:
            best_chal = max(all_results.items(), key=lambda x: x[1].get("f1_chal", 0))
            f.write(f"Best method (challenging): {best_chal[0]}\n")
            if baseline and "f1_chal" in baseline:
                bl_drop = (baseline["f1_chal"] - baseline["f1"]) / baseline["f1"] * 100
                bc_drop = (best_chal[1].get("f1_chal",0) - best_chal[1]["f1"]) / best_chal[1]["f1"] * 100
                f.write(f"Baseline drop on challenging: {bl_drop:+.1f}%\n")
                f.write(f"Best method drop on challenging: {bc_drop:+.1f}%\n")
    print(f"  Saved summary table to {save_path}")


def main():
    print("=" * 60)
    print("  Generating Comparison Plots")
    print("=" * 60)
    os.makedirs("results", exist_ok=True)
    all_results = load_results()
    if not all_results:
        print("\nNo results found! Run baseline_hog.py first."); return
    print(f"\nFound results for {len(all_results)} methods:")
    for name in all_results:
        print(f"  - {name}")
    print("\nGenerating plots...")
    plot_det_curves(all_results)
    plot_roc_curves(all_results)
    plot_metrics_comparison(all_results)
    plot_dual_condition_comparison(all_results)
    plot_performance_drop(all_results)
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
