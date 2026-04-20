"""
HOG Pedestrian Detection - Dual Condition Evaluation
Trains on ORIGINAL data, tests on BOTH original and challenging conditions.

Pipeline:
  Input → Preprocessing (Gamma/CLAHE/Bilateral) → Gradient computation →
  9-bin orientation histogram (8×8 cells) → L2-Hys block normalization (16×16) →
  Linear SVM classification

Methods compared:
  1. Baseline HOG (Dalal & Triggs, CVPR 2005)
  2. CLAHE + HOG
  3. Bilateral Filter + HOG
  4. Scharr HOG
  5. Combined (CLAHE + Bilateral + Scharr + LBP)

BITS F311 Image Processing Project
"""

import os
import time
import numpy as np
import pickle
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, roc_curve, auc)
from sklearn.preprocessing import StandardScaler

from utils import load_challenging_dataset, METHODS


def train_and_evaluate_dual(method_name, extract_fn, 
                            train_pos, train_neg,
                            test_orig_pos, test_orig_neg,
                            test_chal_pos, test_chal_neg,
                            results_dir="results"):
    """
    Train on original data, evaluate on BOTH original and challenging test sets.
    Returns a dict of metrics for both conditions.
    """
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"  Method: {method_name}")
    print(f"{'='*60}")
    
    # ---- Feature Extraction (Training) ----
    print("\n[1/4] Extracting training features (original data)...")
    t0 = time.time()
    X_train_pos = extract_fn(train_pos)
    X_train_neg = extract_fn(train_neg)
    train_time = time.time() - t0
    
    X_train = np.vstack([X_train_pos, X_train_neg])
    y_train = np.hstack([np.ones(len(X_train_pos)), np.zeros(len(X_train_neg))])
    
    print(f"       Feature dim: {X_train.shape[1]}")
    print(f"       Pos: {len(X_train_pos)}, Neg: {len(X_train_neg)}")
    print(f"       Time: {train_time:.1f}s")
    
    # ---- Feature Extraction (Test - Original) ----
    print("\n[2/4] Extracting ORIGINAL test features...")
    t0 = time.time()
    X_test_orig_pos = extract_fn(test_orig_pos)
    X_test_orig_neg = extract_fn(test_orig_neg)
    test_orig_time = time.time() - t0
    
    X_test_orig = np.vstack([X_test_orig_pos, X_test_orig_neg])
    y_test_orig = np.hstack([np.ones(len(X_test_orig_pos)), np.zeros(len(X_test_orig_neg))])
    print(f"       Pos: {len(X_test_orig_pos)}, Neg: {len(X_test_orig_neg)}")
    print(f"       Time: {test_orig_time:.1f}s")
    
    # ---- Feature Extraction (Test - Challenging) ----
    print("\n[3/4] Extracting CHALLENGING test features...")
    t0 = time.time()
    X_test_chal_pos = extract_fn(test_chal_pos)
    X_test_chal_neg = extract_fn(test_chal_neg)
    test_chal_time = time.time() - t0
    
    X_test_chal = np.vstack([X_test_chal_pos, X_test_chal_neg])
    y_test_chal = np.hstack([np.ones(len(X_test_chal_pos)), np.zeros(len(X_test_chal_neg))])
    print(f"       Pos: {len(X_test_chal_pos)}, Neg: {len(X_test_chal_neg)}")
    print(f"       Time: {test_chal_time:.1f}s")
    
    # ---- Feature Scaling ----
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test_orig = scaler.transform(X_test_orig)
    X_test_chal = scaler.transform(X_test_chal)
    
    # ---- Train Linear SVM ----
    print("\n[4/4] Training Linear SVM...")
    t0 = time.time()
    svm = LinearSVC(C=0.01, max_iter=10000, random_state=42)
    svm.fit(X_train, y_train)
    svm_time = time.time() - t0
    print(f"       Training time: {svm_time:.1f}s")
    
    # ---- Evaluate on ORIGINAL test set ----
    y_pred_orig = svm.predict(X_test_orig)
    scores_orig = svm.decision_function(X_test_orig)
    
    acc_orig = accuracy_score(y_test_orig, y_pred_orig)
    prec_orig = precision_score(y_test_orig, y_pred_orig)
    rec_orig = recall_score(y_test_orig, y_pred_orig)
    f1_orig = f1_score(y_test_orig, y_pred_orig)
    fpr_orig, tpr_orig, _ = roc_curve(y_test_orig, scores_orig)
    auc_orig = auc(fpr_orig, tpr_orig)
    miss_rate_orig = 1 - tpr_orig
    
    # ---- Evaluate on CHALLENGING test set ----
    y_pred_chal = svm.predict(X_test_chal)
    scores_chal = svm.decision_function(X_test_chal)
    
    acc_chal = accuracy_score(y_test_chal, y_pred_chal)
    prec_chal = precision_score(y_test_chal, y_pred_chal)
    rec_chal = recall_score(y_test_chal, y_pred_chal)
    f1_chal = f1_score(y_test_chal, y_pred_chal)
    fpr_chal, tpr_chal, _ = roc_curve(y_test_chal, scores_chal)
    auc_chal = auc(fpr_chal, tpr_chal)
    miss_rate_chal = 1 - tpr_chal
    
    total_images = len(train_pos) + len(train_neg) + len(test_orig_pos) + len(test_orig_neg)
    avg_time = (train_time + test_orig_time) / total_images
    
    results = {
        "method": method_name,
        # Original condition results
        "accuracy": acc_orig,
        "precision": prec_orig,
        "recall": rec_orig,
        "f1": f1_orig,
        "auc": auc_orig,
        "fpr": fpr_orig,
        "tpr": tpr_orig,
        "miss_rate": miss_rate_orig,
        # Challenging condition results
        "accuracy_chal": acc_chal,
        "precision_chal": prec_chal,
        "recall_chal": rec_chal,
        "f1_chal": f1_chal,
        "auc_chal": auc_chal,
        "fpr_chal": fpr_chal,
        "tpr_chal": tpr_chal,
        "miss_rate_chal": miss_rate_chal,
        # Timing
        "feature_dim": X_train.shape[1],
        "train_extract_time": train_time,
        "test_extract_time": test_orig_time,
        "svm_train_time": svm_time,
        "avg_time_per_image": avg_time,
    }
    
    # Print results
    print(f"\n  Results for {method_name}:")
    print(f"  {'Metric':<15} {'Original':>10} {'Challenging':>12} {'Drop':>8}")
    print(f"  {'-'*47}")
    print(f"  {'Accuracy':<15} {acc_orig:>10.4f} {acc_chal:>12.4f} {(acc_chal-acc_orig):>+8.4f}")
    print(f"  {'Precision':<15} {prec_orig:>10.4f} {prec_chal:>12.4f} {(prec_chal-prec_orig):>+8.4f}")
    print(f"  {'Recall':<15} {rec_orig:>10.4f} {rec_chal:>12.4f} {(rec_chal-rec_orig):>+8.4f}")
    print(f"  {'F1 Score':<15} {f1_orig:>10.4f} {f1_chal:>12.4f} {(f1_chal-f1_orig):>+8.4f}")
    print(f"  {'AUC':<15} {auc_orig:>10.4f} {auc_chal:>12.4f} {(auc_chal-auc_orig):>+8.4f}")
    print(f"  {'Feature Dim':<15} {results['feature_dim']:>10d}")
    print(f"  {'Avg Time/Img':<15} {results['avg_time_per_image']*1000:>10.1f}ms")
    
    # Save results
    safe_name = method_name.replace(" ", "_").replace("+", "").replace("(", "").replace(")", "")
    with open(os.path.join(results_dir, f"{safe_name}_results.pkl"), "wb") as f:
        pickle.dump(results, f)
    
    return results


def main():
    """Run all methods with dual-condition evaluation."""
    print("=" * 60)
    print("  BITS F311: HOG Pedestrian Detection")
    print("  Dual Condition: Original vs Challenging")
    print("  Paper: Dalal & Triggs, CVPR 2005")
    print("=" * 60)
    
    # Load challenging dataset
    print("\nLoading Challenging Dataset...")
    print("(Pre-generated patches: original + degraded)")
    dataset = load_challenging_dataset("data/dataset")
    
    train_pos = dataset["train_orig_pos"]
    train_neg = dataset["train_orig_neg"]
    test_orig_pos = dataset["test_orig_pos"]
    test_orig_neg = dataset["test_orig_neg"]
    test_chal_pos = dataset["test_chal_pos"]
    test_chal_neg = dataset["test_chal_neg"]
    
    print(f"\nDataset Summary:")
    print(f"  Training (original only): {len(train_pos)} pos, {len(train_neg)} neg")
    print(f"  Test (original):          {len(test_orig_pos)} pos, {len(test_orig_neg)} neg")
    print(f"  Test (challenging):       {len(test_chal_pos)} pos, {len(test_chal_neg)} neg")
    
    # Run all methods
    all_results = {}
    for method_name, extract_fn in METHODS.items():
        results = train_and_evaluate_dual(
            method_name, extract_fn,
            train_pos, train_neg,
            test_orig_pos, test_orig_neg,
            test_chal_pos, test_chal_neg
        )
        all_results[method_name] = results
    
    # Save combined results
    os.makedirs("results", exist_ok=True)
    with open("results/all_results.pkl", "wb") as f:
        pickle.dump(all_results, f)
    
    # Print final comparison table
    print("\n\n" + "=" * 100)
    print("  FINAL COMPARISON TABLE: Original vs Challenging")
    print("=" * 100)
    print(f"{'Method':<40} {'Orig F1':>8} {'Chal F1':>8} {'Drop':>8} "
          f"{'Orig AUC':>9} {'Chal AUC':>9} {'ms/img':>7}")
    print("-" * 100)
    for name, r in all_results.items():
        f1_drop = r['f1_chal'] - r['f1']
        print(f"{name:<40} {r['f1']:>8.4f} {r['f1_chal']:>8.4f} {f1_drop:>+8.4f} "
              f"{r['auc']:>9.4f} {r['auc_chal']:>9.4f} "
              f"{r['avg_time_per_image']*1000:>7.1f}")
    print("=" * 100)
    
    # Highlight key findings
    baseline = all_results.get("Baseline HOG", {})
    combined_key = "Combined (CLAHE+Bilateral+Scharr+LBP)"
    combined = all_results.get(combined_key, {})
    
    if baseline and combined:
        baseline_drop = baseline.get('f1_chal', 0) - baseline.get('f1', 0)
        combined_drop = combined.get('f1_chal', 0) - combined.get('f1', 0)
        print(f"\n  KEY FINDINGS:")
        print(f"  Baseline HOG F1 drop on challenging data: {baseline_drop:+.4f}")
        print(f"  Combined method F1 drop on challenging:   {combined_drop:+.4f}")
        print(f"  Combined is {abs(baseline_drop) - abs(combined_drop):.4f} more robust!")
    
    print("\nResults saved to results/ directory.")
    print("Run compare_results.py to generate comparison plots.")


if __name__ == "__main__":
    main()
