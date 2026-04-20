"""
Baseline HOG Pedestrian Detection
Replicates the original Dalal & Triggs (CVPR 2005) pipeline.

Pipeline:
  Input → Gamma Normalization → Gradient (simple filter) → 
  9-bin orientation histogram (8×8 cells) → L2-Hys block normalization (16×16) →
  Linear SVM classification

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

from utils import load_dataset, METHODS


def train_and_evaluate(method_name, extract_fn, train_pos, train_neg,
                       test_pos, test_neg, results_dir="results"):
    """
    Train a Linear SVM on features extracted by extract_fn and evaluate.
    Returns a dict of metrics.
    """
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"  Method: {method_name}")
    print(f"{'='*60}")
    
    # ---- Feature Extraction ----
    print("\n[1/3] Extracting training features...")
    t0 = time.time()
    X_train_pos = extract_fn(train_pos)
    X_train_neg = extract_fn(train_neg)
    train_time = time.time() - t0
    
    X_train = np.vstack([X_train_pos, X_train_neg])
    y_train = np.hstack([np.ones(len(X_train_pos)), np.zeros(len(X_train_neg))])
    
    print(f"       Feature dim: {X_train.shape[1]}")
    print(f"       Pos: {len(X_train_pos)}, Neg: {len(X_train_neg)}")
    print(f"       Time: {train_time:.1f}s")
    
    print("\n[2/3] Extracting test features...")
    t0 = time.time()
    X_test_pos = extract_fn(test_pos)
    X_test_neg = extract_fn(test_neg)
    test_extract_time = time.time() - t0
    
    X_test = np.vstack([X_test_pos, X_test_neg])
    y_test = np.hstack([np.ones(len(X_test_pos)), np.zeros(len(X_test_neg))])
    
    print(f"       Pos: {len(X_test_pos)}, Neg: {len(X_test_neg)}")
    print(f"       Time: {test_extract_time:.1f}s")
    
    # ---- Feature Scaling ----
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # ---- Train Linear SVM (as in the paper) ----
    print("\n[3/3] Training Linear SVM...")
    t0 = time.time()
    svm = LinearSVC(C=0.01, max_iter=10000, random_state=42)
    svm.fit(X_train, y_train)
    svm_time = time.time() - t0
    print(f"       Training time: {svm_time:.1f}s")
    
    # ---- Evaluate ----
    y_pred = svm.predict(X_test)
    decision_scores = svm.decision_function(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, decision_scores)
    roc_auc = auc(fpr, tpr)
    
    # DET curve (Miss Rate vs FPPW)
    miss_rate = 1 - tpr
    
    results = {
        "method": method_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": roc_auc,
        "fpr": fpr,
        "tpr": tpr,
        "miss_rate": miss_rate,
        "feature_dim": X_train.shape[1],
        "train_extract_time": train_time,
        "test_extract_time": test_extract_time,
        "svm_train_time": svm_time,
        "avg_time_per_image": (train_time + test_extract_time) / (len(train_pos) + len(train_neg) + len(test_pos) + len(test_neg)),
    }
    
    print(f"\n  Results for {method_name}:")
    print(f"    Accuracy:  {acc:.4f}")
    print(f"    Precision: {prec:.4f}")
    print(f"    Recall:    {rec:.4f}")
    print(f"    F1 Score:  {f1:.4f}")
    print(f"    AUC:       {roc_auc:.4f}")
    print(f"    Feature Dim: {results['feature_dim']}")
    print(f"    Avg Time/Image: {results['avg_time_per_image']*1000:.1f}ms")
    
    # Save results
    safe_name = method_name.replace(" ", "_").replace("+", "").replace("(", "").replace(")", "")
    with open(os.path.join(results_dir, f"{safe_name}_results.pkl"), "wb") as f:
        pickle.dump(results, f)
    
    return results


def main():
    """Run all methods and save results."""
    print("=" * 60)
    print("  BITS F311: HOG Pedestrian Detection - All Methods")
    print("  Paper: Dalal & Triggs, CVPR 2005")
    print("=" * 60)
    
    # Load dataset
    print("\nLoading INRIA Person Dataset...")
    train_pos, train_neg, test_pos, test_neg = load_dataset("data")
    
    print(f"\nDataset Summary:")
    print(f"  Training: {len(train_pos)} positive, {len(train_neg)} negative")
    print(f"  Testing:  {len(test_pos)} positive, {len(test_neg)} negative")
    
    # Run all methods
    all_results = {}
    for method_name, extract_fn in METHODS.items():
        results = train_and_evaluate(
            method_name, extract_fn,
            train_pos, train_neg, test_pos, test_neg
        )
        all_results[method_name] = results
    
    # Save combined results
    with open("results/all_results.pkl", "wb") as f:
        pickle.dump(all_results, f)
    
    # Print summary table
    print("\n\n" + "=" * 80)
    print("  FINAL COMPARISON TABLE")
    print("=" * 80)
    print(f"{'Method':<35} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7} {'ms/img':>7}")
    print("-" * 80)
    for name, r in all_results.items():
        print(f"{name:<35} {r['accuracy']:>7.4f} {r['precision']:>7.4f} "
              f"{r['recall']:>7.4f} {r['f1']:>7.4f} {r['auc']:>7.4f} "
              f"{r['avg_time_per_image']*1000:>7.1f}")
    print("=" * 80)
    
    print("\nResults saved to results/ directory.")
    print("Run compare_results.py to generate comparison plots.")


if __name__ == "__main__":
    main()
