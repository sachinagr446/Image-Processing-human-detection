"""
Challenging Dataset Generator
Takes INRIA Person images and creates both original and challenging versions.

Challenging conditions:
  1. Low light (darkening)
  2. Gaussian noise
  3. Gaussian blur
  4. All combined

Output structure:
  data/dataset/
  ├── original_pos/           # Clean positive patches (train)
  ├── original_neg/           # Clean negative patches (train)
  ├── challenging_pos/        # Degraded positive patches (train)
  ├── challenging_neg/        # Degraded negative patches (train)
  ├── test_original_pos/      # Clean positive patches (test)
  ├── test_original_neg/      # Clean negative patches (test)
  ├── test_challenging_pos/   # Degraded positive patches (test)
  └── test_challenging_neg/   # Degraded negative patches (test)

BITS F311 Image Processing Project
"""

import os
import sys
import numpy as np
import cv2
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (find_dataset_root, crop_positive_patches,
                   sample_negative_patches_voc, sample_negative_patches,
                   load_images_from_dir, WINDOW_SIZE)


# ============================================================
#  Degradation Functions
# ============================================================

def darken(img):
    """Simulate low-light / poor illumination conditions."""
    return cv2.convertScaleAbs(img, alpha=0.5, beta=-30)


def brighten(img):
    """Simulate over-exposure / bright conditions."""
    return cv2.convertScaleAbs(img, alpha=1.5, beta=30)


def add_noise(img):
    """Add Gaussian noise to simulate sensor noise."""
    noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
    return cv2.add(img, noise)


def blur(img):
    """Apply Gaussian blur to simulate motion blur / defocus."""
    return cv2.GaussianBlur(img, (5, 5), 0)


def reduce_contrast(img):
    """Reduce contrast to simulate hazy/foggy conditions."""
    return cv2.convertScaleAbs(img, alpha=0.6, beta=40)


def create_challenging(img):
    """
    Apply combined degradations to create a challenging image.
    Pipeline: darken → add noise → blur
    This simulates a realistic worst-case scenario:
    low light + sensor noise + slight defocus.
    """
    img = darken(img)
    img = add_noise(img)
    img = blur(img)
    return img


# ============================================================
#  Dataset Creation
# ============================================================

def save_patches(patches, output_dir, prefix="img"):
    """Save a list of image patches to a directory."""
    os.makedirs(output_dir, exist_ok=True)
    for i, patch in enumerate(tqdm(patches, desc=f"Saving to {os.path.basename(output_dir)}", leave=False)):
        filename = f"{prefix}_{i:05d}.png"
        cv2.imwrite(os.path.join(output_dir, filename), patch)
    print(f"  Saved {len(patches)} images to {output_dir}")


def create_challenging_patches(patches):
    """Apply challenging degradations to all patches."""
    challenging = []
    for patch in tqdm(patches, desc="Applying degradations", leave=False):
        challenging.append(create_challenging(patch))
    return challenging


def main():
    """Generate the challenging dataset."""
    print("=" * 60)
    print("  Challenging Dataset Generator")
    print("  INRIA Person Dataset → Original + Challenging")
    print("=" * 60)

    # Output directory
    dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "dataset")
    os.makedirs(dataset_dir, exist_ok=True)

    # Find original dataset
    print("\nLocating original INRIA dataset...")
    root, fmt = find_dataset_root("data")
    print(f"  Format: {fmt}")
    print(f"  Root: {root}")

    # ---- Extract patches from original images ----
    if fmt == "voc":
        train_img_dir = os.path.join(root, "Train", "JPEGImages")
        train_ann_dir = os.path.join(root, "Train", "Annotations")
        test_img_dir = os.path.join(root, "Test", "JPEGImages")
        test_ann_dir = os.path.join(root, "Test", "Annotations")

        print("\n[1/4] Cropping positive training patches...")
        train_pos = crop_positive_patches(train_img_dir, train_ann_dir)
        print(f"       Got {len(train_pos)} positive training patches")

        print("[2/4] Sampling negative training patches...")
        train_neg = sample_negative_patches_voc(train_img_dir, train_ann_dir)
        print(f"       Got {len(train_neg)} negative training patches")

        print("[3/4] Cropping positive test patches...")
        test_pos = crop_positive_patches(test_img_dir, test_ann_dir)
        print(f"       Got {len(test_pos)} positive test patches")

        print("[4/4] Sampling negative test patches...")
        test_neg = sample_negative_patches_voc(test_img_dir, test_ann_dir)
        print(f"       Got {len(test_neg)} negative test patches")

    else:
        train_pos_dir = os.path.join(root, "Train", "pos")
        train_neg_dir = os.path.join(root, "Train", "neg")
        test_pos_dir = os.path.join(root, "Test", "pos")
        test_neg_dir = os.path.join(root, "Test", "neg")

        print("\n[1/4] Loading positive training images...")
        train_pos = load_images_from_dir(train_pos_dir)
        print(f"       Got {len(train_pos)} positive training images")

        print("[2/4] Sampling negative training patches...")
        train_neg = sample_negative_patches(train_neg_dir)
        print(f"       Got {len(train_neg)} negative training patches")

        print("[3/4] Loading positive test images...")
        test_pos = load_images_from_dir(test_pos_dir)
        print(f"       Got {len(test_pos)} positive test images")

        print("[4/4] Sampling negative test patches...")
        test_neg = sample_negative_patches(test_neg_dir)
        print(f"       Got {len(test_neg)} negative test patches")

    # ---- Create challenging versions ----
    print("\n" + "=" * 60)
    print("  Creating Challenging Versions")
    print("=" * 60)

    print("\nApplying degradations to training positives...")
    train_pos_chal = create_challenging_patches(train_pos)

    print("Applying degradations to training negatives...")
    train_neg_chal = create_challenging_patches(train_neg)

    print("Applying degradations to test positives...")
    test_pos_chal = create_challenging_patches(test_pos)

    print("Applying degradations to test negatives...")
    test_neg_chal = create_challenging_patches(test_neg)

    # ---- Save all patches ----
    print("\n" + "=" * 60)
    print("  Saving Dataset")
    print("=" * 60)

    # Training sets
    print("\nSaving training patches...")
    save_patches(train_pos, os.path.join(dataset_dir, "original_pos"), "pos")
    save_patches(train_neg, os.path.join(dataset_dir, "original_neg"), "neg")
    save_patches(train_pos_chal, os.path.join(dataset_dir, "challenging_pos"), "pos")
    save_patches(train_neg_chal, os.path.join(dataset_dir, "challenging_neg"), "neg")

    # Test sets
    print("\nSaving test patches...")
    save_patches(test_pos, os.path.join(dataset_dir, "test_original_pos"), "pos")
    save_patches(test_neg, os.path.join(dataset_dir, "test_original_neg"), "neg")
    save_patches(test_pos_chal, os.path.join(dataset_dir, "test_challenging_pos"), "pos")
    save_patches(test_neg_chal, os.path.join(dataset_dir, "test_challenging_neg"), "neg")

    # ---- Save sample visualization ----
    print("\nGenerating sample visualization...")
    save_sample_visualization(train_pos[:5], train_pos_chal[:5], dataset_dir)

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("  Dataset Summary")
    print("=" * 60)
    print(f"\n  Training:")
    print(f"    Original  - Pos: {len(train_pos)}, Neg: {len(train_neg)}")
    print(f"    Challenging - Pos: {len(train_pos_chal)}, Neg: {len(train_neg_chal)}")
    print(f"\n  Testing:")
    print(f"    Original  - Pos: {len(test_pos)}, Neg: {len(test_neg)}")
    print(f"    Challenging - Pos: {len(test_pos_chal)}, Neg: {len(test_neg_chal)}")
    print(f"\n  Dataset saved to: {dataset_dir}")
    print("=" * 60)


def save_sample_visualization(original_samples, challenging_samples, output_dir):
    """Save a grid showing original vs challenging samples."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_samples = min(len(original_samples), len(challenging_samples), 5)
    if n_samples == 0:
        return

    fig, axes = plt.subplots(2, n_samples, figsize=(3 * n_samples, 7))

    for i in range(n_samples):
        # Original
        axes[0, i].imshow(cv2.cvtColor(original_samples[i], cv2.COLOR_BGR2RGB))
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('Original', fontsize=14, fontweight='bold')

        # Challenging
        axes[1, i].imshow(cv2.cvtColor(challenging_samples[i], cv2.COLOR_BGR2RGB))
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel('Challenging', fontsize=14, fontweight='bold')

    axes[0, n_samples // 2].set_title('Original vs Challenging Samples',
                                       fontsize=16, fontweight='bold')

    plt.suptitle('Dataset: Original vs Challenging (Dark + Noise + Blur)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sample_comparison.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved sample visualization to {output_dir}/sample_comparison.png")


if __name__ == "__main__":
    main()
