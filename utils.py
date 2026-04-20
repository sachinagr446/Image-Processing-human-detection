"""
Utility functions for HOG pedestrian detection project.
Handles data loading, preprocessing, and shared helper functions.
BITS F311 Image Processing Project - Improving HOG (Dalal & Triggs, CVPR 2005)
"""

import os
import glob
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from tqdm import tqdm

# ============================================================
#  Constants matching the original paper
# ============================================================
WINDOW_SIZE = (64, 128)          # Detection window (width, height)
CELL_SIZE = (8, 8)               # Cell size in pixels
BLOCK_SIZE = (16, 16)            # Block size in pixels (2x2 cells)
BLOCK_STRIDE = (8, 8)            # Block stride (50% overlap)
NUM_BINS = 9                     # Number of orientation bins
NEG_PATCHES_PER_IMAGE = 10       # Negative patches sampled per negative image


def find_dataset_root(base_path="data"):
    """
    Automatically find the INRIA dataset root directory.
    Supports both original format (pos/neg) and PASCAL-VOC format (JPEGImages/Annotations).
    """
    base_path = os.path.abspath(base_path)
    
    # Check for PASCAL-VOC format (Kaggle version)
    candidates = [
        base_path,
        os.path.join(base_path, "archive"),
        os.path.join(base_path, "INRIAPerson"),
    ]
    
    for candidate in candidates:
        train_imgs = os.path.join(candidate, "Train", "JPEGImages")
        test_imgs = os.path.join(candidate, "Test", "JPEGImages")
        train_anns = os.path.join(candidate, "Train", "Annotations")
        if os.path.isdir(train_imgs) and os.path.isdir(test_imgs):
            print(f"[INFO] Found dataset (PASCAL-VOC format): {candidate}")
            return candidate, "voc"
    
    # Check for original format (pos/neg directories)
    for candidate in candidates:
        train_pos = os.path.join(candidate, "Train", "pos")
        test_pos = os.path.join(candidate, "Test", "pos")
        if os.path.isdir(train_pos) and os.path.isdir(test_pos):
            print(f"[INFO] Found dataset (original format): {candidate}")
            return candidate, "original"
    
    # Deeper search
    for root, dirs, files in os.walk(base_path):
        if "Train" in dirs and "Test" in dirs:
            train_imgs = os.path.join(root, "Train", "JPEGImages")
            if os.path.isdir(train_imgs):
                print(f"[INFO] Found dataset (PASCAL-VOC format): {root}")
                return root, "voc"
            train_pos = os.path.join(root, "Train", "pos")
            if os.path.isdir(train_pos):
                print(f"[INFO] Found dataset (original format): {root}")
                return root, "original"
    
    raise FileNotFoundError(
        f"Could not find INRIA dataset under '{base_path}'.\n"
        f"Please download from: https://www.kaggle.com/datasets/jcoral/inriaperson"
    )


def load_images_from_dir(directory, target_size=WINDOW_SIZE, max_images=None):
    """
    Load and resize images from a directory.
    Returns list of BGR images resized to target_size.
    """
    extensions = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.ppm", "*.pgm"]
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(directory, ext)))
        image_paths.extend(glob.glob(os.path.join(directory, "**", ext), recursive=True))
    
    image_paths = sorted(set(image_paths))
    if max_images:
        image_paths = image_paths[:max_images]
    
    images = []
    for path in tqdm(image_paths, desc=f"Loading {os.path.basename(directory)}", leave=False):
        img = cv2.imread(path)
        if img is not None:
            img = cv2.resize(img, target_size)
            images.append(img)
    
    return images


def parse_voc_annotation(xml_path):
    """Parse a PASCAL-VOC XML annotation file. Returns list of person bboxes."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    bboxes = []
    for obj in root.findall('object'):
        if obj.find('name').text == 'person':
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            bboxes.append((xmin, ymin, xmax, ymax))
    return bboxes


def crop_positive_patches(img_dir, ann_dir, target_size=WINDOW_SIZE, max_images=None):
    """
    Crop person regions from annotated images.
    Each bounding box is cropped and resized to target_size.
    """
    extensions = ["*.png", "*.jpg", "*.jpeg"]
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(img_dir, ext)))
    image_paths = sorted(set(image_paths))
    if max_images:
        image_paths = image_paths[:max_images]
    
    patches = []
    for img_path in tqdm(image_paths, desc="Cropping positives", leave=False):
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # Find corresponding annotation
        basename = os.path.splitext(os.path.basename(img_path))[0]
        ann_path = os.path.join(ann_dir, basename + ".xml")
        if not os.path.exists(ann_path):
            continue
        
        bboxes = parse_voc_annotation(ann_path)
        h, w = img.shape[:2]
        
        for (xmin, ymin, xmax, ymax) in bboxes:
            # Clip to image bounds
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(w, xmax)
            ymax = min(h, ymax)
            
            if xmax - xmin < 20 or ymax - ymin < 40:
                continue  # Skip tiny boxes
            
            crop = img[ymin:ymax, xmin:xmax]
            crop = cv2.resize(crop, target_size)
            patches.append(crop)
    
    return patches


def sample_negative_patches_voc(img_dir, ann_dir, num_patches_per_image=NEG_PATCHES_PER_IMAGE,
                                target_size=WINDOW_SIZE, max_images=None):
    """
    Sample patches from image regions that do NOT overlap with person bounding boxes.
    This creates negative (non-person) training examples.
    """
    extensions = ["*.png", "*.jpg", "*.jpeg"]
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(img_dir, ext)))
    image_paths = sorted(set(image_paths))
    if max_images:
        image_paths = image_paths[:max_images]
    
    patches = []
    rng = np.random.RandomState(42)
    tw, th = target_size
    
    for img_path in tqdm(image_paths, desc="Sampling negatives", leave=False):
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        h, w = img.shape[:2]
        if h < th or w < tw:
            continue
        
        # Get person bounding boxes
        basename = os.path.splitext(os.path.basename(img_path))[0]
        ann_path = os.path.join(ann_dir, basename + ".xml")
        bboxes = []
        if os.path.exists(ann_path):
            bboxes = parse_voc_annotation(ann_path)
        
        attempts = 0
        sampled = 0
        max_attempts = num_patches_per_image * 10
        
        while sampled < num_patches_per_image and attempts < max_attempts:
            attempts += 1
            y = rng.randint(0, h - th + 1)
            x = rng.randint(0, w - tw + 1)
            
            # Check overlap with any person bbox
            patch_box = (x, y, x + tw, y + th)
            overlaps = False
            for (bx1, by1, bx2, by2) in bboxes:
                # Compute IoU-like overlap
                ix1 = max(patch_box[0], bx1)
                iy1 = max(patch_box[1], by1)
                ix2 = min(patch_box[2], bx2)
                iy2 = min(patch_box[3], by2)
                
                if ix1 < ix2 and iy1 < iy2:
                    inter_area = (ix2 - ix1) * (iy2 - iy1)
                    patch_area = tw * th
                    if inter_area / patch_area > 0.3:  # >30% overlap = skip
                        overlaps = True
                        break
            
            if not overlaps:
                patch = img[y:y+th, x:x+tw]
                patches.append(patch)
                sampled += 1
    
    return patches


def sample_negative_patches(neg_dir, num_patches_per_image=NEG_PATCHES_PER_IMAGE,
                            target_size=WINDOW_SIZE, max_images=None):
    """
    Sample random patches from negative (person-free) images.
    Used for original dataset format with dedicated negative images.
    """
    extensions = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.ppm", "*.pgm"]
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(neg_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(neg_dir, "**", ext), recursive=True))
    
    image_paths = sorted(set(image_paths))
    if max_images:
        image_paths = image_paths[:max_images]
    
    patches = []
    rng = np.random.RandomState(42)
    
    for path in tqdm(image_paths, desc="Sampling negatives", leave=False):
        img = cv2.imread(path)
        if img is None:
            continue
        
        h, w = img.shape[:2]
        tw, th = target_size
        
        if h < th or w < tw:
            patches.append(cv2.resize(img, target_size))
            continue
        
        for _ in range(num_patches_per_image):
            y = rng.randint(0, h - th + 1)
            x = rng.randint(0, w - tw + 1)
            patch = img[y:y+th, x:x+tw]
            patches.append(patch)
    
    return patches


def load_dataset(data_root="data", max_train_pos=None, max_train_neg=None,
                 max_test_pos=None, max_test_neg=None):
    """
    Load the full INRIA dataset.
    Supports both PASCAL-VOC format (Kaggle) and original format (pos/neg dirs).
    Returns: train_pos, train_neg, test_pos, test_neg (lists of BGR images)
    """
    root, fmt = find_dataset_root(data_root)
    
    if fmt == "voc":
        # PASCAL-VOC format (Kaggle download)
        train_img_dir = os.path.join(root, "Train", "JPEGImages")
        train_ann_dir = os.path.join(root, "Train", "Annotations")
        test_img_dir = os.path.join(root, "Test", "JPEGImages")
        test_ann_dir = os.path.join(root, "Test", "Annotations")
        
        print("\n[1/4] Cropping positive training patches from bounding boxes...")
        train_pos = crop_positive_patches(train_img_dir, train_ann_dir, max_images=max_train_pos)
        print(f"       Cropped {len(train_pos)} positive training patches")
        
        print("[2/4] Sampling negative training patches (non-person regions)...")
        train_neg = sample_negative_patches_voc(train_img_dir, train_ann_dir, max_images=max_train_neg)
        print(f"       Sampled {len(train_neg)} negative training patches")
        
        print("[3/4] Cropping positive test patches...")
        test_pos = crop_positive_patches(test_img_dir, test_ann_dir, max_images=max_test_pos)
        print(f"       Cropped {len(test_pos)} positive test patches")
        
        print("[4/4] Sampling negative test patches...")
        test_neg = sample_negative_patches_voc(test_img_dir, test_ann_dir, max_images=max_test_neg)
        print(f"       Sampled {len(test_neg)} negative test patches")
        
    else:
        # Original format (pos/neg directories)
        train_pos_dir = os.path.join(root, "Train", "pos")
        train_neg_dir = os.path.join(root, "Train", "neg")
        test_pos_dir = os.path.join(root, "Test", "pos")
        test_neg_dir = os.path.join(root, "Test", "neg")
        
        print("\n[1/4] Loading positive training images...")
        train_pos = load_images_from_dir(train_pos_dir, max_images=max_train_pos)
        print(f"       Loaded {len(train_pos)} positive training images")
        
        print("[2/4] Sampling negative training patches...")
        train_neg = sample_negative_patches(train_neg_dir, max_images=max_train_neg)
        print(f"       Sampled {len(train_neg)} negative training patches")
        
        print("[3/4] Loading positive test images...")
        test_pos = load_images_from_dir(test_pos_dir, max_images=max_test_pos)
        print(f"       Loaded {len(test_pos)} positive test images")
        
        print("[4/4] Sampling negative test patches...")
        test_neg = sample_negative_patches(test_neg_dir, max_images=max_test_neg)
        print(f"       Sampled {len(test_neg)} negative test patches")
    
    return train_pos, train_neg, test_pos, test_neg


# ============================================================
#  Preprocessing functions
# ============================================================

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
    This is one of our proposed improvements over the paper's simple
    square-root gamma normalization.
    """
    if len(image.shape) == 3:
        # Convert to LAB color space, apply CLAHE on L channel
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        lab[:, :, 0] = clahe.apply(l_channel)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(image)


def gamma_correction(image, gamma=0.5):
    """
    Square-root gamma correction as used in the original paper.
    gamma=0.5 corresponds to square root.
    """
    img_float = image.astype(np.float64) / 255.0
    corrected = np.power(img_float, gamma)
    return (corrected * 255).astype(np.uint8)


# ============================================================
#  HOG feature extraction
# ============================================================

def extract_hog_opencv(image, use_scharr=False):
    """
    Extract HOG features using OpenCV's HOGDescriptor.
    
    Parameters:
    - image: BGR image (64x128)
    - use_scharr: if True, compute gradients with Scharr operator first
    
    Returns: 1D feature vector
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    hog = cv2.HOGDescriptor(
        _winSize=WINDOW_SIZE,
        _blockSize=BLOCK_SIZE,
        _blockStride=BLOCK_STRIDE,
        _cellSize=CELL_SIZE,
        _nbins=NUM_BINS
    )
    
    if use_scharr:
        return extract_hog_scharr(gray)
    
    features = hog.compute(gray)
    return features.flatten()


def extract_hog_scharr(gray_image):
    """
    Custom HOG extraction using Scharr gradient operator.
    Scharr has better rotational symmetry than the simple [-1,0,1] filter
    used in the original paper.
    
    Scharr kernel:
    Gx = [-3  0  3]      Gy = [-3 -10 -3]
         [-10 0  10]           [ 0   0   0]
         [-3  0  3]            [ 3  10   3]
    """
    # Compute gradients using Scharr operator
    grad_x = cv2.Scharr(gray_image, cv2.CV_64F, 1, 0)
    grad_y = cv2.Scharr(gray_image, cv2.CV_64F, 0, 1)
    
    # Magnitude and orientation
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    orientation = np.arctan2(grad_y, grad_x) * (180.0 / np.pi) % 180  # Unsigned 0-180
    
    h, w = gray_image.shape
    cell_h, cell_w = CELL_SIZE
    n_cells_y = h // cell_h
    n_cells_x = w // cell_w
    
    # Compute cell histograms
    cell_hists = np.zeros((n_cells_y, n_cells_x, NUM_BINS))
    bin_width = 180.0 / NUM_BINS
    
    for cy in range(n_cells_y):
        for cx in range(n_cells_x):
            y0, y1 = cy * cell_h, (cy + 1) * cell_h
            x0, x1 = cx * cell_w, (cx + 1) * cell_w
            
            cell_mag = magnitude[y0:y1, x0:x1]
            cell_ori = orientation[y0:y1, x0:x1]
            
            for b in range(NUM_BINS):
                bin_center = b * bin_width + bin_width / 2
                diff = np.abs(cell_ori - bin_center)
                diff = np.minimum(diff, 180.0 - diff)
                weight = np.maximum(0, 1.0 - diff / bin_width)
                cell_hists[cy, cx, b] = np.sum(cell_mag * weight)
    
    # Block normalization (L2-Hys as in the paper)
    block_cells_y = BLOCK_SIZE[1] // cell_h
    block_cells_x = BLOCK_SIZE[0] // cell_w
    stride_y = BLOCK_STRIDE[1] // cell_h
    stride_x = BLOCK_STRIDE[0] // cell_w
    
    features = []
    eps = 1e-5
    
    for by in range(0, n_cells_y - block_cells_y + 1, stride_y):
        for bx in range(0, n_cells_x - block_cells_x + 1, stride_x):
            block = cell_hists[by:by+block_cells_y, bx:bx+block_cells_x].flatten()
            
            # L2-Hys normalization (L2 norm, clip to 0.2, renormalize)
            norm = np.sqrt(np.sum(block**2) + eps**2)
            block = block / norm
            block = np.clip(block, 0, 0.2)
            norm = np.sqrt(np.sum(block**2) + eps**2)
            block = block / norm
            
            features.extend(block)
    
    return np.array(features, dtype=np.float32)


def extract_lbp_features(image, radius=1, n_points=8):
    """
    Extract Local Binary Pattern (LBP) histogram features.
    LBP captures micro-texture patterns that HOG misses.
    Uses uniform LBP for rotation invariance.
    """
    from skimage.feature import local_binary_pattern
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Compute LBP
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    
    # Compute histogram per cell (same grid as HOG for consistency)
    h, w = gray.shape
    cell_h, cell_w = CELL_SIZE
    n_cells_y = h // cell_h
    n_cells_x = w // cell_w
    n_bins = n_points + 2  # Uniform LBP has P+2 bins
    
    features = []
    for cy in range(n_cells_y):
        for cx in range(n_cells_x):
            y0, y1 = cy * cell_h, (cy + 1) * cell_h
            x0, x1 = cx * cell_w, (cx + 1) * cell_w
            cell_lbp = lbp[y0:y1, x0:x1]
            hist, _ = np.histogram(cell_lbp, bins=n_bins, range=(0, n_bins), density=True)
            features.extend(hist)
    
    return np.array(features, dtype=np.float32)


# ============================================================
#  Feature extraction pipelines
# ============================================================

def extract_features_baseline(images):
    """Baseline HOG (replicating the original paper)."""
    features = []
    for img in tqdm(images, desc="Extracting Baseline HOG", leave=False):
        img = gamma_correction(img, gamma=0.5)
        feat = extract_hog_opencv(img)
        features.append(feat)
    return np.array(features)


def extract_features_clahe_hog(images):
    """Improvement 1: CLAHE preprocessing + HOG."""
    features = []
    for img in tqdm(images, desc="Extracting CLAHE+HOG", leave=False):
        img = apply_clahe(img)
        feat = extract_hog_opencv(img)
        features.append(feat)
    return np.array(features)


def extract_features_scharr_hog(images):
    """Improvement 2: HOG with Scharr gradient operator."""
    features = []
    for img in tqdm(images, desc="Extracting Scharr HOG", leave=False):
        img = gamma_correction(img, gamma=0.5)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        feat = extract_hog_scharr(gray)
        features.append(feat)
    return np.array(features)


def extract_features_combined(images):
    """Improvement 3: CLAHE + Scharr HOG + LBP (all improvements)."""
    features = []
    for img in tqdm(images, desc="Extracting Combined", leave=False):
        img_clahe = apply_clahe(img)
        gray = cv2.cvtColor(img_clahe, cv2.COLOR_BGR2GRAY)
        
        # Scharr HOG features
        hog_feat = extract_hog_scharr(gray)
        
        # LBP features
        lbp_feat = extract_lbp_features(img_clahe)
        
        # Concatenate
        feat = np.concatenate([hog_feat, lbp_feat])
        features.append(feat)
    return np.array(features)


# ============================================================
#  Extraction dispatch
# ============================================================

METHODS = {
    "Baseline HOG": extract_features_baseline,
    "CLAHE + HOG": extract_features_clahe_hog,
    "Scharr HOG": extract_features_scharr_hog,
    "Combined (CLAHE+Scharr+LBP)": extract_features_combined,
}


if __name__ == "__main__":
    # Quick test
    print("Testing utility functions...")
    dummy = np.random.randint(0, 255, (128, 64, 3), dtype=np.uint8)
    
    print(f"  Baseline HOG shape: {extract_hog_opencv(dummy).shape}")
    gray = cv2.cvtColor(dummy, cv2.COLOR_BGR2GRAY)
    print(f"  Scharr HOG shape:   {extract_hog_scharr(gray).shape}")
    print(f"  LBP shape:          {extract_lbp_features(dummy).shape}")
    print(f"  CLAHE output shape: {apply_clahe(dummy).shape}")
    print("All tests passed!")
