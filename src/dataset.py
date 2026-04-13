"""
Dataset loading, splitting, and preprocessing pipeline.

Handles:
- Merging the tiny default val split into train
- Stratified 80/10/10 re-split
- ImageNet normalisation + augmentation
- Class weight computation for imbalanced loss
- PyTorch DataLoaders ready for training
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split


# ──────────────────────────────────────────────
# 1. Collect all image paths + labels
# ──────────────────────────────────────────────

def collect_image_paths(raw_dir: str) -> pd.DataFrame:
    """
    Walk through the Kaggle chest_xray folder and build a DataFrame
    with columns: [filepath, label, original_split].

    Kaggle structure:
      chest_xray/train/NORMAL/  chest_xray/train/PNEUMONIA/
      chest_xray/val/NORMAL/    chest_xray/val/PNEUMONIA/
      chest_xray/test/NORMAL/   chest_xray/test/PNEUMONIA/

    Labels: NORMAL=0, PNEUMONIA=1
    """
    records = []
    label_map = {"NORMAL": 0, "PNEUMONIA": 1}

    for split in ["train", "val", "test"]:
        split_dir = os.path.join(raw_dir, split)
        if not os.path.exists(split_dir):
            print(f"Warning: {split_dir} not found, skipping")
            continue

        for class_name in ["NORMAL", "PNEUMONIA"]:
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.exists(class_dir):
                continue

            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".jpeg", ".jpg", ".png")):
                    records.append({
                        "filepath": os.path.join(class_dir, fname),
                        "label": label_map[class_name],
                        "original_split": split,
                    })

    df = pd.DataFrame(records)
    print(f"Total images found: {len(df)}")
    print(f"  By original split: {dict(df['original_split'].value_counts())}")
    print(f"  By label: {dict(df['label'].value_counts())}")
    return df


# ──────────────────────────────────────────────
# 2. Stratified re-split
# ──────────────────────────────────────────────

def stratified_split(
    df: pd.DataFrame,
    split_ratios: list = [0.8, 0.1, 0.1],
    seed: int = 42,
    save_path: str = None,
) -> tuple:
    """
    Merge train+val, keep original test, re-split train+val into
    new train/val using stratified sampling.

    Returns: (train_df, val_df, test_df)
    """
    # Keep original test set untouched
    test_df = df[df["original_split"] == "test"].copy()

    # Merge train + val (the val split is only 16 images)
    trainval_df = df[df["original_split"].isin(["train", "val"])].copy()

    print(f"\nMerged train+val: {len(trainval_df)} images")
    print(f"Original test kept: {len(test_df)} images")

    # Split trainval into new train and val
    # Ratio: from the trainval pool, val = 0.1 / (0.8 + 0.1) ≈ 0.111
    val_ratio = split_ratios[1] / (split_ratios[0] + split_ratios[1])

    train_df, val_df = train_test_split(
        trainval_df,
        test_size=val_ratio,
        stratify=trainval_df["label"],
        random_state=seed,
    )

    # Reset indices
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print(f"\nNew split sizes:")
    print(f"  Train: {len(train_df)}  (Normal: {sum(train_df['label']==0)}, Pneumonia: {sum(train_df['label']==1)})")
    print(f"  Val:   {len(val_df)}  (Normal: {sum(val_df['label']==0)}, Pneumonia: {sum(val_df['label']==1)})")
    print(f"  Test:  {len(test_df)}  (Normal: {sum(test_df['label']==0)}, Pneumonia: {sum(test_df['label']==1)})")

    # Save split indices for reproducibility
    if save_path:
        train_df["split"] = "train"
        val_df["split"] = "val"
        test_df["split"] = "test"
        all_splits = pd.concat([train_df, val_df, test_df], ignore_index=True)
        all_splits.to_csv(save_path, index=False)
        print(f"\nSplit indices saved to: {save_path}")
        # Remove the temporary column
        train_df = train_df.drop(columns=["split"])
        val_df = val_df.drop(columns=["split"])
        test_df = test_df.drop(columns=["split"])

    return train_df, val_df, test_df


# ──────────────────────────────────────────────
# 3. Transforms
# ──────────────────────────────────────────────

def get_transforms(config: dict) -> dict:
    """
    Build train and val/test transforms from config.

    Train: augmentation + normalise
    Val/Test: just resize + normalise
    """
    prep = config["preprocessing"]
    img_size = prep["image_size"]
    mean = prep["imagenet_mean"]
    std = prep["imagenet_std"]
    aug = prep["augmentation"]

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip() if aug["horizontal_flip"] else transforms.Lambda(lambda x: x),
        transforms.RandomRotation(aug["rotation_degrees"]),
        transforms.ColorJitter(
            brightness=aug["brightness"],
            contrast=aug["contrast"],
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return {"train": train_transform, "val": eval_transform, "test": eval_transform}


# ──────────────────────────────────────────────
# 4. PyTorch Dataset
# ──────────────────────────────────────────────

class ChestXrayDataset(Dataset):
    """
    PyTorch Dataset for chest X-ray images.

    Args:
        dataframe: DataFrame with 'filepath' and 'label' columns
        transform: torchvision transform to apply
    """

    def __init__(self, dataframe: pd.DataFrame, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image = Image.open(row["filepath"]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(row["label"], dtype=torch.long)
        return image, label


# ──────────────────────────────────────────────
# 5. Class weights
# ──────────────────────────────────────────────

def compute_class_weights(train_df: pd.DataFrame) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for imbalanced loss.

    Returns: torch.Tensor of shape [2] — weights for [NORMAL, PNEUMONIA]
    """
    counts = Counter(train_df["label"].values)
    total = sum(counts.values())
    weights = [total / (len(counts) * counts[i]) for i in range(len(counts))]
    weights_tensor = torch.FloatTensor(weights)
    print(f"\nClass weights: Normal={weights[0]:.3f}, Pneumonia={weights[1]:.3f}")
    return weights_tensor


# ──────────────────────────────────────────────
# 6. DataLoaders — the main entry point
# ──────────────────────────────────────────────

def get_dataloaders(config: dict) -> dict:
    """
    Complete pipeline: collect images → split → transform → DataLoaders.

    Usage:
        config = load_config()
        loaders = get_dataloaders(config)
        for images, labels in loaders['train']:
            ...

    Returns: dict with keys 'train', 'val', 'test' (DataLoaders)
             and 'class_weights' (torch.Tensor)
    """
    # Collect all image paths
    df = collect_image_paths(config["data"]["raw_dir"])

    # Stratified split
    train_df, val_df, test_df = stratified_split(
        df,
        split_ratios=config["data"]["split_ratios"],
        seed=config["data"]["random_seed"],
        save_path="data/split_indices.csv",
    )

    # Build transforms
    tfms = get_transforms(config)

    # Build datasets
    train_ds = ChestXrayDataset(train_df, transform=tfms["train"])
    val_ds = ChestXrayDataset(val_df, transform=tfms["val"])
    test_ds = ChestXrayDataset(test_df, transform=tfms["test"])

    # Build DataLoaders
    batch_size = config["training"]["batch_size"]
    num_workers = config["data"]["num_workers"]

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Class weights
    class_weights = compute_class_weights(train_df)

    print(f"\nDataLoaders ready:")
    print(f"  Train: {len(train_loader)} batches ({len(train_ds)} images)")
    print(f"  Val:   {len(val_loader)} batches ({len(val_ds)} images)")
    print(f"  Test:  {len(test_loader)} batches ({len(test_ds)} images)")

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "class_weights": class_weights,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
    }
