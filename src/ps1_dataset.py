"""
ps1_dataset.py  (improved version)
------------------------------------
Key fix: instead of using the small 'valid' folder as-is (only 282 images,
Grade 3 had just 40), we combine ALL images and re-split 85/15 ourselves.
This gives every grade a fair share of validation images.
"""

import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms

# ── Your exact path ──────────────────────────────────────────────────────────
PS1_ROOT = r"C:\Users\Yash Nagrale\Desktop\hackathon\PS1 - AI-Based Prediction of High-Risk Plantar Pressure Zones for Prevention of Diabetic Foot Ulcers"

GRADE_TO_LABEL = {
    "Grade 1": 0,
    "Grade 2": 1,
    "Grade 3": 2,
    "Grade 4": 3,
}
LABEL_TO_GRADE = {v: k for k, v in GRADE_TO_LABEL.items()}

IMG_SIZE      = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── Transforms ───────────────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class FootWoundDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


def collect_all_samples(root_dir):
    all_samples = []
    for split_folder in ["train", "valid"]:
        split_dir = os.path.join(root_dir, split_folder)
        if not os.path.exists(split_dir):
            continue
        for grade_name, label in GRADE_TO_LABEL.items():
            grade_dir = os.path.join(split_dir, grade_name)
            if not os.path.exists(grade_dir):
                continue
            for fname in os.listdir(grade_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    all_samples.append((os.path.join(grade_dir, fname), label))
    return all_samples


def get_dataloaders(batch_size=32, val_size=0.15, random_seed=42):
    print("Loading PS1 foot wound dataset (combined + re-split)...")
    all_samples = collect_all_samples(PS1_ROOT)
    all_labels  = [label for _, label in all_samples]

    print(f"  Total images found: {len(all_samples)}")
    for grade_name, label in GRADE_TO_LABEL.items():
        print(f"    {grade_name}: {all_labels.count(label)}")

    train_samples, val_samples = train_test_split(
        all_samples, test_size=val_size,
        stratify=all_labels, random_state=random_seed,
    )

    print(f"\n  Train: {len(train_samples)} | Val: {len(val_samples)}")
    val_labels = [l for _, l in val_samples]
    for grade_name, label in GRADE_TO_LABEL.items():
        print(f"    Val {grade_name}: {val_labels.count(label)}")

    train_labels  = [l for _, l in train_samples]
    total         = len(train_labels)
    class_weights = torch.FloatTensor([
        total / (4 * train_labels.count(i)) for i in range(4)
    ])
    print(f"\n  Class weights: {[f'{w:.3f}' for w in class_weights]}")

    train_loader = DataLoader(FootWoundDataset(train_samples, train_transform),
                              batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(FootWoundDataset(val_samples,   val_transform),
                              batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    print(f"  Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    return train_loader, val_loader, class_weights


if __name__ == "__main__":
    train_loader, val_loader, weights = get_dataloaders(batch_size=8)
    images, labels = next(iter(train_loader))
    print(f"\nBatch shape: {images.shape}")
    print(f"Labels: {[LABEL_TO_GRADE[l.item()] for l in labels]}")
    print("Dataset loading works correctly!")
