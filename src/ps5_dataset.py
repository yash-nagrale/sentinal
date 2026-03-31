"""
ps5_dataset.py
--------------
Loads CT scan images for stroke classification.

Your folder structure:
PS5/
  classification/
    train/
      Normal/    ← normal brain CT scans
      Stroke/    ← CT scans with hemorrhage
    val/
      Normal/
      Stroke/
"""

import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# ── Your exact path ──────────────────────────────────────────────────────────
PS5_ROOT = r"C:\Users\Yash Nagrale\Desktop\hackathon\PS5 - AI-Based Brain Stroke Detection and Lesion Segmentation from CT Scans"

CLASS_TO_LABEL = {"Normal": 0, "Stroke": 1}
LABEL_TO_CLASS = {0: "Normal", 1: "Stroke"}

IMG_SIZE      = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── Transforms ────────────────────────────────────────────────────────────────
# CT scans: don't use vertical flip or hue — not medically meaningful
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ── Dataset class ─────────────────────────────────────────────────────────────
class CTScanDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.transform = transform
        self.samples   = []

        # Try both capitalisation variants for folder names
        for class_name, label in CLASS_TO_LABEL.items():
            for variant in [class_name, class_name.lower(), class_name.upper()]:
                class_dir = os.path.join(root_dir, "classification", split, variant)
                if os.path.exists(class_dir):
                    for fname in os.listdir(class_dir):
                        if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                            self.samples.append((os.path.join(class_dir, fname), label))
                    break

        print(f"  Loaded {split} set: {len(self.samples)} images")
        for class_name, label in CLASS_TO_LABEL.items():
            count = sum(1 for _, l in self.samples if l == label)
            print(f"    {class_name}: {count} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


# ── DataLoader builder ────────────────────────────────────────────────────────
def get_dataloaders(batch_size=32):
    print("Loading PS5 CT scan dataset...")
    train_dataset = CTScanDataset(PS5_ROOT, split="train", transform=train_transform)
    val_dataset   = CTScanDataset(PS5_ROOT, split="val",   transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, num_workers=0, pin_memory=True)

    print(f"\n  Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    return train_loader, val_loader


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_loader, val_loader = get_dataloaders(batch_size=8)
    images, labels = next(iter(train_loader))
    print(f"\nBatch shape: {images.shape}")
    print(f"Labels: {[LABEL_TO_CLASS[l.item()] for l in labels]}")
    print("Dataset loading works correctly!")
