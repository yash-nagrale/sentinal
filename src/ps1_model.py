"""
ps1_model.py
------------
Defines the EfficientNet-B0 model for foot wound grade classification.

What this file does:
- Takes EfficientNet-B0 (pretrained on 1 million images by Google)
- Replaces the final layer with one that outputs 4 grades instead of 1000 classes
- That's it — this is transfer learning in ~20 lines of code
"""

import torch
import torch.nn as nn
import torchvision.models as models

NUM_CLASSES = 4   # Grade 1, 2, 3, 4


class FootWoundClassifier(nn.Module):
    """
    EfficientNet-B0 fine-tuned for 4-class foot wound grading.

    Architecture:
    Input image (224x224x3)
        ↓
    EfficientNet-B0 backbone (pretrained — already knows edges, textures, shapes)
        ↓
    Dropout (randomly turns off 30% of neurons — prevents overfitting)
        ↓
    Linear layer (1280 → 4)  ← this is the part we train from scratch
        ↓
    Output: 4 scores, one per grade
    """

    def __init__(self, num_classes=NUM_CLASSES, dropout=0.3, freeze_backbone=False):
        super(FootWoundClassifier, self).__init__()

        # Load EfficientNet-B0 with pretrained ImageNet weights
        # This downloads ~20MB of weights the first time you run it
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

        # Optionally freeze backbone (don't update pretrained weights)
        # Set to True if you want faster training but slightly lower accuracy
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Replace the final classification layer
        # Original: 1280 → 1000 (ImageNet has 1000 classes)
        # Ours:     1280 → 4    (we have 4 wound grades)
        in_features = self.backbone.classifier[1].in_features  # = 1280
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(256, num_classes),
            # No softmax here — CrossEntropyLoss applies it internally
        )

    def forward(self, x):
        """
        x: batch of images, shape (batch_size, 3, 224, 224)
        returns: scores for each grade, shape (batch_size, 4)
        """
        return self.backbone(x)


def get_model():
    """Returns the model ready for training."""
    model = FootWoundClassifier(num_classes=NUM_CLASSES, dropout=0.5)

    # Count parameters
    total  = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: EfficientNet-B0 for foot wound grading")
    print(f"  Total parameters    : {total:,}")
    print(f"  Trainable parameters: {trainable:,}")

    return model


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = get_model()

    # Test with a fake batch of 4 images
    dummy_input = torch.randn(4, 3, 224, 224)
    output = model(dummy_input)

    print(f"\nTest forward pass:")
    print(f"  Input shape : {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")   # should be [4, 4]
    print(f"  Output (raw scores):\n{output.detach()}")
    print("\nModel works correctly!")
