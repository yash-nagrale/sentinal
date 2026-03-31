"""
ps5_model.py
------------
EfficientNet-B0 for binary CT stroke classification.
Almost identical to ps1_model.py — just 2 output classes instead of 4.
"""

import torch
import torch.nn as nn
import torchvision.models as models

NUM_CLASSES = 2   # Normal, Stroke


class StrokeClassifier(nn.Module):
    """
    EfficientNet-B0 fine-tuned for stroke detection.

    Input : CT scan image (224x224x3)
    Output: 2 scores — [Normal score, Stroke score]
    """

    def __init__(self, num_classes=NUM_CLASSES, dropout=0.4):
        super(StrokeClassifier, self).__init__()

        self.backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )

        in_features = self.backbone.classifier[1].in_features  # 1280
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)

    def predict_proba(self, x):
        """Returns probability of Stroke (0 to 1)."""
        logits = self.forward(x)
        probs  = torch.softmax(logits, dim=1)
        return probs[:, 1]   # probability of class 1 = Stroke


def get_model():
    model = StrokeClassifier()
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: EfficientNet-B0 for stroke detection")
    print(f"  Total parameters    : {total:,}")
    print(f"  Trainable parameters: {trainable:,}")
    return model


if __name__ == "__main__":
    model = get_model()
    dummy = torch.randn(4, 3, 224, 224)
    out   = model(dummy)
    prob  = model.predict_proba(dummy)
    print(f"\nTest forward pass:")
    print(f"  Input shape : {dummy.shape}")
    print(f"  Output shape: {out.shape}")    # [4, 2]
    print(f"  Stroke probs: {prob.detach().tolist()}")
    print("Model works correctly!")
