"""
image_baseline.py — DenseNet121 multi-label classifier for OpenI dataset.

Backbone: DenseNet121 (ImageNet pretrained)
Head: Linear layer with 5 outputs (Sigmoid applied via BCEWithLogitsLoss)
Findings: Cardiomegaly, Pleural Effusion, Edema, Pneumothorax, Consolidation
"""

import torch
import torch.nn as nn
from torchvision import models

class ImageBaseline(nn.Module):
    """DenseNet121 multi-label classifier."""
    
    def __init__(self, backbone_name="densenet121", num_classes=5, pretrained=True):
        super(ImageBaseline, self).__init__()
        
        if backbone_name == "densenet121":
            weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.densenet121(weights=weights)
            # DenseNet121 last layer is 'classifier.in_features'
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()  # Remove original classifier
        elif backbone_name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
            
        self.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: (B, 3, 224, 224)
        Output: (B, 5) logits
        """
        features = self.backbone(x)
        logits = self.fc(features)
        
        # Assertion on output shape
        # Note: In PyTorch models, batch_size is dynamic, but we can check num_classes.
        assert logits.shape[1] == 5, f"Bad output shape: {logits.shape}"
        
        return logits

def get_model(config):
    """Factory function for model creation."""
    return ImageBaseline(
        backbone_name=config.get("backbone", "densenet121"),
        num_classes=5,
        pretrained=True
    )

if __name__ == "__main__":
    # Test forward pass with dummy data
    model = ImageBaseline(backbone_name="densenet121", num_classes=5, pretrained=False)
    dummy_input = torch.randn(4, 3, 224, 224)
    logits = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {logits.shape}")
    assert logits.shape == (4, 5)
    print("Forward pass successful.")
