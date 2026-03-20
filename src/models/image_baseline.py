"""
image_baseline.py — Multi-label classifier for OpenI dataset.

Backbone: EfficientNet-B0 (ImageNet pretrained) — configurable via train_image.yaml
Head: Dropout(0.4) + Linear with 5 outputs (Sigmoid applied via BCEWithLogitsLoss)
Findings: Cardiomegaly, Pleural Effusion, Edema, Pneumothorax, Consolidation
"""

import torch
import torch.nn as nn
from torchvision import models

class ImageBaseline(nn.Module):
    """EfficientNet-B0 multi-label classifier (backbone configurable)."""
    
    def __init__(self, backbone_name="efficientnet_b0", num_classes=5, pretrained=True):
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
            
        # Dropout before classifier head — reduces overfitting on small OpenI dataset
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: (B, 3, 224, 224)
        Output: (B, 5) logits
        """
        features = self.backbone(x)
        logits = self.fc(self.dropout(features))
        
        # Assertion on output shape
        # Note: In PyTorch models, batch_size is dynamic, but we can check num_classes.
        assert logits.shape[1] == 5, f"Bad output shape: {logits.shape}"
        
        return logits

def get_model(config):
    """Factory function for model creation."""
    return ImageBaseline(
        backbone_name=config.get("backbone", "efficientnet_b0"),
        num_classes=5,
        pretrained=True
    )

if __name__ == "__main__":
    # Test both backbones
    dummy_input = torch.randn(4, 3, 224, 224)
    for backbone in ["densenet121", "efficientnet_b0"]:
        model = ImageBaseline(backbone_name=backbone, num_classes=5, pretrained=False)
        logits = model(dummy_input)
        assert logits.shape == (4, 5), f"{backbone}: bad shape {logits.shape}"
        print(f"{backbone}: input={dummy_input.shape} output={logits.shape} OK")