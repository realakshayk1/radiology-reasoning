"""
predict.py — Inference with calibrated probabilities.
"""

import os
import yaml
import torch
import pandas as pd
from PIL import Image
from src.models.image_baseline import get_model
from src.data.preprocess_images import TRANSFORM

LABEL_COLS = ["cardiomegaly", "effusion", "edema", "pneumothorax", "consolidation"]

class RadiologistCNN:
    """Wrapper for the image classifier with calibration support."""
    
    def __init__(self, model_path="artifacts/models/best_model.pt", 
                 config_path="configs/train_image.yaml",
                 calib_path="artifacts/models/calibration.yaml"):
        
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = get_model(self.cfg).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Load temperature if available
        self.temperature = 1.0
        if os.path.exists(calib_path):
            with open(calib_path, "r") as f:
                calib = yaml.safe_load(f)
                self.temperature = calib.get("temperature", 1.0)
                
    @torch.no_grad()
    def predict(self, image_path: str):
        """Returns calibrated probabilities for the image."""
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise ValueError(f"Could not load image at {image_path}: {e}")
            
        img_tensor = TRANSFORM(image).unsqueeze(0).to(self.device)
        logits = self.model(img_tensor)
        
        # Apply temperature scaling
        calibrated_logits = logits / self.temperature
        probs = torch.sigmoid(calibrated_logits).squeeze(0).cpu().numpy()
        
        return {label: float(prob) for label, prob in zip(LABEL_COLS, probs)}

if __name__ == "__main__":
    # Test with a sample from processed studies
    from src.utils.paths import get_project_root
    
    # Simple check
    predictor = RadiologistCNN()
    studies = pd.read_csv("data/processed/studies.csv")
    sample_img = studies.iloc[0]["image_path"]
    
    print(f"Predicting for: {sample_img}")
    res = predictor.predict(sample_img)
    print("\nCalibrated Predictions:")
    for label, prob in res.items():
        print(f"  {label:<15}: {prob:.4f}")
