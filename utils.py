import os
import torch
from torchvision import models
import torchvision.transforms as T
from PIL import Image

# === Configuration ===
PROJECT_ROOT = r"E:\00. Master's in AI\Sem 1\AI\Project\code new\Hybrid-Model-for-Skin-Disease-Classification-EfficientNetB0-and-ViT-"
MODEL_PATH = os.path.join(PROJECT_ROOT, 'efficientnet_model.pth')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Standard ImageNet normalization
preprocess = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_image(path):
    """Loads an image and converts to RGB."""
    return Image.open(path).convert('RGB')

def load_model(device=DEVICE):
    """Loads the EfficientNet model with specific error handling."""
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 7)
    
    try:
        # weights_only=False handles the ONNX/Pickle security warning
        state = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
        model.load_state_dict(state, strict=False)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Warning: Could not load weights from {MODEL_PATH}")
        print(f"Error details: {e}")
        
    model.to(device)
    model.eval()
    return model