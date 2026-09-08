"""
explainers.py

Visual Explainability tools for Skin Disease Model.
- Grad-CAM: Focused heatmaps with peak activation circles.
- LIME: Yellow filled regions for defected areas (Superpixels).
- SHAP: Robust Superpixel-based estimation (Fixes DimensionError & RuntimeError).
"""

import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import torchvision.transforms as T
from torchvision import models
import torch.nn.functional as F

# Optional packages
try:
    from lime import lime_image
    from skimage.segmentation import mark_boundaries
except ImportError:
    print("Please install lime and scikit-image: pip install lime scikit-image")
    lime_image = None

try:
    import shap
except ImportError:
    print("Please install shap: pip install shap")
    shap = None

# === Configuration ===
PROJECT_ROOT = r"E:\00. Master's in AI\Sem 1\AI\Project\code new\Hybrid-Model-for-Skin-Disease-Classification-EfficientNetB0-and-ViT-"
MODEL_PATH = os.path.join(PROJECT_ROOT, 'efficientnet_model.pth')
# Adjust DEVICE if needed
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Standard ImageNet normalization
preprocess = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_image(path):
    return Image.open(path).convert('RGB')

def preprocess_tensor(img_pil):
    return preprocess(img_pil).unsqueeze(0)

# ===================== Grad-CAM Explainer =====================
class GradCAMExplainer:
    def __init__(self, model_path=MODEL_PATH, device=DEVICE, target_layer_name=None):
        self.device = torch.device(device)
        self.model = self._load_model(model_path)
        self.model.eval()
        self.model.to(self.device)
        self.target_layer_name = target_layer_name or self._find_last_conv(self.model)

    def _load_model(self, path):
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 7)
        try:
            # weights_only=False fixes the "WeightsUnpickler" error
            state = torch.load(path, map_location='cpu', weights_only=False)
            model.load_state_dict(state, strict=False)
        except Exception as e:
            print(f"Warning (GradCAM): Could not load weights: {e}")
        return model

    def _find_last_conv(self, model):
        last_name = None
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                last_name = name
        return last_name

    def _get_module(self, model, name):
        parts = name.split('.')
        mod = model
        for p in parts:
            mod = getattr(mod, p)
        return mod

    def run_on_image(self, img_path, out_path='explainers_outputs', class_idx=None):
        os.makedirs(out_path, exist_ok=True)
        img_pil = load_image(img_path).resize((224, 224))
        img_tensor = preprocess_tensor(img_pil).to(self.device)

        features = None
        grads = None

        def forward_hook(module, inp, out):
            nonlocal features
            features = out.detach()

        def backward_hook(module, grad_in, grad_out):
            nonlocal grads
            grads = grad_out[0].detach()

        target_module = self._get_module(self.model, self.target_layer_name)
        fh = target_module.register_forward_hook(forward_hook)
        
        # Backward hook compatibility
        try:
            bh = target_module.register_full_backward_hook(backward_hook)
        except AttributeError:
            bh = target_module.register_backward_hook(backward_hook)

        # Forward
        outputs = self.model(img_tensor)
        probs = F.softmax(outputs, dim=1)
        if class_idx is None:
            class_idx = torch.argmax(probs, dim=1).item()

        # Backward
        loss = outputs[0, class_idx]
        self.model.zero_grad()
        loss.backward()

        # Generate CAM
        if grads is None or features is None:
            print("Error: GradCAM hooks failed to capture data.")
            fh.remove(); bh.remove()
            return None

        pooled_grads = torch.mean(grads, dim=[0, 2, 3])
        cam = torch.zeros(features.shape[2:], dtype=torch.float32).to(self.device)
        for i in range(features.shape[1]):
            cam += pooled_grads[i] * features[0, i, :, :]
        
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-9)
        cam_np = cam.cpu().numpy()

        # --- Visualization ---
        cam_resized = cv2.resize(cam_np, (224, 224))
        
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        original_img = np.array(img_pil)

        # Blend
        blended = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)

        # Masking: Only color the active regions
        mask = cam_resized > 0.15 # threshold
        final_img = original_img.copy()
        final_img[mask] = blended[mask]

        # Draw Target Circle
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(cam_resized)
        cv2.circle(final_img, maxLoc, 20, (0, 255, 0), 2)
        cv2.circle(final_img, maxLoc, 22, (255, 255, 255), 1)

        base = os.path.splitext(os.path.basename(img_path))[0]
        save_path = os.path.join(out_path, f"{base}_gradcam.png")
        Image.fromarray(final_img).save(save_path)
        
        fh.remove(); bh.remove()
        print(f"[Grad-CAM] Saved to {save_path}")
        return save_path

# ===================== LIME Explainer =====================
class LIMEExplainerWrapper:
    def __init__(self, model_path=MODEL_PATH, device=DEVICE):
        if lime_image is None:
            raise RuntimeError('LIME not installed.')
        self.device = torch.device(device)
        self.model = self._load_model(model_path)
        self.model.eval().to(self.device)

    def _load_model(self, path):
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 7)
        try:
            state = torch.load(path, map_location='cpu', weights_only=False)
            model.load_state_dict(state, strict=False)
        except Exception:
            pass
        return model

    def predict_fn(self, images):
        self.model.eval()
        batch = []
        for img in images:
            # LIME passes numpy uint8 images
            pil = Image.fromarray(img.astype('uint8'))
            t = preprocess(pil).unsqueeze(0)
            batch.append(t)
        batch = torch.cat(batch, dim=0).to(self.device)
        with torch.no_grad():
            out = self.model(batch)
            probs = F.softmax(out, dim=1).cpu().numpy()
        return probs

    def explain(self, img_path, out_path='explainers_outputs'):
        os.makedirs(out_path, exist_ok=True)
        explainer = lime_image.LimeImageExplainer()
        img_pil = load_image(img_path).resize((224, 224))
        img_np = np.array(img_pil)

        # Run LIME
        explanation = explainer.explain_instance(
            img_np, 
            self.predict_fn, 
            top_labels=1, 
            hide_color=0, 
            num_samples=1000 
        )

        top_label = explanation.top_labels[0]
        
        # Get image and mask (mask is 1 for the important superpixels)
        temp, mask = explanation.get_image_and_mask(
            top_label, 
            positive_only=True, 
            num_features=5, 
            hide_rest=False
        )

        # --- Enhanced Visualization ---
        # 1. Create a color overlay (Yellow)
        overlay = img_np.copy()
        # Set pixels where mask is True to Yellow (255, 255, 0)
        overlay[mask == 1] = [255, 255, 0] 

        # 2. Blend original with overlay
        alpha = 0.5
        cv2_img = img_np.astype(np.float32)
        cv2_overlay = overlay.astype(np.float32)
        blended = cv2.addWeighted(cv2_img, 1 - alpha, cv2_overlay, alpha, 0).astype(np.uint8)

        # 3. Add Boundaries on top for sharpness
        img_boundary = mark_boundaries(blended, mask, color=(1, 1, 0), mode='thick')
        img_boundary = (img_boundary * 255).astype(np.uint8)

        base = os.path.splitext(os.path.basename(img_path))[0]
        save_path = os.path.join(out_path, f"{base}_lime.png")
        Image.fromarray(img_boundary).save(save_path)
        print(f"[LIME] Saved to {save_path}")
        return save_path

# ===================== SHAP Explainer (Robust Fix) =====================
class SHAPExplainer:
    def __init__(self, model_path=MODEL_PATH, device=DEVICE):
        if shap is None:
            raise RuntimeError('SHAP not installed.')
        self.device = torch.device(device)
        self.model = models.efficientnet_b0(weights=None)
        self.model.classifier[1] = torch.nn.Linear(self.model.classifier[1].in_features, 7)
        try:
            state = torch.load(model_path, map_location='cpu', weights_only=False)
            self.model.load_state_dict(state, strict=False)
        except Exception:
             pass
        self.model.to(self.device).eval()

    def predict_wrapper(self, images_numpy):
        """
        Wrapper to convert SHAP's numpy input to Tensor -> Model -> Probabilities
        """
        # SHAP might pass a list or a numpy array
        if isinstance(images_numpy, list):
            images_numpy = np.array(images_numpy)
            
        # Ensure input is standard range 0-255 uint8 for preprocessing
        # If SHAP passes 0-1 floats, convert them.
        if images_numpy.max() <= 1.0:
            images_numpy = (images_numpy * 255).astype(np.uint8)
        else:
            images_numpy = images_numpy.astype(np.uint8)

        batch_tensors = []
        for img in images_numpy:
            pil_img = Image.fromarray(img)
            tensor = preprocess(pil_img).unsqueeze(0)
            batch_tensors.append(tensor)
            
        batch = torch.cat(batch_tensors, dim=0).to(self.device)
        
        with torch.no_grad():
            out = self.model(batch)
            probs = F.softmax(out, dim=1).cpu().numpy()
            
        return probs

    def explain(self, img_path, out_path='explainers_outputs'):
        os.makedirs(out_path, exist_ok=True)
        img_pil = load_image(img_path).resize((224, 224))
        img_np = np.array(img_pil) # (224, 224, 3)

        # --- ROBUST SHAP STRATEGY ---
        # Instead of KernelExplainer on pixels (DimensionError/Slow), 
        # we use shap.Explainer with an Image Masker (PartitionExplainer).
        # This handles the 3D structure automatically.
        
        # 1. Define Masker (blurring out parts of the image)
        masker = shap.maskers.Image("inpaint_telea", img_np.shape)

        # 2. Define Explainer
        explainer = shap.Explainer(self.predict_wrapper, masker, output_names=["Lesion"])

        # 3. Calculate SHAP values
        # max_evals controls speed. 500 is a decent trade-off.
        shap_values = explainer(np.expand_dims(img_np, 0), max_evals=500, batch_size=10, outputs=shap.Explanation.argsort.flip[:1])

        # shap_values.values is (1, 224, 224, 3, 1) usually (Batch, H, W, C, Class)
        s_vals = shap_values.values[0] # (224, 224, 3, 1)
        
        # If multiple classes were computed, take top 1
        if s_vals.ndim == 4: 
            s_vals = s_vals[:, :, :, 0] # (224, 224, 3)

        # Sum absolute importance across channels -> (224, 224)
        s_vals_2d = np.sum(s_vals, axis=2)

        # --- Visualization ---
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img_pil)
        
        # Normalize for vivid colors
        max_val = np.max(np.abs(s_vals_2d))
        if max_val == 0: max_val = 1e-9
        
        # Use 'seismic' (Blue-White-Red) or 'jet'
        # We use 'jet' for heatmap style or 'seismic' for +/-
        # Let's use a clear Red overlay for positive importance
        
        # Create a custom heatmap
        heatmap = plt.get_cmap('seismic')( (s_vals_2d + max_val) / (2*max_val) )
        
        # Apply alpha based on intensity (so 0 importance is transparent)
        importance_map = np.abs(s_vals_2d) / max_val
        heatmap[:, :, 3] = importance_map * 0.7 # Set alpha
        
        ax.imshow(heatmap)
        ax.axis('off')
        
        base = os.path.splitext(os.path.basename(img_path))[0]
        save_path = os.path.join(out_path, f"{base}_shap.png")
        
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=150)
        plt.close(fig)
        
        print(f"[SHAP] Saved to {save_path}")
        return save_path

if __name__ == "__main__":
    TEST_IMG = r"E:\00. Master's in AI\Sem 1\AI\Project\code new\archive\HAM10000_images_part_1\ISIC_0024306.jpg"
    
    print("Processing:", TEST_IMG)

    print("\n--- Running Grad-CAM ---")
    grad = GradCAMExplainer()
    grad.run_on_image(TEST_IMG)

    print("\n--- Running LIME ---")
    lime_exp = LIMEExplainerWrapper()
    lime_exp.explain(TEST_IMG)

    print("\n--- Running SHAP (Using PartitionExplainer for Images) ---")
    shap_exp = SHAPExplainer()
    shap_exp.explain(TEST_IMG)
    
    print("\nDone! Check 'explainers_outputs' folder.")