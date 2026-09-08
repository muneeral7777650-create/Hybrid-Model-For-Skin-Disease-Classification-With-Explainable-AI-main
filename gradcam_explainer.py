import os
import torch
import numpy as np
import cv2
import torch.nn.functional as F
from PIL import Image
# Import shared utilities
from utils import load_model, load_image, preprocess, DEVICE

class GradCAMExplainer:
    def __init__(self, device=DEVICE):
        self.device = torch.device(device)
        self.model = load_model(self.device)
        self.target_layer_name = self._find_last_conv(self.model)

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

    def run_on_image(self, img_path, out_path='efficientNet_explainers_outputs'):
        os.makedirs(out_path, exist_ok=True)
        img_pil = load_image(img_path).resize((224, 224))
        img_tensor = preprocess(img_pil).unsqueeze(0).to(self.device)

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
        
        try:
            bh = target_module.register_full_backward_hook(backward_hook)
        except AttributeError:
            bh = target_module.register_backward_hook(backward_hook)

        # Forward
        outputs = self.model(img_tensor)
        probs = F.softmax(outputs, dim=1)
        class_idx = torch.argmax(probs, dim=1).item()

        # Backward
        loss = outputs[0, class_idx]
        self.model.zero_grad()
        loss.backward()

        if grads is None or features is None:
            print("GradCAM Error: Hooks failed.")
            fh.remove(); bh.remove()
            return

        # Generate CAM
        pooled_grads = torch.mean(grads, dim=[0, 2, 3])
        cam = torch.zeros(features.shape[2:], dtype=torch.float32).to(self.device)
        for i in range(features.shape[1]):
            cam += pooled_grads[i] * features[0, i, :, :]
        
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-9)
        cam_np = cam.cpu().numpy()

        # Visuals
        cam_resized = cv2.resize(cam_np, (224, 224))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        original_img = np.array(img_pil)
        blended = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)

        mask = cam_resized > 0.15
        final_img = original_img.copy()
        final_img[mask] = blended[mask]

        # Draw Circle
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(cam_resized)
        cv2.circle(final_img, maxLoc, 20, (0, 255, 0), 2)
        cv2.circle(final_img, maxLoc, 22, (255, 255, 255), 1)

        base = os.path.splitext(os.path.basename(img_path))[0]
        save_path = os.path.join(out_path, f"{base}_gradcam.png")
        Image.fromarray(final_img).save(save_path)
        
        fh.remove(); bh.remove()
        print(f"[Grad-CAM] Saved to {save_path}")

#for 1 image testing
# if __name__ == "__main__":
#     TEST_IMG = r"E:\00. Master's in AI\Sem 1\AI\Project\code new\archive\HAM10000_images_part_1\ISIC_0024306.jpg"
#     explainer = GradCAMExplainer()
#     explainer.run_on_image(TEST_IMG)

#for 5 images testing
if __name__ == "__main__":
    folder = r"E:\00. Master's in AI\Sem 1\AI\Project\code new\archive\HAM10000_images_part_1"

    # Take first 5 images from folder
    image_paths = []
    for f in os.listdir(folder):
        if f.lower().endswith((".jpg", ".png", ".jpeg")):
            image_paths.append(os.path.join(folder, f))
        if len(image_paths) == 5:
            break

    explainer = GradCAMExplainer()

    for img_path in image_paths:
        print(f"Running Grad-CAM on: {img_path}")
        explainer.run_on_image(img_path)

#for all images testing
# if __name__ == "__main__":
#     folder = r"E:\00. Master's in AI\Sem 1\AI\Project\code new\archive\HAM10000_images_part_1"

#     images = [os.path.join(folder, f) 
#               for f in os.listdir(folder) 
#               if f.lower().endswith((".jpg", ".png", ".jpeg"))]

#     explainer = GradCAMExplainer()

#     for img_path in images:
#         explainer.run_on_image(img_path)

