import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image
# Import shared utilities
from utils import load_model, load_image, preprocess, DEVICE

try:
    import shap
except ImportError:
    print("Error: Install shap: pip install shap")
    exit()

class SHAPExplainer:
    def __init__(self, device=DEVICE):
        self.device = torch.device(device)
        self.model = load_model(self.device)

    def predict_wrapper(self, images_numpy):
        # Handle SHAP input variations (list or numpy)
        if isinstance(images_numpy, list):
            images_numpy = np.array(images_numpy)
            
        # Ensure 0-255 uint8 range
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

    def explain(self, img_path, out_path='efficientNet_explainers_outputs'):
        os.makedirs(out_path, exist_ok=True)
        img_pil = load_image(img_path).resize((224, 224))
        img_np = np.array(img_pil)

        print("Running SHAP (PartitionExplainer)...")
        
        # 1. Define Masker
        masker = shap.maskers.Image("inpaint_telea", img_np.shape)

        # 2. FIX: Define all 7 Class Names so SHAP doesn't crash on indexing
        class_names = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

        # 3. Create Explainer with correct output names
        explainer = shap.Explainer(self.predict_wrapper, masker, output_names=class_names)

        # 4. Run estimation
        # We slice outputs to get the top 1 predicted class
        shap_values = explainer(
            np.expand_dims(img_np, 0), 
            max_evals=500, 
            batch_size=10, 
            outputs=shap.Explanation.argsort.flip[:1]
        )

        # 5. Extract values
        s_vals = shap_values.values[0] # (224, 224, 3, 1) usually
        
        # Handle dimensions (Batch, H, W, C, Class) -> (H, W, C)
        if s_vals.ndim == 4: 
            s_vals = s_vals[:, :, :, 0]

        # Sum channels to get 2D Heatmap
        s_vals_2d = np.sum(s_vals, axis=2)

        # 6. Visualization
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img_pil)
        
        max_val = np.max(np.abs(s_vals_2d))
        if max_val == 0: max_val = 1e-9
        
        # Create Heatmap (Seismic: Red=High Importance, Blue=Low)
        heatmap = plt.get_cmap('seismic')( (s_vals_2d + max_val) / (2*max_val) )
        importance_map = np.abs(s_vals_2d) / max_val
        heatmap[:, :, 3] = importance_map * 0.7 # Alpha
        
        ax.imshow(heatmap)
        ax.axis('off')
        
        base = os.path.splitext(os.path.basename(img_path))[0]
        save_path = os.path.join(out_path, f"{base}_shap.png")
        
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=150)
        plt.close(fig)
        
        print(f"[SHAP] Saved to {save_path}")

#for 1 image testing
# if __name__ == "__main__":
#     TEST_IMG = r"E:\00. Master's in AI\Sem 1\AI\Project\code new\archive\HAM10000_images_part_1\ISIC_0024306.jpg"
#     explainer = SHAPExplainer()
#     explainer.explain(TEST_IMG)

#for 5 images testing
if __name__ == "__main__":
    folder = r"E:\00. Master's in AI\Sem 1\AI\Project\code new\archive\HAM10000_images_part_1"

    # Collect 5 image paths
    image_paths = []
    for f in os.listdir(folder):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            image_paths.append(os.path.join(folder, f))
        if len(image_paths) == 5:
            break

    explainer = SHAPExplainer()

    # Run SHAP on the 5 selected images
    for img_path in image_paths:
        print(f"\n===== Running SHAP on {os.path.basename(img_path)} =====")
        explainer.explain(img_path)

#for all images testing
# if __name__ == "__main__":
#     folder = r"E:\00. Master's in AI\Sem 1\AI\Project\code new\archive\HAM10000_images_part_1"

#     images = [os.path.join(folder, f) 
#               for f in os.listdir(folder) 
#               if f.lower().endswith((".jpg", ".png", ".jpeg"))]

#     explainer = SHAPExplainer()

#     for img_path in images:
#         explainer.run_on_image(img_path)
