import os
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
# Import shared utilities
from utils import load_model, load_image, preprocess, DEVICE

# Try importing LIME and Scikit-Image
try:
    from lime import lime_image
    from skimage.segmentation import mark_boundaries
    import cv2
except ImportError:
    print("Error: Missing libraries. Please run: pip install lime scikit-image opencv-python")
    exit()

class LIMEExplainerWrapper:
    def __init__(self, device=DEVICE):
        self.device = torch.device(device)
        self.model = load_model(self.device)

    def predict_fn(self, images):
        """
        LIME passes a numpy array of images. We convert to Tensor -> Model -> Probs.
        """
        self.model.eval()
        batch = []
        for img in images:
            # LIME images are numpy arrays (H, W, 3)
            pil = Image.fromarray(img.astype('uint8'))
            t = preprocess(pil).unsqueeze(0)
            batch.append(t)
            
        batch = torch.cat(batch, dim=0).to(self.device)
        
        with torch.no_grad():
            out = self.model(batch)
            probs = F.softmax(out, dim=1).cpu().numpy()
            
        return probs

    def explain(self, img_path, out_path='efficientNet_explainers_outputs'):
        os.makedirs(out_path, exist_ok=True)
        
        # Load and resize image
        img_pil = load_image(img_path).resize((224, 224))
        img_np = np.array(img_pil)

        print(f"Running LIME on {os.path.basename(img_path)} (this usually takes 30-60 seconds)...")
        
        explainer = lime_image.LimeImageExplainer()
        
        # Run LIME
        # top_labels=1: focus on the predicted class
        # num_samples=1000: higher = more accurate, lower = faster
        explanation = explainer.explain_instance(
            img_np, 
            self.predict_fn, 
            top_labels=1, 
            hide_color=0, 
            num_samples=1000
        )

        top_label = explanation.top_labels[0]
        
        # Get image and mask
        # positive_only=True: Only show areas that contribute TO the disease
        # num_features=5: Only show the top 5 most important 'blobs'
        temp, mask = explanation.get_image_and_mask(
            top_label, 
            positive_only=True, 
            num_features=5, 
            hide_rest=False
        )

        # --- CUSTOM VISUALIZATION ---
        
        # 1. Create a Yellow Overlay
        overlay = img_np.copy()
        # Set pixels where mask is active to Yellow [R=255, G=255, B=0]
        overlay[mask == 1] = [255, 255, 0] 

        # 2. Blend Overlay with Original Image (Translucent Fill)
        alpha = 0.4 # Opacity of the yellow fill
        cv2_img = img_np.astype(np.float32)
        cv2_overlay = overlay.astype(np.float32)
        blended = cv2.addWeighted(cv2_img, 1 - alpha, cv2_overlay, alpha, 0).astype(np.uint8)

        # 3. Add Thick Boundaries (Contours)
        # mark_boundaries returns a float image (0-1), we convert back to uint8 (0-255)
        img_boundary = mark_boundaries(blended, mask, color=(1, 1, 0), mode='thick', outline_color=(1, 1, 0))
        final_img = (img_boundary * 255).astype(np.uint8)

        # Save
        base = os.path.splitext(os.path.basename(img_path))[0]
        save_path = os.path.join(out_path, f"{base}_lime.png")
        Image.fromarray(final_img).save(save_path)
        
        print(f"[LIME] Saved to {save_path}")

#for 1 image testing
# if __name__ == "__main__":
#     TEST_IMG = r"E:\00. Master's in AI\Sem 1\AI\Project\code new\archive\HAM10000_images_part_1\ISIC_0024306.jpg"
    
#     explainer = LIMEExplainerWrapper()
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

    explainer = LIMEExplainerWrapper()

    # Run LIME on 5 images
    for img_path in image_paths:
        print(f"\n===== Running LIME on {os.path.basename(img_path)} =====")
        explainer.explain(img_path)

#for all images testing
# if __name__ == "__main__":
#     folder = r"E:\00. Master's in AI\Sem 1\AI\Project\code new\archive\HAM10000_images_part_1"

#     images = [os.path.join(folder, f) 
#               for f in os.listdir(folder) 
#               if f.lower().endswith((".jpg", ".png", ".jpeg"))]

#     explainer = LIMEExplainerWrapper()

#     for img_path in images:
#         explainer.run_on_image(img_path)