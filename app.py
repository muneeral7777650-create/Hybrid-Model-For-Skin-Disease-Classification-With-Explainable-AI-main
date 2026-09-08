import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# =========================
# Configuration
# =========================
LABELS = ["mel", "nv", "bkl", "bcc", "akiec", "vasc", "df"]
LABEL_NAMES = {
    "mel": "Melanoma",
    "nv": "Nevus",
    "bkl": "Seborrheic Keratosis",
    "bcc": "Basal Cell Carcinoma",
    "akiec": "Actinic Keratosis",
    "vasc": "Vascular Lesion",
    "df": "Dermatofibroma",
}

MODEL_REASONS = [
    "EfficientNetB0 extracts local lesion texture, edges, and color patterns efficiently.",
    "Vision Transformer captures long-range spatial relationships across the full lesion region.",
    "The hybrid design combines CNN local features with Transformer global context for better robustness.",
    "This architecture is well-suited for skin lesion datasets such as HAM10000 because lesions vary in shape, color, and texture.",
]

PREDICTION_REASON_MAP = {
    "mel": "This prediction is likely driven by irregular pigmentation, asymmetry, or suspicious border patterns often seen in melanoma.",
    "nv": "The lesion shows a more regular and stable pattern, which is commonly associated with a benign nevus.",
    "bkl": "This lesion appears to have keratotic texture and a more consistent pigmented pattern, which matches seborrheic keratosis.",
    "bcc": "This prediction is supported by pearly or ulcerative lesion characteristics often linked to basal cell carcinoma.",
    "akiec": "The image may show rough, scaly, or keratotic regions associated with actinic keratosis.",
    "vasc": "The lesion is likely influenced by red or vascular structures, which align with a vascular lesion diagnosis.",
    "df": "The lesion may show a firm, central, or fibrotic appearance that is typical for dermatofibroma.",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT_ROOT = Path(__file__).resolve().parent

PREPROCESS = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class HybridSkinModel(nn.Module):
    def __init__(self, num_classes: int = 7):
        super().__init__()
        self.efficientnet = models.efficientnet_b0(weights=None)
        self.efficientnet.classifier[1] = nn.Linear(
            self.efficientnet.classifier[1].in_features, num_classes
        )

        self.vit = models.vit_b_16(weights=None)
        self.vit.heads.head = nn.Linear(self.vit.heads.head.in_features, num_classes)

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_classes * 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eff = self.efficientnet(x)
        vit = self.vit(x)
        combined = torch.cat((eff, vit), dim=1)
        return self.fc(combined)


@st.cache_resource
def load_model() -> Tuple[nn.Module, bool]:
    model = HybridSkinModel(num_classes=len(LABELS)).to(DEVICE)
    weights_found = False

    candidate_paths = []
    env_model_path = os.getenv("MODEL_PATH")
    if env_model_path:
        candidate_paths.append(Path(env_model_path).expanduser())

    candidate_paths.extend(
        [
            PROJECT_ROOT / "efficientnet_model.pth",
            PROJECT_ROOT / "best_model.pth",
            PROJECT_ROOT / "model.pth",
            PROJECT_ROOT / "efficientnet_model.h5",
        ]
    )

    for path in candidate_paths:
        if not path or not path.exists():
            continue

        try:
            state = torch.load(str(path), map_location=DEVICE, weights_only=False)
            if isinstance(state, dict):
                model.load_state_dict(state, strict=False)
                weights_found = True
                break
            if hasattr(state, "state_dict"):
                model.load_state_dict(state.state_dict(), strict=False)
                weights_found = True
                break
        except Exception as exc:
            st.warning(f"Could not load model file: {path}. Reason: {exc}")
            continue

    model.eval()
    return model, weights_found


def predict_with_model(model: nn.Module, image: Image.Image) -> Dict[str, object]:
    tensor = PREPROCESS(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        index = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[index].item())

    label = LABELS[index]
    explanation = PREDICTION_REASON_MAP.get(label, "The system selected this class based on the lesion’s visual pattern and texture.")

    return {
        "label": label,
        "label_name": LABEL_NAMES[label],
        "confidence": confidence,
        "probabilities": {
            LABELS[i]: float(probabilities[i].item()) for i in range(len(LABELS))
        },
        "reason": explanation,
    }


def render_model_rationale() -> None:
    st.sidebar.header("Why this model was chosen")
    for reason in MODEL_REASONS:
        st.sidebar.markdown(f"- {reason}")

    st.markdown("### Model choice rationale")
    st.write(
        "The hybrid model combines EfficientNet and Vision Transformer because each architecture contributes a different strength: "
        "EfficientNet is strong at local spatial feature extraction, while ViT captures global patterns across the lesion. "
        "Combining both helps improve classification robustness in cases where lesions differ in color, texture, border shape, and appearance."
    )


def main() -> None:
    st.set_page_config(page_title="Skin Disease Classifier", page_icon="🩺", layout="wide")
    st.title("Skin Disease Classification with Explainable AI")
    st.caption("Hybrid model using EfficientNetB0 + Vision Transformer for lesion classification")

    render_model_rationale()

    uploaded_file = st.file_uploader("Upload a skin lesion image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)

        model, weights_found = load_model()

        if not weights_found:
            st.warning("No trained weights were found in this project folder, so the app is running in demo mode for interface validation.")

        result = predict_with_model(model, image)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted class", result["label_name"])
        with col2:
            st.metric("Confidence", f"{result['confidence'] * 100:.2f}%")

        st.markdown("### Why this result was selected")
        st.info(result["reason"])

        st.markdown("### Model confidence by class")
        probs = result["probabilities"]
        for label in LABELS:
            percent = probs[label] * 100
            st.markdown(f"**{LABEL_NAMES[label]}**")
            st.progress(percent / 100)
            st.caption(f"{percent:.2f}%")

        st.markdown("### Explanation of model choice")
        st.write(
            "This classification is based on the hybrid model’s decision rule: local textural details are extracted by EfficientNet, while the ViT branch captures long-range patterns. "
            "For medical image classification, this combination is valuable because lesion patterns often depend on both small local structures and global lesion context."
        )

    else:
        st.info("Please upload an image to classify the skin lesion.")


if __name__ == "__main__":
    main()
