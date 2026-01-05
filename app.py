import streamlit as st
import torch
import torch.nn as nn
from torchvision.models import vgg16
from torchvision import transforms
from PIL import Image
import os
import gdown

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
WEIGHTS_PATH = os.path.join(MODEL_DIR, "VGG16_model.pth")

# ---------------- CONFIG ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# classes order should be same as (train_dataset.class_to_idx)
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
WEIGHTS_PATH = "VGG16_model.pth"

torch.set_grad_enabled(False)

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    """
    Load VGG16 model ONCE and cache it.
    """
    if not os.path.exists(WEIGHTS_PATH):
        st.error(f"Model weights not found: {WEIGHTS_PATH}")
        st.stop()

    # Load base VGG16 WITHOUT ImageNet weights
    model = vgg16(weights=None)

    # Replace classifier FIRST
    model.classifier = nn.Sequential(
        nn.Linear(25088, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, len(CLASSES))
    )

    # Load trained weights
    state_dict = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    return model

# ---------------- TRANSFORMS ----------------
@st.cache_resource
def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

# ---------------- PREDICT ----------------
def predict(model, image):
    image = get_transforms()(image).unsqueeze(0).to(DEVICE)
    outputs = model(image)
    pred = outputs.argmax(dim=1).item()
    return CLASSES[pred]

# ---------------- STREAMLIT UI ----------------
def main():
    st.set_page_config(page_title="TrashNet Classifier", page_icon="♻️")

    st.title("♻️ TrashNet Image Classifier")
    st.write(
        "Upload an image of trash (Glass, Paper, Cardboard, Plastic, Metal, Trash) "
        "and the model will classify it."
    )

    # ✅ LOAD MODEL ONCE HERE
    model = load_model()

    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)

        with col2:
            st.subheader("Classification Result")
            with st.spinner("Classifying..."):
                prediction = predict(model, image)
            st.success(f"**Predicted Class:** {prediction}")

if __name__ == "__main__":
    main()
