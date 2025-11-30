import streamlit as st
import warnings
warnings.filterwarnings("ignore", message="The pynvml package is deprecated")
import torch
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

# Download SAM weights if not present
from download_weights import download_weights
download_weights()

from vrp_sam.models.sam_wrapper import SAMWrapper

# Page config
st.set_page_config(page_title="VRP-SAM Demo", layout="wide")

st.title("VRP-SAM: Visual Reference Prompt Segment Anything")
st.markdown("""
This demo allows you to segment objects in a target image based on a visual reference.
1. Upload a **Reference Image** and its corresponding **Mask**.
2. Upload a **Target Image**.
3. The model will segment the object in the target image that semantically matches the reference.
""")

# Sidebar for model configuration
st.sidebar.header("Configuration")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
st.sidebar.write(f"Device: {device}")

sam_checkpoint = st.sidebar.text_input("SAM Checkpoint Path", "weights/sam_vit_h_4b8939.pth")
vrp_checkpoint = st.sidebar.text_input("VRP Checkpoint Path", "")

@st.cache_resource
def load_model(sam_ckpt, vrp_ckpt):
    model = SAMWrapper(sam_checkpoint=sam_ckpt, model_type="vit_h", freeze_sam=True)
    if vrp_ckpt and os.path.exists(vrp_ckpt):
        checkpoint = torch.load(vrp_ckpt, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.to(device)
    model.eval()
    return model

if os.path.exists(sam_checkpoint):
    try:
        model = load_model(sam_checkpoint, vrp_checkpoint)
        st.sidebar.success("Model loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")
        model = None
else:
    st.sidebar.warning("SAM weights not found. Please download them first.")
    model = None

# Main layout
col1, col2 = st.columns(2)

with col1:
    st.header("Reference")
    ref_image_file = st.file_uploader("Upload Reference Image", type=['jpg', 'jpeg', 'png'])
    ref_mask_file = st.file_uploader("Upload Reference Mask", type=['jpg', 'jpeg', 'png'])
    
    if ref_image_file:
        ref_image = Image.open(ref_image_file).convert("RGB")
        st.image(ref_image, caption="Reference Image", use_container_width=True)
        
    if ref_mask_file:
        ref_mask_img = Image.open(ref_mask_file).convert("L") # Grayscale
        st.image(ref_mask_img, caption="Reference Mask", use_container_width=True)

with col2:
    st.header("Target")
    target_image_file = st.file_uploader("Upload Target Image", type=['jpg', 'jpeg', 'png'])
    
    if target_image_file:
        target_image = Image.open(target_image_file).convert("RGB")
        st.image(target_image, caption="Target Image", use_container_width=True)

# Inference
if st.button("Segment Target"):
    if model is None:
        st.error("Model not loaded.")
    elif not ref_image_file or not ref_mask_file or not target_image_file:
        st.error("Please upload all required images.")
    else:
        with st.spinner("Running inference..."):
            # Preprocess
            # Resize to 1024x1024 for simplicity (or whatever model expects)
            # In dataset.py we used 1024x1024.
            
            # Reference
            ref_img_np = np.array(ref_image.resize((1024, 1024)))
            ref_mask_np = np.array(ref_mask_img.resize((1024, 1024), resample=Image.NEAREST))
            
            ref_img_tensor = torch.from_numpy(ref_img_np).permute(2, 0, 1).float().unsqueeze(0).to(device)
            ref_mask_tensor = torch.from_numpy(ref_mask_np).long().unsqueeze(0).unsqueeze(0).to(device)
            # Ensure mask is binary 0/1
            ref_mask_tensor = (ref_mask_tensor > 128).long()
            
            # Target
            target_img_np = np.array(target_image.resize((1024, 1024)))
            target_img_tensor = torch.from_numpy(target_img_np).permute(2, 0, 1).float().unsqueeze(0).to(device)
            
            # Inference
            with torch.no_grad():
                pred_masks, _ = model(target_img_tensor, ref_img_tensor, ref_mask_tensor)
                
            # Post-process result
            pred_mask = pred_masks[0, 0].cpu().numpy()
            pred_mask = (pred_mask > 0).astype(np.uint8) * 255
            
            # Overlay
            result_image = target_img_np.copy()
            # Green overlay
            color = np.array([0, 255, 0], dtype=np.uint8)
            alpha = 0.5
            mask_bool = pred_mask > 128
            
            result_image[mask_bool] = result_image[mask_bool] * (1 - alpha) + color * alpha
            
            st.header("Result")
            st.image(result_image, caption="Segmentation Result", use_container_width=True)
