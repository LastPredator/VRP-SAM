---
title: VRP-SAM CV Project
emoji: 🎯
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.28.0"
app_file: app.py
pinned: false
---

# VRP-SAM: Visual Reference Prompt Segment Anything

This is a demo application for VRP-SAM (Visual Reference Prompt Segment Anything Model), which allows you to segment objects in target images based on visual references.

## How to Use

1. **Upload Reference Image**: Upload an image containing the object you want to segment
2. **Upload Reference Mask**: Upload a binary mask highlighting the object in the reference image
3. **Upload Target Image**: Upload the image where you want to find and segment similar objects
4. **Click "Segment Target"**: The model will segment objects in the target image that match the reference

## Model Details

This demo uses:
- **SAM (Segment Anything Model)**: Base segmentation model from Meta AI
- **VRP Module**: Custom visual reference prompt module trained on COCO dataset

The model automatically downloads the required SAM weights on first run.

