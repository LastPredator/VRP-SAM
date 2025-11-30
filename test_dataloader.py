import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.getcwd())

from vrp_sam.data.dataset import COCOVRPSAMDataset

def test_dataloader():
    root_dir = r"u:\Semester7\CV_Project\VRP-SAM-Project\data\coco2014_full"
    dataset = COCOVRPSAMDataset(root_dir=root_dir, set_name='train2014')
    
    print(f"Dataset size: {len(dataset)}")
    
    # Get a sample
    sample = dataset[0]
    
    target_image = sample['target_image']
    target_mask = sample['target_mask']
    ref_images = sample['ref_images']
    ref_masks = sample['ref_masks']
    class_id = sample['class_id']
    
    print(f"Target Image Shape: {target_image.shape}")
    print(f"Target Mask Shape: {target_mask.shape}")
    print(f"Ref Images Shape: {ref_images.shape}")
    print(f"Ref Masks Shape: {ref_masks.shape}")
    print(f"Class ID: {class_id}")
    
    # Check mask values
    print(f"Target Mask Unique Values: {torch.unique(target_mask)}")
    print(f"Ref Mask Unique Values: {torch.unique(ref_masks)}")
    
    if len(torch.unique(target_mask)) > 1:
        print("SUCCESS: Target mask contains object.")
    else:
        print("WARNING: Target mask is empty (all zeros). This might happen if the random class selected is not present in the mask file for some reason, or if loading failed.")

if __name__ == "__main__":
    test_dataloader()
