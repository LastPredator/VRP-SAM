import sys
import os
import torch
from vrp_sam.data.dataset import COCOVRPSAMDataset

def test_dataset():
    # Adjust path to where your data is located
    root_dir = "data/coco2014_full"
    
    print(f"Initializing dataset from {root_dir}...")
    try:
        dataset = COCOVRPSAMDataset(root_dir=root_dir, set_name='train2014', shots=1)
        print(f"Dataset size: {len(dataset)}")
        
        sample = dataset[0]
        print("Sample keys:", sample.keys())
        print("Target Image Shape:", sample['target_image'].shape)
        print("Target Mask Shape:", sample['target_mask'].shape)
        print("Ref Images Shape:", sample['ref_images'].shape)
        print("Ref Masks Shape:", sample['ref_masks'].shape)
        print("Class ID:", sample['class_id'])
        
        print("Data loading verification successful!")
    except Exception as e:
        print(f"Data loading failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dataset()
