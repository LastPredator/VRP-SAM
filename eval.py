import os
import warnings
warnings.filterwarnings("ignore", message="The pynvml package is deprecated")
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from vrp_sam.models.sam_wrapper import SAMWrapper
from vrp_sam.data.dataset import COCOVRPSAMDataset

def compute_iou(pred_mask, target_mask):
    # pred_mask: (H, W) bool or 0/1
    # target_mask: (H, W) bool or 0/1
    
    intersection = (pred_mask & target_mask).sum()
    union = (pred_mask | target_mask).sum()
    
    if union == 0:
        return 1.0
    
    return intersection / union

def eval(args):
    # Setup device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')
        
    print(f"Using device: {device}")
    
    # Dataset
    dataset = COCOVRPSAMDataset(
        root_dir=args.data_root,
        set_name='val2014', # Use validation set
        shots=args.shots
    )
    
    # Subset if requested
    if args.num_samples > 0 and args.num_samples < len(dataset):
        indices = np.random.choice(len(dataset), args.num_samples, replace=False)
        dataset = torch.utils.data.Subset(dataset, indices)
        print(f"Evaluating on a random subset of {args.num_samples} images.")
    
    dataloader = DataLoader(
        dataset,
        batch_size=1, # Eval one by one for accurate metrics
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if args.workers > 0 else False
    )
    
    # Model
    model = SAMWrapper(
        sam_checkpoint=args.sam_weights,
        model_type="vit_h",
        freeze_sam=True
    )
    
    # Load checkpoint if provided
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
    model.to(device)
    model.eval()
    
    ious = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            target_images = batch['target_image'].to(device, non_blocking=True)
            target_masks = batch['target_mask'].to(device, non_blocking=True)
            ref_images = batch['ref_images'].to(device, non_blocking=True)
            ref_masks = batch['ref_masks'].to(device, non_blocking=True)
            
            # Forward with AMP
            with torch.amp.autocast('cuda'):
                pred_masks, _ = model(target_images, ref_images, ref_masks)
            
            # Post-process
            # pred_masks is (B, 1, H, W) logits
            pred_masks = (pred_masks > 0.0).float()
            
            # Compute IoU
            for i in range(pred_masks.shape[0]):
                iou = compute_iou(
                    pred_masks[i, 0].cpu().numpy().astype(bool),
                    target_masks[i].cpu().numpy().astype(bool)
                )
                ious.append(iou)
                
    mean_iou = np.mean(ious)
    print(f"Mean IoU: {mean_iou:.4f} (over {len(ious)} samples)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data/coco2014_full')
    parser.add_argument('--sam_weights', type=str, default='weights/sam_vit_h_4b8939.pth')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--shots', type=int, default=1)
    parser.add_argument('--num_samples', type=int, default=100, help="Number of samples to evaluate on. 0 for full set.")
    
    args = parser.parse_args()
    
    eval(args)
