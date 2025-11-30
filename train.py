import os
import warnings
warnings.filterwarnings("ignore", message="The pynvml package is deprecated")
import torch
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from vrp_sam.models.sam_wrapper import SAMWrapper
from vrp_sam.data.dataset import COCOVRPSAMDataset
from vrp_sam.utils.loss import CombinedLoss

def train(args):
    # Setup device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True # Optimize for fixed input sizes
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    print(f"Torch version: {torch.__version__}")
    
    # Create output directory
    os.makedirs(args.run_dir, exist_ok=True)
    
    # Dataset
    dataset = COCOVRPSAMDataset(
        root_dir=args.data_root,
        set_name='train2014',
        shots=args.shots
    )
    
    # Optimized DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=args.bsz,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if args.workers > 0 else False,
        prefetch_factor=2 if args.workers > 0 else None
    )
    
    # Model
    model = SAMWrapper(
        sam_checkpoint=args.sam_weights,
        model_type="vit_h",
        freeze_sam=True
    )
    model.to(device)
    
    # Compile VRP Encoder for speedup (Torch 2.0+)
    # Note: torch.compile requires Triton which is not fully supported on Windows yet.
    # Uncomment if running on Linux with Triton installed.
    # try:
    #     print("Compiling VRP Encoder with torch.compile...")
    #     model.vrp_encoder = torch.compile(model.vrp_encoder)
    # except Exception as e:
    #     print(f"Could not compile model: {e}")

    # Optimizer - Only optimize VRP Encoder
    optimizer = torch.optim.AdamW(
        model.vrp_encoder.parameters(),
        lr=args.lr,
        weight_decay=0.01
    )
    
    # Loss
    criterion = CombinedLoss()
    
    # Mixed Precision Scaler
    scaler = torch.amp.GradScaler('cuda')
    
    # Training Loop
    for epoch in range(args.epochs):
        model.train()
        model.sam.eval() 
        model.vrp_encoder.train()
        
        epoch_loss = 0
        # Create an infinite iterator or just restart if needed
        # But simpler: just loop over dataloader and break if steps exceeded
        # If steps_per_epoch is None, use full dataloader
        
        if args.steps_per_epoch > 0:
            total_steps = args.steps_per_epoch
        else:
            total_steps = len(dataloader)
            
        pbar = tqdm(total=total_steps, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        step_count = 0
        iter_loader = iter(dataloader)
        
        while step_count < total_steps:
            try:
                batch = next(iter_loader)
            except StopIteration:
                iter_loader = iter(dataloader)
                batch = next(iter_loader)
                
            target_images = batch['target_image'].to(device, non_blocking=True)
            target_masks = batch['target_mask'].to(device, non_blocking=True)
            ref_images = batch['ref_images'].to(device, non_blocking=True)
            ref_masks = batch['ref_masks'].to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda'):
                pred_masks, iou_preds = model(target_images, ref_images, ref_masks)
                loss = criterion(pred_masks, target_masks)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            pbar.update(1)
            
            step_count += 1
            
        pbar.close()
            
        avg_loss = epoch_loss / total_steps
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, os.path.join(args.run_dir, f"checkpoint_epoch_{epoch+1}.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data/coco2014_full')
    parser.add_argument('--sam_weights', type=str, default='weights/sam_vit_h_4b8939.pth')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--bsz', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--shots', type=int, default=1)
    parser.add_argument('--run_dir', type=str, default='runs/experiment_1')
    parser.add_argument('--steps_per_epoch', type=int, default=1000, help="Number of steps per epoch. Set to 0 for full dataset.")
    
    args = parser.parse_args()
    
    train(args)
