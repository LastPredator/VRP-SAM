import torch
import torch.nn as nn
from segment_anything import sam_model_registry
from .vrp_encoder import VRPEncoder

class SAMWrapper(nn.Module):
    def __init__(self, sam_checkpoint, model_type="vit_h", freeze_sam=True):
        super().__init__()
        
        # Load SAM
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        
        # Freeze SAM parameters
        if freeze_sam:
            for param in self.sam.parameters():
                param.requires_grad = False
                
        # VRP Encoder
        self.vrp_encoder = VRPEncoder(prompt_dim=256)
        
    def forward(self, target_images, ref_images, ref_masks):
        # target_images: (B, 3, 1024, 1024) - Preprocessed for SAM
        # ref_images: (B, 3, H, W)
        # ref_masks: (B, 1, H, W)
        
        # 1. Get Image Embeddings for Target
        with torch.no_grad():
            # SAM expects input as a list of dicts or batched tensor if resized
            # Here we assume target_images are already resized and normalized
            # But SAM's image encoder expects (B, 3, 1024, 1024)
            image_embeddings = self.sam.image_encoder(target_images)
            
        # 2. Get Prompt Embeddings from Reference
        # If multiple references (shots > 1), we might average them or handle them as multiple points
        # For now, assume B includes the shot dimension or we average here.
        # If ref_images is (B, Shots, 3, H, W), we flatten to (B*Shots, ...) then average back
        
        # If ref_images is (B, Shots, 3, H, W), we flatten to (B*Shots, ...) then average back
        
        if ref_images.dim() == 5: # (B, Shots, 3, H, W)
            b, s, c, h, w = ref_images.shape
            ref_images_flat = ref_images.view(b*s, c, h, w)
            ref_masks_flat = ref_masks.view(b*s, 1, h, w)
            
            prompt_embeddings = self.vrp_encoder(ref_images_flat, ref_masks_flat) # (B*S, 1, 256)
            prompt_embeddings = prompt_embeddings.view(b, s, 1, 256)
            
            # Average embeddings across shots
            sparse_embeddings = prompt_embeddings.mean(dim=1) # (B, 1, 256)
        else:
            sparse_embeddings = self.vrp_encoder(ref_images, ref_masks) # (B, 1, 256)

        dense_embeddings = torch.zeros(
            sparse_embeddings.shape[0], 256, 64, 64, 
            device=sparse_embeddings.device
        ) # No dense prompt for now
        
        dense_embeddings = torch.zeros(
            sparse_embeddings.shape[0], 256, 64, 64, 
            device=sparse_embeddings.device
        ) # No dense prompt for now
        
        # 3. Decode Masks
        # SAM mask decoder expects:
        # image_embeddings: (B, 256, 64, 64)
        # image_pe: (1, 256, 64, 64)
        # sparse_prompt_embeddings: (B, N, 256)
        # dense_prompt_embeddings: (B, 256, 64, 64)
        # multimask_output: True/False
        
        low_res_masks, iou_predictions = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False, # We want the best single mask
        )
        
        # Upscale masks to original size (1024x1024)
        masks = self.sam.postprocess_masks(
            low_res_masks,
            input_size=(1024, 1024),
            original_size=(1024, 1024),
        )
        
        return masks, iou_predictions
