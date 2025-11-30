import torch
import torch.nn as nn
import torchvision.models as models

class VRPEncoder(nn.Module):
    def __init__(self, prompt_dim=256):
        super().__init__()
        # Load ResNet50
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Modify first layer to accept 4 channels (RGB + Mask)
        original_conv1 = resnet.conv1
        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Initialize new weights: copy RGB weights, initialize Mask weights to zero
        with torch.no_grad():
            self.conv1.weight[:, :3, :, :] = original_conv1.weight
            self.conv1.weight[:, 3, :, :] = 0
            
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        self.avgpool = resnet.avgpool
        
        # Projection head to match SAM's prompt dimension
        self.projection = nn.Linear(2048, prompt_dim)
        
    def forward(self, images, masks):
        # images: (B, 3, H, W)
        # masks: (B, 1, H, W) - binary mask of the reference object
        
        x = torch.cat([images, masks.float()], dim=1)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        embeddings = self.projection(x)
        
        # Reshape to (B, 1, prompt_dim) to act as a single sparse prompt (like a point)
        embeddings = embeddings.unsqueeze(1)
        
        return embeddings
