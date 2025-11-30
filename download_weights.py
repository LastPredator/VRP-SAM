import os
import urllib.request
from pathlib import Path

def download_sam_weights():
    """Download SAM weights if not present."""
    weights_dir = Path("weights")
    weights_dir.mkdir(exist_ok=True)
    
    sam_checkpoint = weights_dir / "sam_vit_h_4b8939.pth"
    
    if not sam_checkpoint.exists():
        print("Downloading SAM weights...")
        url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        urllib.request.urlretrieve(url, sam_checkpoint)
        print(f"Downloaded SAM weights to {sam_checkpoint}")
    else:
        print(f"SAM weights already exist at {sam_checkpoint}")

if __name__ == "__main__":
    download_sam_weights()
