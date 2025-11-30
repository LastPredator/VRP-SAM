import os
import requests

def download_weights():
    url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    out_path = "weights/sam_vit_h_4b8939.pth"

    # Create weights folder if not exists
    os.makedirs("weights", exist_ok=True)

    # If weights already exist, skip download
    if os.path.exists(out_path):
        print("✔ Weights already downloaded.")
        return out_path

    # Download weights
    print("⬇ Downloading SAM weights...")
    response = requests.get(url, stream=True)

    if response.status_code != 200:
        raise Exception(f"Failed to download weights. Status code: {response.status_code}")

    with open(out_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print("✔ Download complete.")
    return out_path
