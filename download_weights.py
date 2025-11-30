import os
import requests

def download_weights():
    url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    out_path = "weights/sam_vit_h_4b8939.pth"
    os.makedirs("weights", exist_ok=True)

    if os.path.exists(out_path):
        print("✔ Weights already downloaded.")
        return out_path

    print("⬇ Downloading SAM weights...")
    response = requests.get(url, stream=True)
    response.raise_for_status()  # fail fast if bad response

    with open(out_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print("✔ Download complete.")
    return out_path
