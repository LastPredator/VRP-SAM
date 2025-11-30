import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from pycocotools.coco import COCO

class COCOVRPSAMDataset(Dataset):
    """Dataset that loads images and pre‑computed semantic masks (PNG) from the
    COCO ``annotations/<set_name>`` folder. The PNG masks contain a background value
    0 and sequential class indices (1, 2, …) that correspond to the sorted COCO
    category IDs. This implementation builds a mapping ``cat_id -> mask_value`` so
    that a binary mask for the randomly selected target class can be extracted.
    """

    def __init__(self, root_dir, set_name="train2014", transform=None, shots=1):
        self.root_dir = root_dir
        self.set_name = set_name
        self.transform = transform
        self.shots = shots

        # Directories
        self.img_dir = os.path.join(root_dir, "images", set_name)
        self.ann_file = os.path.join(root_dir, "annotations", f"instances_{set_name}.json")
        self.mask_dir = os.path.join(root_dir, "annotations", set_name)  # PNG masks

        # COCO API
        self.coco = COCO(self.ann_file)
        self.cat_ids = self.coco.getCatIds()
        # Mapping from COCO category id to the integer value stored in the PNG mask
        # PNG masks use 0 for background and then 1‑N for categories in sorted order
        self.cat_id_to_mask_val = {cat_id: idx for idx, cat_id in enumerate(sorted(self.cat_ids), start=1)}

        # Build image ↔ category indexes
        self.cat_to_img = {}
        self.img_ids = []
        for cat_id in self.cat_ids:
            img_ids = self.coco.getImgIds(catIds=[cat_id])
            if img_ids:
                self.cat_to_img[cat_id] = img_ids
                self.img_ids.extend(img_ids)
        self.img_ids = list(set(self.img_ids))

        # Pre‑compute which categories appear in each image (speeds up __getitem__)
        self.img_to_cats = {}
        for img_id in self.img_ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
            anns = self.coco.loadAnns(ann_ids)
            cats = list({ann["category_id"] for ann in anns})
            self.img_to_cats[img_id] = cats

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        # ----- Target image ---------------------------------------------------
        target_img_id = self.img_ids[idx]
        target_info = self.coco.loadImgs(target_img_id)[0]
        target_path = os.path.join(self.img_dir, target_info["file_name"])
        target_image = cv2.imread(target_path)
        target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)

        # Choose a random category present in the image
        possible_cats = self.img_to_cats[target_img_id]
        if not possible_cats:
            return self.__getitem__((idx + 1) % len(self))
        target_cat_id = np.random.choice(possible_cats)
        mask_val = self.cat_id_to_mask_val.get(target_cat_id)

        # Load the pre‑computed semantic mask and extract binary mask for the class
        mask_filename = target_info["file_name"].replace('.jpg', '.png')
        mask_path = os.path.join(self.mask_dir, mask_filename)
        if os.path.exists(mask_path) and mask_val is not None:
            semantic_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            target_mask = (semantic_mask == mask_val).astype(np.uint8)
        else:
            # Fallback: generate mask from COCO annotations (rare)
            ann_ids = self.coco.getAnnIds(imgIds=target_img_id, catIds=target_cat_id, iscrowd=None)
            anns = self.coco.loadAnns(ann_ids)
            target_mask = np.zeros((target_info["height"], target_info["width"]), dtype=np.uint8)
            for ann in anns:
                target_mask = np.maximum(target_mask, self.coco.annToMask(ann))

        # ----- Reference images (support set) --------------------------------
        candidate_ref_ids = self.cat_to_img.get(target_cat_id, [])
        candidates = [i for i in candidate_ref_ids if i != target_img_id]
        if len(candidates) < self.shots:
            ref_ids = np.random.choice(candidate_ref_ids, self.shots, replace=True)
        else:
            ref_ids = np.random.choice(candidates, self.shots, replace=False)

        ref_images, ref_masks = [], []
        for ref_id in ref_ids:
            ref_info = self.coco.loadImgs(int(ref_id))[0]
            ref_path = os.path.join(self.img_dir, ref_info["file_name"])
            ref_img = cv2.imread(ref_path)
            ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)

            ref_mask_filename = ref_info["file_name"].replace('.jpg', '.png')
            ref_mask_path = os.path.join(self.mask_dir, ref_mask_filename)
            if os.path.exists(ref_mask_path) and mask_val is not None:
                ref_semantic = cv2.imread(ref_mask_path, cv2.IMREAD_GRAYSCALE)
                ref_mask = (ref_semantic == mask_val).astype(np.uint8)
            else:
                ann_ids = self.coco.getAnnIds(imgIds=ref_id, catIds=target_cat_id, iscrowd=None)
                anns = self.coco.loadAnns(ann_ids)
                ref_mask = np.zeros((ref_info["height"], ref_info["width"]), dtype=np.uint8)
                for ann in anns:
                    ref_mask = np.maximum(ref_mask, self.coco.annToMask(ann))

            # Apply optional transforms / resize to 1024×1024
            if self.transform:
                transformed = self.transform(image=ref_img, mask=ref_mask)
                ref_img = transformed["image"]
                ref_mask = transformed["mask"]
            else:
                ref_img = cv2.resize(ref_img, (1024, 1024))
                ref_mask = cv2.resize(ref_mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)
                ref_img = torch.from_numpy(ref_img).permute(2, 0, 1).float()
                ref_mask = torch.from_numpy(ref_mask).long()
            ref_images.append(ref_img)
            ref_masks.append(ref_mask)

        ref_images = torch.stack(ref_images)
        ref_masks = torch.stack(ref_masks)

        # ----- Transform target ------------------------------------------------
        if self.transform:
            transformed = self.transform(image=target_image, mask=target_mask)
            target_image = transformed["image"]
            target_mask = transformed["mask"]
        else:
            target_image = cv2.resize(target_image, (1024, 1024))
            target_mask = cv2.resize(target_mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)
            target_image = torch.from_numpy(target_image).permute(2, 0, 1).float()
            target_mask = torch.from_numpy(target_mask).long()

        return {
            "target_image": target_image,
            "target_mask": target_mask,
            "ref_images": ref_images,
            "ref_masks": ref_masks,
            "class_id": target_cat_id,
        }
