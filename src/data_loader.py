from pathlib import Path
from typing import Tuple, Dict

import random
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms.functional as F

class SalientObjectDataset(Dataset):

    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        img_size: Tuple[int, int] = (224, 224),
        augment: bool = False,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.img_size = img_size
        self.augment = augment
        
        self.image_paths = sorted(
            [p for p in self.images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        )
        if len(self.image_paths) == 0:
            raise RuntimeError(f"No image files found in {self.images_dir}")
            
        self.mask_paths = []
        for img_path in self.image_paths:
            mask_path = self.masks_dir / f"{img_path.stem}.png"
            if not mask_path.exists():
                # provo edhe me .jpg
                alt = self.masks_dir / f"{img_path.stem}.jpg"
                if alt.exists():
                    mask_path = alt
                else:
                    raise RuntimeError(f"Mask not found for image: {img_path.name}")
            self.mask_paths.append(mask_path)

    def __len__(self) -> int:
        return len(self.image_paths)

    def _augment(self, image, mask):
        if random.random() < 0.5:
            image = F.hflip(image)
            mask = F.hflip(mask)
        if random.random() < 0.3:
            factor = 0.8 + 0.4 * random.random()
            image = F.adjust_brightness(image, factor)

        return image, mask
        
    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # grayscale

        image = image.resize(self.img_size, resample=Image.BILINEAR)
        mask = mask.resize(self.img_size, resample=Image.NEAREST)

        image = F.to_tensor(image)  # shape [3, H, W]
        mask = F.to_tensor(mask)    # shape [1, H, W]

        if self.augment:
            image, mask = self._augment(image, mask)
            
        mask = (mask > 0.5).float()

        return image, mask

def create_dataloaders(
    data_root: str,
    img_size: Tuple[int, int] = (224, 224),
    batch_size: int = 8,
    val_split: float = 0.15,
    num_workers: int = 2,
    seed: int = 42,
) -> Dict[str, DataLoader]:
    
    data_root = Path(data_root)

    tr_images = data_root / "DUTS-TR" / "DUTS-TR-Image"
    tr_masks = data_root / "DUTS-TR" / "DUTS-TR-Mask"

    full_train_dataset = SalientObjectDataset(
        images_dir=str(tr_images),
        masks_dir=str(tr_masks),
        img_size=img_size,
        augment=True,   
    )
    val_size = int(len(full_train_dataset) * val_split)
    train_size = len(full_train_dataset) - val_size

    torch.manual_seed(seed)
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    te_images = data_root / "DUTS-TE" / "DUTS-TE-Image"
    te_masks = data_root / "DUTS-TE" / "DUTS-TE-Mask"

    test_dataset = SalientObjectDataset(
        images_dir=str(te_images),
        masks_dir=str(te_masks),
        img_size=img_size,
        augment=False,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader}

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    duts_root = project_root / "dataDUTS"

    loaders = create_dataloaders(
        data_root=str(duts_root),
        img_size=(224, 224),
        batch_size=4,
    )
    batch = next(iter(loaders["train"]))
    images, masks = batch
    print("Train batch shapes:", images.shape, masks.shape)
