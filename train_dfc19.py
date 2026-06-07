# train_dfc19.py (SAM ile etiketlenmiş 100 bina maskesiyle eğitim)
import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torchvision.transforms as T
import segmentation_models_pytorch as smp
from albumentations.pytorch import ToTensorV2
import albumentations as A
from tqdm import tqdm

# --- Config ---
TRAIN_IMG_DIR = "dfc19/train/images"
TRAIN_MASK_DIR = "dfc19/train/masks"
BATCH_SIZE = 4
NUM_EPOCHS = 30
IMAGE_SIZE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Dataset Class ---
class DFCDataset(Dataset):
    def __init__(self, image_list, image_dir, mask_dir, transform=None):
        self.image_list = image_list
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_name = img_name.replace("RGB", "CLS")
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)  # bina=1, diğer=0

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask'].unsqueeze(0)  # [1, H, W]

        return image, mask

# --- Transforms ---
train_transform = A.Compose([
    A.RandomCrop(IMAGE_SIZE, IMAGE_SIZE),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2(),
])

# --- Sadece maskesi olanları seç ---
all_images = os.listdir(TRAIN_IMG_DIR)
all_masks = set([m.replace("_CLS", "_RGB") for m in os.listdir(TRAIN_MASK_DIR)])

labeled_images = [img for img in all_images if img in all_masks and img.endswith(".tif")]
print(f"✅ Etiketli görüntü sayısı: {len(labeled_images)}")

train_dataset = DFCDataset(labeled_images, TRAIN_IMG_DIR, TRAIN_MASK_DIR, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- Model ---
model = smp.Unet(encoder_name="resnet34", in_channels=3, classes=1)
model.to(DEVICE)

# --- Loss & Optimizer ---
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# --- Training Loop ---
print("\n🚀 SAM ile etiketlenmiş 100 maskeyle eğitim başlıyor...\n")
for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0
    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        preds = model(images)
        loss = loss_fn(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Loss: {train_loss:.4f}")

# --- Save model ---
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/unet_dfc19_sam_masks.pth")
print("\n✅ Eğitim tamamlandı! Model kaydedildi: checkpoints/unet_dfc19_sam_masks.pth")
