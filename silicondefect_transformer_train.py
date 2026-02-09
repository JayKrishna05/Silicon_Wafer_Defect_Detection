import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from timm import create_model
from sklearn.model_selection import train_test_split

# -------------------------------------------------
# CPU OPTIMIZATION 
# -------------------------------------------------
torch.set_num_threads(8)
torch.set_num_interop_threads(1)
torch.backends.mkldnn.enabled = True

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
PKL_PATH = r"E:\silicon\Training Code\dataset\LSWMD.pkl"
BATCH_SIZE = 16          
EPOCHS = 5
IMG_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT_PATH = "vit_wafer_checkpoint.pth"
FINAL_MODEL_PATH = "vit_wafer_final.pth"
PAUSE_FILE = "PAUSE.txt"

# -------------------------------------------------
# PAUSE HANDLER
# -------------------------------------------------
def wait_if_paused():
    while os.path.exists(PAUSE_FILE):
        print("⏸ Training paused (PAUSE.txt exists). Delete it to resume.")
        time.sleep(5)

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
print("Loading LSWMD.pkl ...")
df = pd.read_pickle(PKL_PATH)

def clean_label(x):
    while isinstance(x, (list, tuple, np.ndarray)):
        if len(x) == 0:
            return None
        x = x[0]
    if x is None:
        return None
    return str(x)

df["failureType"] = df["failureType"].apply(clean_label)
df = df[df["failureType"].notnull()]

print("Labeled samples:", len(df))
print(df["failureType"].value_counts())

# -------------------------------------------------
# LABEL ENCODING
# -------------------------------------------------
class_names = sorted(df["failureType"].unique())
label_to_id = {name: i for i, name in enumerate(class_names)}
labels = df["failureType"].map(label_to_id).values
NUM_CLASSES = len(class_names)

wafer_maps = df["waferMap"].values

# -------------------------------------------------
# TRAIN / VAL SPLIT
# -------------------------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    wafer_maps,
    labels,
    test_size=0.2,
    stratify=labels,
    random_state=42
)

# -------------------------------------------------
# DATASET
# -------------------------------------------------
class WaferDataset(Dataset):
    def __init__(self, maps, labels):
        self.maps = maps
        self.labels = labels

    def __len__(self):
        return len(self.maps)

    def __getitem__(self, idx):
        img = np.array(self.maps[idx], dtype=np.uint8)
        return img, int(self.labels[idx])

# -------------------------------------------------
# TRANSFORMS
# -------------------------------------------------
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.Grayscale(num_output_channels=3),
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3)
])

class TransformWrapper(Dataset):
    def __init__(self, base_ds, transform):
        self.base_ds = base_ds
        self.transform = transform

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        img, lbl = self.base_ds[idx]
        img = self.transform(img)
        return img, lbl

train_loader = DataLoader(
    TransformWrapper(WaferDataset(X_train, y_train), transform),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=False
)

val_loader = DataLoader(
    TransformWrapper(WaferDataset(X_val, y_val), transform),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False
)

# -------------------------------------------------
# MODEL
# -------------------------------------------------
model = create_model(
    "vit_tiny_patch16_224",
    pretrained=True,
    num_classes=NUM_CLASSES,
    img_size=IMG_SIZE
).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# -------------------------------------------------
# RESUME FROM CHECKPOINT
# -------------------------------------------------
start_epoch = 0
if os.path.exists(CHECKPOINT_PATH):
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    start_epoch = ckpt["epoch"] + 1
    print(f"Resuming from epoch {start_epoch}")

# -------------------------------------------------
# TRAIN LOOP (PAUSE + CTRL+C SAFE)
# -------------------------------------------------
try:
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_loss = 0.0

        for imgs, lbls in train_loader:
            wait_if_paused()

            imgs = imgs.to(DEVICE)
            lbls = lbls.to(DEVICE)

            optimizer.zero_grad()
            loss = criterion(model(imgs), lbls)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                preds = model(imgs.to(DEVICE)).argmax(1)
                correct += (preds == lbls.to(DEVICE)).sum().item()
                total += lbls.size(0)

        acc = correct / total
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss {total_loss:.4f} | Val Acc {acc:.4f}")

        # -------- SAVE CHECKPOINT --------
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "class_names": class_names,
            "img_size": IMG_SIZE
        }, CHECKPOINT_PATH)

        print("Checkpoint saved.")

except KeyboardInterrupt:
    print("Training interrupted safely. Checkpoint already saved.")

# -------------------------------------------------
# SAVE FINAL MODEL
# -------------------------------------------------
torch.save({
    "model_state": model.state_dict(),
    "class_names": class_names,
    "img_size": IMG_SIZE
}, FINAL_MODEL_PATH)

print("Final model saved.")


