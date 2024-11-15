import torch
from torch import nn, optim
from torchvision import transforms
from omegaconf import DictConfig, OmegaConf
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
import os
from PIL import Image
from tqdm import tqdm
from sam2.build_sam import build_sam2


# Hydra 초기화 상태 확인 및 정리
if GlobalHydra.instance().is_initialized():
    GlobalHydra.instance().clear()

# Hydra 초기화 및 구성 불러오기
try:
    with initialize(config_path="./sam2/configs", job_name="sam2_config"):
        cfg = compose(config_name="sam2.1_training/sam2.1_finetune.yaml")
    print("Configuration loaded successfully with Hydra!")
except Exception as e:
    print(f"Error loading configuration with Hydra: {e}")
    exit(1)

# 디버깅: 전체 설정 확인
print("Loaded configuration:")
print(OmegaConf.to_yaml(cfg))

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Step 1: Model Configuration
model_cfg = cfg.get("trainer").get("model")
if model_cfg is None:
    print("Error: 'trainer.model' configuration is missing!")
    exit(1)

# 디버깅: 모델 설정 확인
print("Model configuration details:")
print(OmegaConf.to_yaml(model_cfg))

# Checkpoint 경로
sam2_checkpoint = "/home/jovyan/SAM2_Para/sam2.1_hiera_large.pt"

# 모델 생성
try:
    print("Attempting to build the model...")
    model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    print("Model built successfully!")
except Exception as e:
    print(f"Error building the model: {e}")
    exit(1)

# Step 2: Dataset Preparation
original_images_dir = "/home/jovyan/backup/FOR_YOLO/0_Testset_Mea"
masked_images_dir = "/home/jovyan/backup/FOR_SAM/Train"
output_model_path = "/home/jovyan/Project/STEP2/SAM/Train_Result/sam2_finetuned.pth"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

class PairedImageDataset(torch.utils.data.Dataset):
    def __init__(self, original_dir, masked_dir, transform=None):
        self.original_dir = original_dir
        self.masked_dir = masked_dir
        self.transform = transform
        self.image_names = os.listdir(original_dir)
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        orig_image_path = os.path.join(self.original_dir, self.image_names[idx])
        mask_image_path = os.path.join(self.masked_dir, self.image_names[idx])
        
        orig_image = Image.open(orig_image_path).convert("RGB")
        mask_image = Image.open(mask_image_path).convert("RGB")
        
        if self.transform:
            orig_image = self.transform(orig_image)
            mask_image = self.transform(mask_image)
        
        return orig_image, mask_image

train_data = PairedImageDataset(original_images_dir, masked_images_dir, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=cfg.scratch.train_batch_size, shuffle=True)

# Step 3: Training Setup
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=cfg.scratch.base_lr, weight_decay=0.01)

num_epochs = cfg.trainer.max_epochs if "max_epochs" in cfg.trainer else 1

# Step 4: Fine-tune the Model
model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for original, masked in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        original, masked = original.to(device), masked.to(device)
        
        optimizer.zero_grad()
        outputs = model(original)
        loss = criterion(outputs, masked)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# Step 5: Save Fine-tuned Model
torch.save(model.state_dict(), output_model_path)
print("Fine-tuned model saved successfully!")
