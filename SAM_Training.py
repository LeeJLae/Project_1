import os
import torch
from torch.optim import Adam
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from PIL import Image
import numpy as np
import json

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 경로 설정
IMAGE_DIR = "/home/jovyan/data-vol-1/dataset_SAM/train/image"
TXT_DIR = "/home/jovyan/data-vol-1/dataset_SAM/train/Class_txt"
JSON_DIR = "/home/jovyan/data-vol-1/dataset_SAM/train/measure"
OUTPUT_DIR = "/home/jovyan/data-vol-1/Result/2_Measure/SAM/test"
MODEL_SAVE_PATH = "/home/jovyan/data-vol-1/Result/2_Measure/SAM/BB_weight_SAM/SAM_FineTune.pth"  # 가중치 저장 경로

# SAM 모델 로드 (ViT-B)
sam_checkpoint_path = "/home/jovyan/SAM/sam_vit_b_01ec64.pth"
sam_model = sam_model_registry["vit_b"](checkpoint=sam_checkpoint_path).to(device)

# 학습 하이퍼파라미터
num_epochs = 10
learning_rate = 1e-4
batch_size = 4  # 원하는 배치 크기 설정
optimizer = Adam(sam_model.parameters(), lr=learning_rate)

# IoU 손실 계산 함수
def compute_iou_loss(pred_mask, true_mask):
    intersection = torch.logical_and(pred_mask, true_mask).float().sum()
    union = torch.logical_or(pred_mask, true_mask).float().sum()
    iou = intersection / (union + 1e-6)  # 분모가 0이 되지 않도록 작은 값을 더함
    return (1 - iou).requires_grad_(True)  # requires_grad 설정

def train_sam():
    mask_generator = SamAutomaticMaskGenerator(sam_model)
    total_images = len([f for f in os.listdir(IMAGE_DIR) if f.endswith((".jpg", ".png"))])
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        running_loss = 0.0
        images_processed = 0
        
        for i, image_file in enumerate(os.listdir(IMAGE_DIR)):
            if image_file.endswith(".jpg") or image_file.endswith(".png"):
                image_path = os.path.join(IMAGE_DIR, image_file)
                txt_path = os.path.join(TXT_DIR, image_file.replace(".jpg", ".txt").replace(".png", ".txt"))
                json_path = os.path.join(JSON_DIR, image_file.replace(".jpg", ".json").replace(".png", ".json"))
                
                image = Image.open(image_path).convert("RGB")
                bboxes = load_bboxes(txt_path)
                measurements = parse_json(json_path)
                
                image_np = np.array(image)
                segmented_image, measurements_dict = perform_segmentation_and_measurement(mask_generator, image_np, bboxes, measurements)
                
                # 손실 계산
                true_mask = np.zeros_like(segmented_image)  # 실제 마스크 데이터를 정의 필요
                loss = compute_iou_loss(torch.tensor(segmented_image).to(device), torch.tensor(true_mask).to(device))
                
                # 배치 손실과 학습 진행률 출력
                running_loss += loss.item()
                images_processed += 1
                
                # 배치 처리
                if images_processed % batch_size == 0 or images_processed == total_images:
                    avg_loss = running_loss / min(batch_size, images_processed)
                    completion_percent = 100 * images_processed / total_images
                    print(f"[{epoch + 1}/{num_epochs}] - Batch: {images_processed // batch_size} - Avg Loss: {avg_loss:.4f} - {completion_percent:.2f}% 완료")
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    running_loss = 0.0
                
                # 저장
                save_path = os.path.join(OUTPUT_DIR, f"segmented_{image_file}")
                Image.fromarray(segmented_image).save(save_path)

                output_json_path = os.path.join(OUTPUT_DIR, f"{image_file.replace('.jpg', '.json').replace('.png', '.json')}")
                with open(output_json_path, 'w') as f:
                    json.dump(measurements_dict, f)
        
        print(f"Epoch {epoch + 1} 완료")

    torch.save(sam_model.state_dict(), MODEL_SAVE_PATH)
    print(f"모델 가중치가 {MODEL_SAVE_PATH}에 저장되었습니다.")

def load_bboxes(txt_path):
    bboxes = []
    with open(txt_path, 'r') as f:
        for line in f:
            class_id, x, y, w, h = map(float, line.strip().split())
            bboxes.append({"class_id": int(class_id), "x": x, "y": y, "w": w, "h": h})
    return bboxes

def parse_json(json_path):
    with open(json_path, 'r') as f:
        measurements = json.load(f)
    return measurements

def perform_segmentation_and_measurement(mask_generator, image, bboxes, measurements):
    masks = mask_generator.generate(image)
    segmented_image = apply_masks_to_image(image, masks)
    
    measurements_dict = {}
    for bbox in bboxes:
        class_id = bbox["class_id"]
        x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
        measurements_dict[class_id] = {
            "position": (x, y),
            "width": w,
            "height": h,
            "measurements": measurements[class_id] if class_id < len(measurements) else 0
        }
    return segmented_image, measurements_dict

def apply_masks_to_image(image, masks):
    return image  # 현재는 입력 이미지를 그대로 반환


# 학습 시작
train_sam()
