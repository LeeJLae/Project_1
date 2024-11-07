import os
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from PIL import Image
import numpy as np
import json

# 경로 설정
TEST_IMAGE_DIR = "/home/jovyan/data-vol-1/dataset_WB/test/images"
TEST_OUTPUT_DIR = "/home/jovyan/data-vol-1/Result/2_Measure/SAM/test"
MODEL_LOAD_PATH = "/home/jovyan/data-vol-1/Result/2_Measure/SAM/BB_weight_SAM/SAM_FineTune.pth"  # 로드할 가중치 경로
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# SAM 모델 로드 (ViT-B)
sam_model = sam_model_registry["vit_b"]()
state_dict = torch.load(MODEL_LOAD_PATH, map_location="cuda" if torch.cuda.is_available() else "cpu", weights_only=True)
sam_model.load_state_dict(state_dict)
sam_model.to(device)
mask_generator = SamAutomaticMaskGenerator(sam_model)

# 테스트 함수 정의
def test_sam():
    for image_file in os.listdir(TEST_IMAGE_DIR):
        if image_file.endswith(".jpg") or image_file.endswith(".png"):
            image_path = os.path.join(TEST_IMAGE_DIR, image_file)
            
            # 이미지 로드
            image = Image.open(image_path).convert("RGB")
            image_np = np.array(image)  # numpy 배열로 변환
            
            # 세그멘테이션 수행 및 계측값 계산
            segmented_image, measurements_dict = perform_segmentation_and_measurement_test(mask_generator, image_np)
            
            # 결과 저장
            save_path = os.path.join(TEST_OUTPUT_DIR, f"segmented_{image_file}")
            Image.fromarray(segmented_image).save(save_path)
            
            # 계측 값과 위치 좌표 저장 (JSON 형식)
            json_path = os.path.join(TEST_OUTPUT_DIR, f"{image_file.replace('.jpg', '.json').replace('.png', '.json')}")
            with open(json_path, 'w') as f:
                json.dump(measurements_dict, f)

def perform_segmentation_and_measurement_test(mask_generator, image):
    # SAM 모델을 사용하여 세그멘테이션 수행
    masks = mask_generator.generate(image)
    segmented_image = apply_masks_to_image(image, masks)
    measurements_dict = {}
    
    for i, mask in enumerate(masks):
        class_id = i  # 각 마스크에 대한 임의의 클래스 ID (필요에 따라 수정 가능)
        
        # 마스크의 좌표 정보 및 크기 계산
        bbox = mask["bbox"]
        x, y, w, h = bbox
        
        measurements_dict[class_id] = {
            "position": (x, y),
            "width": w,
            "height": h
        }
    return segmented_image, measurements_dict

def apply_masks_to_image(image, masks):
    return image  # 필요에 따라 마스크 오버레이 처리 추가

# 테스트 시작
test_sam()
