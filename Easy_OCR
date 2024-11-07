import cv2
import os
import torch
import sys

# 경로 설정
MODULE_PATH = '/home/jovyan/STN/EasyOCR/easyocr'
CRAFT_WEIGHT_PATH = '/home/jovyan/jonglae2/OCR/OCR/craft_mlt_25k.pth'  # CRAFT 가중치 경로
RECOGNITION_WEIGHT_PATH = '/home/jovyan/jonglae2/OCR/OCR/latin_g2.pth'  # Recognition 가중치 경로
INPUT_IMAGE_FOLDER = '/home/jovyan/data-vol-1/ResultForOCR/mask/maskedimg'  # 입력 이미지 폴더
OUTPUT_FOLDER = '/home/jovyan/data-vol-1/Result/2_Measure/YOLO/WB/OCR/test_horizontal_Easyocr'  # OCR 결과 저장 폴더

sys.path.append(MODULE_PATH)
from detection import get_textbox, get_detector  # Detection 모듈 불러오기
from recognition import get_text, get_recognizer  # Recognition 모듈 불러오기

# CRAFT 및 OCR 모델 초기화
detector = get_detector(CRAFT_WEIGHT_PATH, device='cuda' if torch.cuda.is_available() else 'cpu')
recognizer, converter = get_recognizer(
    recog_network='generation1',
    network_params={'input_channel': 1, 'output_channel': 512, 'hidden_size': 256},
    character='0123456789',  # 숫자만 포함
    separator_list=[],
    dict_list=[],
    model_path=RECOGNITION_WEIGHT_PATH,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

def perform_ocr(image_path, output_path):
    """입력 이미지에 대한 OCR 수행 후 결과를 TXT 파일에 저장"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return

    # OCR 수행
    ocr_results = get_text(
        character='0123456789', imgH=image.shape[0], imgW=image.shape[1],
        recognizer=recognizer, converter=converter,
        image_list=[((0, 0), image)], device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # 결과를 텍스트 파일에 저장
    with open(output_path, 'w') as f:
        for result in ocr_results:
            text, confidence = result[1], result[2]
            f.write(f"{text}\n")

# 모든 이미지에 대해 OCR 수행
for filename in os.listdir(INPUT_IMAGE_FOLDER):
    if filename.endswith(('.jpg', '.png')):
        image_path = os.path.join(INPUT_IMAGE_FOLDER, filename)
        output_text_file = os.path.join(OUTPUT_FOLDER, f"{os.path.splitext(filename)[0]}.txt")

        perform_ocr(image_path, output_text_file)
        print(f"OCR 결과가 {output_text_file}에 저장되었습니다.")
