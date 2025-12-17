import cv2
import pytesseract
from PIL import Image
import time
import os
import glob
import multiprocessing

# Worker: 단일 이미지 통합 처리
def process_single_image(image_path):
    """이미지 전처리, OCR, 후처리를 순차 수행"""
    try:
        # Stage 1: 전처리
        preprocess_start = time.time()
        
        img = cv2.imread(image_path)
        if img is None:
            return {"text": "", "times": {"preprocess": 0, "ocr": 0, "postprocess": 0}}

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binary_img = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        preprocess_time = time.time() - preprocess_start
        
        # Stage 2: OCR
        ocr_start = time.time()
        pil_img = Image.fromarray(binary_img)
        config = "--psm 6 --oem 1"
        text = pytesseract.image_to_string(pil_img, lang='eng', config=config)
        ocr_time = time.time() - ocr_start
        
        # Stage 3: 후처리
        postprocess_start = time.time()
        text = text.strip()
        postprocess_time = time.time() - postprocess_start
        
        return {
            "text": text,
            "times": {
                "preprocess": preprocess_time,
                "ocr": ocr_time,
                "postprocess": postprocess_time
            }
        }
        
    except Exception as e:
        return ""

# Main: 병렬 처리 설정 및 실행
if __name__ == "__main__":
    
    # 1. 이미지 로드
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    images_dir = os.path.join(project_root, "dataset", "training_data", "images")
    image_paths = sorted(glob.glob(os.path.join(images_dir, "*.png")) + glob.glob(os.path.join(images_dir, "*.jpg")))

    if not image_paths:
        print(f"오류: '{images_dir}'에서 이미지를 찾을 수 없습니다.")
        exit()

    print(f"총 {len(image_paths)}개의 이미지 처리 시작...")
    
    # 2. 프로세스 준비
    cpu_count = multiprocessing.cpu_count()
    print(f"병렬 처리(Model A) 시작... (사용 가능 코어: {cpu_count}개)")
    
    start_time = time.time()

    # 3. 병렬 처리 실행 (Pool)
    with multiprocessing.Pool() as pool:
        results = pool.map(process_single_image, image_paths)

    end_time = time.time()
    total_time = end_time - start_time
    
    # 4. 결과 리포트
    total_times = {"preprocess": 0, "ocr": 0, "postprocess": 0}
    texts = []
    
    for result in results:
        if not result: continue
        texts.append(result["text"])
        for stage, time_taken in result["times"].items():
            total_times[stage] += time_taken
    
    print(f"\n병렬 처리(Model A) 완료. 총 {len(texts)}개 처리.")
    print(f"총 소요 시간: {total_time:.2f} 초")
    print("\n각 단계별 소요 시간:")
    print(f"- 전처리 단계: {total_times['preprocess']:.2f}초 ({(total_times['preprocess']/total_time*100):.1f}%)")
    print(f"- OCR 단계: {total_times['ocr']:.2f}초 ({(total_times['ocr']/total_time*100):.1f}%)")
    print(f"- 후처리 단계: {total_times['postprocess']:.2f}초 ({(total_times['postprocess']/total_time*100):.1f}%)")