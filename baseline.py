import cv2
import pytesseract
from PIL import Image
import time
import os
import glob

def process_single_image(image_path):
    """단일 이미지를 읽어 OCR을 수행하는 함수"""
    try:
        # Stage 1: Pre-processing
        preprocess_start = time.time()
        
        img = cv2.imread(image_path)
        if img is None:
            # 이미지를 못 읽었을 때도 딕셔너리 구조 반환
            return {"text": "", "times": {"preprocess": 0, "ocr": 0, "postprocess": 0}}

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 적응형 임계값 처리
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
        
        # Stage 3: Post-processing
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
        print(f"Skip error file: {image_path}")
        # [수정] 예외 발생 시에도 딕셔너리 구조를 반환하여 메인 루프 에러 방지
        return {"text": "", "times": {"preprocess": 0, "ocr": 0, "postprocess": 0}}

# --- 순차 처리 메인 루프 ---

if __name__ == "__main__":
    # 경로 설정 (다운로드 받은 폴더 경로 확인 필요)
    # 현재 파일 위치 기준 dataset/training_data/images 폴더를 탐색
    project_root = os.path.dirname(os.path.abspath(__file__))
    # 만약 이미지를 ./images에 두셨다면 아래 경로를 "./images"로 바꾸세요.
    images_dir = os.path.join(project_root, "dataset", "training_data", "images")
    
    image_paths = sorted(glob.glob(os.path.join(images_dir, "*.png")) + glob.glob(os.path.join(images_dir, "*.jpg")))

    if not image_paths:
        print(f"오류: 이미지를 찾을 수 없습니다. 경로를 확인해주세요: {images_dir}")
        # 테스트를 위해 빈 리스트가 아닐 때만 실행
    else:
        print(f"순차 처리(Baseline) 시작... 대상 이미지: {len(image_paths)}장")
        start_time = time.time()

        total_times = {"preprocess": 0, "ocr": 0, "postprocess": 0}
        results = []
        
        # 진행 상황을 보기 위해 enumerate 사용
        for i, path in enumerate(image_paths):
            result = process_single_image(path)
            
            # 결과가 유효한 경우에만 집계 (빈 텍스트도 처리는 된 것이므로 포함)
            results.append(result["text"])
            
            # 각 단계별 시간 누적
            for stage, time_taken in result["times"].items():
                total_times[stage] += time_taken
            
            # 100장마다 로그 출력 (진행상황 파악용)
            if (i + 1) % 100 == 0:
                print(f"... {i + 1}장 처리 완료 ({time.time() - start_time:.2f}초 경과)")

        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n[완료] 순차 처리(Baseline) 끝.")
        print(f"총 처리 이미지: {len(results)}장")
        print(f"총 소요 시간: {total_time:.2f} 초")
        
        if total_time > 0:
            print("\n[단계별 소요 시간 분석]")
            print(f"- 전처리 단계: {total_times['preprocess']:.2f}초 ({(total_times['preprocess']/total_time*100):.1f}%)")
            print(f"- OCR 단계   : {total_times['ocr']:.2f}초 ({(total_times['ocr']/total_time*100):.1f}%)")
            print(f"- 후처리 단계: {total_times['postprocess']:.2f}초 ({(total_times['postprocess']/total_time*100):.1f}%)")
            print("\n-> 예상대로 OCR 단계가 병목인지 확인해보세요.")