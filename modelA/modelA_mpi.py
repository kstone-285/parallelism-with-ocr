import ipyparallel as ipp
import os
import glob
import time
import sys

# Worker: 단일 이미지 처리 (각 엔진에서 실행)
def process_image(image_path):
    """단일 이미지 전처리 및 OCR 수행"""
    # 엔진별 독립 환경 구성을 위한 임포트
    import cv2
    import pytesseract
    from PIL import Image

    try:
        # Stage 1: 전처리
        img = cv2.imread(image_path)
        if img is None:
            return image_path, "ERROR: Cannot read image"

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binary_img = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        pil_img = Image.fromarray(binary_img)

        # Stage 2: OCR
        config = "--psm 6 --oem 1"
        text = pytesseract.image_to_string(pil_img, lang='eng', config=config)

        return image_path, text.strip()
    except Exception as e:
        return image_path, f"ERROR: {e}"

# Main: ipyparallel 클러스터 제어 및 실행
if __name__ == "__main__":
    try:
        # 1. 클러스터 연결
        print("ipyparallel 클러스터에 연결 시도...")
        client = ipp.Client()
        dview = client[:]  # 모든 엔진 선택
        print(f"연결 성공! {len(dview)}개의 엔진(워커)을 사용합니다.")
    except Exception as e:
        print("오류: 클러스터 연결 실패. (터미널: ipcluster start -n 4 --mpi)", file=sys.stderr)
        sys.exit(1)

    print(f"--- ipyparallel 기반 OCR 처리 시작 ---")
    start_time = time.time()

    # 2. 이미지 로드
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    images_dir = os.path.join(project_root, "dataset", "training_data", "images")
    image_paths = sorted(glob.glob(os.path.join(images_dir, "*.png")) + glob.glob(os.path.join(images_dir, "*.jpg")))
    
    if not image_paths:
        print("오류: 이미지를 찾을 수 없습니다.", file=sys.stderr)
        sys.exit(1)

    N_IMAGES = len(image_paths)
    print(f"총 {N_IMAGES}개의 이미지를 처리합니다.")

    # 3. 작업 분산 실행 (Map Sync)
    print("모든 엔진에 작업을 분배하고 결과를 기다리는 중...")
    results_list = dview.map_sync(process_image, image_paths)

    # 4. 결과 집계 및 리포트
    total_time = time.time() - start_time
    results_dict = dict(results_list)
    
    print("\n" + "="*50)
    print("--- 모든 작업 완료 ---")
    print(f"총 소요 시간: {total_time:.4f} 초")
    print(f"총 처리 이미지: {len(results_dict)} / {N_IMAGES} 개")
    if total_time > 0:
        print(f"평균 처리량: {N_IMAGES / total_time:.2f} images/sec")
    print("="*50)

    # 결과 샘플 출력
    print("\n--- 결과 샘플 (최대 5개) ---")
    count = 0
    for path, text in results_dict.items():
        print(f"FILE: {os.path.basename(path)}")
        print(f"TEXT: {text[:70].replace(chr(10), ' ')}...")
        print("-" * 20)
        count += 1
        if count >= 5:
            break