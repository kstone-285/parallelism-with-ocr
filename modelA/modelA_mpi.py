import ipyparallel as ipp
import os
import glob
import time
import sys

# ----------------------------------------------------------------------------
# 단일 이미지 처리 함수 (워커/엔진이 수행)
# 이 함수는 ipyparallel 엔진(워커 프로세스)에서 실행됩니다.
# ipcluster start -n 4 --engines=mpi
# ----------------------------------------------------------------------------
def process_image(image_path):
    """
    하나의 이미지에 대해 전처리 및 OCR을 모두 수행합니다.
    """
    # 각 엔진(워커)은 독립된 프로세스이므로, 필요한 모듈을 다시 임포트해야 합니다.
    import cv2
    import pytesseract
    from PIL import Image

    try:
        # 1. 전처리
        img = cv2.imread(image_path)
        if img is None:
            return image_path, "ERROR: Cannot read image"

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # type: ignore
        binary_img = cv2.adaptiveThreshold( # type: ignore
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, # type: ignore
            cv2.THRESH_BINARY, 11, 2 # type: ignore
        ) # type: ignore
        pil_img = Image.fromarray(binary_img) # type: ignore

        # 2. OCR
        config = "--psm 6 --oem 1"
        text = pytesseract.image_to_string(pil_img, lang='eng', config=config)

        return image_path, text.strip()
    except Exception as e:
        return image_path, f"ERROR: {e}"

# ----------------------------------------------------------------------------
# MAIN: ipyparallel 클러스터에 작업 분배
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        # 1. 실행 중인 ipyparallel 클러스터에 연결
        print("ipyparallel 클러스터에 연결 시도...")
        client = ipp.Client()
        dview = client[:]  # 사용 가능한 모든 엔진(워커)을 선택
        print(f"연결 성공! {len(dview)}개의 엔진(워커)을 사용합니다.")
    except Exception as e:
        print("="*60, file=sys.stderr)
        print("오류: ipyparallel 클러스터에 연결할 수 없습니다.", file=sys.stderr)
        print("이 스크립트를 실행하기 전에 터미널에서 클러스터를 시작해야 합니다.", file=sys.stderr)
        print("예: ipcluster start -n 4 --mpi", file=sys.stderr)
        print(f"상세 정보: {e}", file=sys.stderr)
        print("="*60, file=sys.stderr)
        sys.exit(1)

    print(f"--- ipyparallel 기반 OCR 처리 시작 ---")
    start_time = time.time()

    # 2. 이미지 경로 불러오기
    # 경로 설정 (다운로드 받은 폴더 경로 확인 필요)
    # 현재 파일 위치 기준 dataset/training_data/images 폴더를 탐색
    # 사용자가 지정한 절대 경로에서 이미지 파일을 찾습니다.
    project_root = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(project_root, "dataset", "training_data", "images")
    image_paths = sorted(glob.glob(os.path.join(images_dir, "*.png")) + glob.glob(os.path.join(images_dir, "*.jpg")))
    if not image_paths:
        print("오류: 'images' 폴더에서 이미지를 찾을 수 없습니다.", file=sys.stderr)
        sys.exit(1)

    N_IMAGES = len(image_paths)
    print(f"총 {N_IMAGES}개의 이미지를 처리합니다.")

    # 3. 작업 실행
    # dview.map_sync는 image_paths의 각 항목을 process_image 함수에 인자로 넘겨
    # 모든 엔진에 작업을 분산시키고, 모든 결과가 올 때까지 기다립니다.
    # multiprocessing.Pool.map과 사용법이 거의 동일합니다.
    print("모든 엔진에 작업을 분배하고 결과를 기다리는 중...")
    results_list = dview.map_sync(process_image, image_paths)

    # 4. 최종 결과 리포트
    total_time = time.time() - start_time

    # 결과를 리스트에서 딕셔너리로 변환
    results_dict = dict(results_list)
    
    print("\n" + "="*50)
    print("--- 모든 작업 완료 ---")
    print(f"총 소요 시간: {total_time:.4f} 초")
    print(f"총 처리 이미지: {len(results_dict)} / {N_IMAGES} 개")
    if total_time > 0:
        print(f"평균 처리량: {N_IMAGES / total_time:.2f} images/sec")
    print("="*50)

    # 5. 결과 샘플 출력
    print("\n--- 결과 샘플 (최대 5개) ---")
    count = 0
    for path, text in results_dict.items():
        print(f"FILE: {os.path.basename(path)}")
        print(f"TEXT: {text[:70].replace(chr(10), ' ')}...")
        print("-" * 20)
        count += 1
        if count >= 5:
            break