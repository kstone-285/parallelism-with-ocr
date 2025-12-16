import ipyparallel as ipp
import os
import glob
import time
import sys

# Worker Functions (ipyparallel)
# Run: ipcluster start -n 7 --engines=mpi
def stage1_preprocess_worker(image_path):
    """S1: 이미지 전처리"""
    import cv2
    import os
    
    pid = os.getpid()
    if not os.path.exists(image_path):
        return pid, image_path, None, f"File not found"

    try:
        img = cv2.imread(image_path)
        if img is None:
            return pid, image_path, None, "Cannot read image"
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binary_img = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        return pid, image_path, binary_img, None
    except Exception as e:
        return pid, image_path, None, str(e)

def stage2_ocr_worker(item):
    """S2: OCR 수행"""
    import pytesseract
    from PIL import Image
    import os

    pid = os.getpid()
    image_path, binary_img = item

    try:
        pil_img = Image.fromarray(binary_img)
        config = "--psm 6 --oem 1"
        text = pytesseract.image_to_string(pil_img, lang='eng', config=config)
        return pid, image_path, text.strip(), None
    except Exception as e:
        return pid, image_path, f"ERROR: {e}", str(e)

def stage3_save_worker(item):
    """S3: 결과 반환"""
    import os
    pid = os.getpid()
    image_path, text = item
    return pid, (image_path, text)

# Main: Orchestrator
def main():
    # 1. 엔진 할당
    P_S1_PREPROCESS = 2  # 전처리 엔진 수
    P_S2_OCR = 3         # OCR 엔진 수 (가장 많이 할당)
    P_S3_SAVE = 1        # 결과 취합 엔진 수
 
    # 2. 클러스터 연결
    try:
        client = ipp.Client()
        all_engines = client.ids
        print(f"연결 성공! 총 {len(all_engines)}개의 엔진 사용 가능.")

        if len(all_engines) < P_S1_PREPROCESS + P_S2_OCR + P_S3_SAVE:
            print("오류: 요청한 엔진 수가 사용 가능한 엔진 수보다 많습니다.", file=sys.stderr)
            print(f"사용 가능: {len(all_engines)}, 요청: {P_S1_PREPROCESS + P_S2_OCR + P_S3_SAVE}", file=sys.stderr)
            return

        # 엔진 ID를 각 스테이지에 할당
        s1_engine_ids = all_engines[:P_S1_PREPROCESS]
        s2_engine_ids = all_engines[P_S1_PREPROCESS : P_S1_PREPROCESS + P_S2_OCR]
        s3_engine_ids = all_engines[P_S1_PREPROCESS + P_S2_OCR : P_S1_PREPROCESS + P_S2_OCR + P_S3_SAVE]
        s1_engines, s2_engines, s3_engines = client[s1_engine_ids], client[s2_engine_ids], client[s3_engine_ids]
        print(f"엔진 할당: S1(전처리)={s1_engines.targets}, S2(OCR)={s2_engines.targets}, S3(저장)={s3_engines.targets}")

    except Exception as e:
        print("="*60, file=sys.stderr)
        print("오류: ipyparallel 클러스터에 연결할 수 없습니다.", file=sys.stderr)
        print("이 스크립트를 실행하기 전에 터미널에서 클러스터를 시작해야 합니다.", file=sys.stderr)
        print("예: ipcluster start -n 7 --engines=mpi", file=sys.stderr)
        print(f"상세 정보: {e}", file=sys.stderr)
        print("="*60, file=sys.stderr)
        return

    # 3. 이미지 로드
    project_root = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(project_root, "dataset", "training_data", "images")
    image_paths = sorted(glob.glob(os.path.join(images_dir, "*.png")) + glob.glob(os.path.join(images_dir, "*.jpg")))

    if not image_paths:
        print("오류: 'images' 폴더에서 이미지를 찾을 수 없습니다.", file=sys.stderr)
        return

    N_IMAGES = len(image_paths)
    print(f"총 {N_IMAGES}개의 이미지를 처리합니다.")

    # 4. 파이프라인 실행
    print("\n--- 모델 B (MPI 파이프라인) 처리 시작 ---")
    pipeline_start_time = time.time()

    # Load Balanced View 설정
    lb_view_s1 = client.load_balanced_view(targets=s1_engine_ids)
    lb_view_s2 = client.load_balanced_view(targets=s2_engine_ids)
    lb_view_s3 = client.load_balanced_view(targets=s3_engine_ids)

    # S1 작업 제출 (Async)
    s1_async_results = [lb_view_s1.apply_async(stage1_preprocess_worker, path) for path in image_paths]
    
    s2_async_results = []
    s3_async_results = []
    final_results = {}
    
    # 파이프라인 관리 루프
    while s1_async_results:
        # Check S1
        for ar in s1_async_results:
            if ar.ready():
                pid, image_path, binary_img, error = ar.get()
                if error:
                    print(f"[S1->S2] 전처리 오류 ({os.path.basename(image_path)}): {error}")
                else:
                    # Submit to S2
                    s2_ar = lb_view_s2.apply_async(stage2_ocr_worker, (image_path, binary_img))
                    s2_async_results.append(s2_ar)
                s1_async_results.remove(ar)

        # Check S2
        for ar in s2_async_results:
            if ar.ready():
                pid, image_path, text, error = ar.get()
                if error:
                    print(f"[S2->S3] OCR 오류 ({os.path.basename(image_path)}): {error}")
                else:
                    # Submit to S3
                    s3_ar = lb_view_s3.apply_async(stage3_save_worker, (image_path, text))
                    s3_async_results.append(s3_ar)
                s2_async_results.remove(ar)

        # Check S3 (Collect)
        for ar in s3_async_results:
            if ar.ready():
                pid, (path, text) = ar.get()
                final_results[path] = text
                s3_async_results.remove(ar)
        
        time.sleep(0.01)

    # 잔여 작업 대기
    print("모든 작업 제출 완료. 남은 결과 수집 중...")
    client.wait(s2_async_results)
    for ar in s2_async_results:
        pid, image_path, text, error = ar.get()
        if not error:
            s3_ar = lb_view_s3.apply_async(stage3_save_worker, (image_path, text))
            s3_async_results.append(s3_ar)

    client.wait(s3_async_results)
    for ar in s3_async_results:
        pid, (path, text) = ar.get()
        final_results[path] = text

    pipeline_end_time = time.time()
    total_time = pipeline_end_time - pipeline_start_time

    # 5. 리포트
    print("\n" + "="*50)
    print("--- 모든 파이프라인 작업 완료 ---")
    print(f"총 소요 시간: {total_time:.4f} 초")
    print(f"총 처리 이미지: {len(final_results)} / {N_IMAGES} 개")
    if total_time > 0:
        print(f"평균 처리량: {N_IMAGES / total_time:.2f} images/sec")
    print("="*50)

    # 6. 샘플 출력
    print("\n--- 결과 샘플 (최대 5개) ---")
    count = 0
    for path, text in final_results.items():
        print(f"FILE: {os.path.basename(path)}")
        print(f"TEXT: {text[:70].replace(chr(10), ' ')}...")
        print("-" * 20)
        count += 1
        if count >= 5:
            break

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n사용자에 의해 중단되었습니다.")