import multiprocessing
import os
import glob
import time
import sys
import cv2
import pytesseract
from PIL import Image

# Stage 1: 전처리 (Producer)
def stage1_preprocess(image_paths, queue_a, p_s2_ocr_count, times_dict):
    """S1: 이미지 로드 및 전처리"""
    times_dict['stage1_start'] = time.time()
    pid = os.getpid()
    print(f" 전처리 작업자 시작. {len(image_paths)}개 이미지 처리.")
    
    try:
        for image_path in image_paths:
            if not os.path.exists(image_path):
                print(f" 경고: 파일을 찾을 수 없습니다. {image_path}")
                continue
            
            img = cv2.imread(image_path)
            if img is None:
                print(f" 경고: 이미지를 읽을 수 없습니다. {image_path}")
                continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            binary_img = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Queue Full -> Block (Backpressure)
            queue_a.put((image_path, binary_img))

    except Exception as e:
        print(f" 치명적 오류: {e}")
    finally:
        # S2 종료 신호 전송
        print(f" 전처리 완료. S2 작업자들에게 {p_s2_ocr_count}개의 종료 신호 전송.")
        for _ in range(p_s2_ocr_count):
            queue_a.put(None)
        
        times_dict['stage1_end'] = time.time()
        print(f" 전처리 작업자 종료.")

# Stage 2: OCR (Worker Pool)
def stage2_ocr(queue_a, queue_b, times_dict, lock):
    """S2: OCR 수행"""
    pid = os.getpid()
    print(f" OCR 작업자 시작.")
    
    # Tesseract 설정
    config = "--psm 6 --oem 1"
    
    try:
        while True:
            # Queue Get
            item = queue_a.get()

            # Sentinel 감지
            if item is None:
                queue_b.put(None) 
                break 
            
            image_path, binary_img = item
            
            try:
                pil_img = Image.fromarray(binary_img)
                
                ocr_start_time = time.time()
                text = pytesseract.image_to_string(pil_img, lang='eng', config=config)
                ocr_duration = time.time() - ocr_start_time
                
                with lock:
                    times_dict['stage2_ocr_total'] = times_dict.get('stage2_ocr_total', 0) + ocr_duration
                
                # 결과 전송
                queue_b.put((image_path, text.strip()))
            
            except Exception as ocr_e:
                print(f" 경고: OCR 처리 오류 {image_path}: {ocr_e}")
                queue_b.put((image_path, f"ERROR: {ocr_e}"))

    except Exception as e:
        print(f" 치명적 오류: {e}")
    finally:
        print(f" OCR 작업자 종료.")

# Stage 3: 결과 취합 (Consumer)
def stage3_save(queue_b, p_s2_ocr_count, results_dict, times_dict):
    """S3: 결과 수집"""
    times_dict['stage3_start'] = time.time()
    pid = os.getpid()
    print(f" 결과 취합 작업자 시작.")
    
    finished_s2_workers = 0
    results_count = 0
    
    try:
        # S2 종료 대기
        while finished_s2_workers < p_s2_ocr_count:
            
            item = queue_b.get()
            
            # Sentinel 감지
            if item is None:
                finished_s2_workers += 1
                continue
                
            image_path, text = item
            results_dict[image_path] = text
            results_count += 1
            
    except Exception as e:
        print(f" 치명적 오류: {e}")
    finally:
        print(f" 결과 취합 완료. 총 {results_count}개 결과 수집.")
        times_dict['stage3_end'] = time.time()
        print(f" 결과 취합 작업자 종료.")

# Main
if __name__ == "__main__":
    
    # 1. 설정
    P_S1_PREPROCESS = 1  
    P_S2_OCR = 6         
    P_S3_SAVE = 1        
    
    total_processes = P_S1_PREPROCESS + P_S2_OCR + P_S3_SAVE
    print(f"--- 모델 B (파이프라인 병렬) 테스트 시작 ---")
    print(f"코어 할당: S1(전처리)={P_S1_PREPROCESS}, S2(OCR)={P_S2_OCR}, S3(저장)={P_S3_SAVE}")
    print(f"총 사용 프로세스: {total_processes}")
    
    QUEUE_MAXSIZE = P_S2_OCR * 2
    
    # 2. 이미지 로드
    print("이미지 파일 로드 중...")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    images_dir = os.path.join(project_root, "dataset", "training_data", "images")
    image_paths = sorted(glob.glob(os.path.join(images_dir, "*.png")) + glob.glob(os.path.join(images_dir, "*.jpg")))
    
    if not image_paths:
        print("="*50)
        print("오류: 테스트할 이미지를 찾을 수 없습니다.")
        print("이 스크립트(.py)와 동일한 폴더에.png 또는.jpg 이미지 파일을 넣어주세요.")
        print("="*50)
        sys.exit(1)
        
    N_IMAGES = len(image_paths)
    print(f"총 {N_IMAGES}개의 이미지 파일을 찾았습니다.")

    # 3. 큐 및 공유 메모리 설정
    manager = multiprocessing.Manager()
    results_dict = manager.dict() 
    times_dict = manager.dict() 
    lock = manager.Lock()
    
    queue_a = multiprocessing.Queue(maxsize=QUEUE_MAXSIZE)
    queue_b = multiprocessing.Queue(maxsize=QUEUE_MAXSIZE)
    
    # 4. 프로세스 시작
    pipeline_start_time = time.time()
    
    processes = []
    
    if P_S1_PREPROCESS > 1:
        print("경고: S1(전처리) 프로세스가 1개 이상이면 모든 S1 워커가 동일한 이미지 목록을 처리합니다.")
        print("      정확한 분산을 위해서는 image_paths를 나누는 로직이 추가로 필요합니다.")

    # S1 (전처리)
    for _ in range(P_S1_PREPROCESS):
        p1 = multiprocessing.Process(
            target=stage1_preprocess, 
            args=(image_paths, queue_a, P_S2_OCR, times_dict)
        )
        processes.append(p1)
        p1.start()

    # S2 (OCR)
    for _ in range(P_S2_OCR):
        p2 = multiprocessing.Process(
            target=stage2_ocr, 
            args=(queue_a, queue_b, times_dict, lock)
        )
        processes.append(p2)
        p2.start()

    # S3 (결과 취합)
    for _ in range(P_S3_SAVE):
        p3 = multiprocessing.Process(
            target=stage3_save, 
            args=(queue_b, P_S2_OCR, results_dict, times_dict)
        )
        processes.append(p3)
        p3.start()

    # 5. 종료 대기
    for p in processes:
        p.join()

    pipeline_end_time = time.time()
    total_time = pipeline_end_time - pipeline_start_time
    
    # 6. 리포트
    stage1_time = times_dict.get('stage1_end', 0) - times_dict.get('stage1_start', 0)
    stage2_ocr_total_time = times_dict.get('stage2_ocr_total', 0)
    stage3_time = times_dict.get('stage3_end', 0) - times_dict.get('stage3_start', 0)

    print("\n" + "="*50)
    print("--- 스테이지별 집계 시간 ---")
    if stage1_time > 0: print(f"스테이지 1 (전처리): {stage1_time:.4f} 초")
    if stage2_ocr_total_time > 0: print(f"스테이지 2 (OCR 총 작업 시간): {stage2_ocr_total_time:.4f} 초")
    if stage3_time > 0: print(f"스테이지 3 (결과 취합): {stage3_time:.4f} 초")
    print("="*50)

    print("\n" + "="*50)
    print("--- 모든 파이프라인 작업 완료 ---")
    print(f"총 소요 시간: {total_time:.4f} 초")
    print(f"총 처리 이미지: {len(results_dict)} / {N_IMAGES} 개")
    if total_time > 0:
        print(f"평균 처리량: {N_IMAGES / total_time:.2f} images/sec")
    print("="*50)

    # 샘플 출력
    print("\n--- 결과 샘플 (최대 5개) ---")
    count = 0
    for path, text in results_dict.items():
        print(f"FILE: {os.path.basename(path)}")
        print(f"TEXT: {text[:70].replace(chr(10), ' ')}...")
        print("-" * 20)
        count += 1
        if count >= 5:
            break