import threading
import queue
import os
import glob
import time
import sys
import cv2
import pytesseract
from PIL import Image

# ----------------------------------------------------------------------------
# STAGE 1: 전처리 (Producer)
# - 이미지 경로를 받아, 디스크에서 이미지를 읽고 OpenCV로 전처리합니다.
# - 처리된 NumPy 배열을 'queue_a'에 넣습니다.
# ----------------------------------------------------------------------------
def stage1_preprocess(image_paths, queue_a, worker_id, times_dict):
    """
    S1: 이미지를 읽고 전처리하여 queue_a에 넣습니다.
    (스레드 버전)
    """
    # 첫 번째 워커만 전체 스테이지의 시작 시간을 기록합니다.
    if worker_id == 0:
        times_dict['stage1_start'] = time.time()
        
    thread_name = threading.current_thread().name
    print(f" 전처리 스레드 {thread_name} (ID: {worker_id}) 시작. {len(image_paths)}개 이미지 처리.")
    
    try:
        for image_path in image_paths:
            if not os.path.exists(image_path):
                print(f" 경고: 파일을 찾을 수 없습니다. {image_path}")
                continue
                
            # II.B의 기준선 코드와 동일한 전처리 로직
            img = cv2.imread(image_path)
            if img is None:
                print(f" 경고: 이미지를 읽을 수 없습니다. {image_path}")
                continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            binary_img = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            
            # (이미지 ID, 처리된 데이터) 튜플을 큐에 삽입
            queue_a.put((image_path, binary_img))

    except Exception as e:
        print(f" 치명적 오류 (S1, 스레드 {worker_id}): {e}")
    finally:
        print(f" 전처리 스레드 {worker_id} 종료.")

# ----------------------------------------------------------------------------
# STAGE 2: OCR (Worker Pool)
# - 'queue_a'에서 전처리된 데이터를 가져옵니다.
# - Tesseract OCR을 수행합니다.
# - (이미지 ID, OCR 결과 텍스트)를 'queue_b'에 넣습니다.
# ----------------------------------------------------------------------------
def stage2_ocr(queue_a, queue_b, times_dict, lock):
    """
    S2: queue_a에서 데이터를 받아 Tesseract OCR을 수행하고 queue_b로 보냅니다.
    'None'을 받으면 루프를 종료하고 'None'을 queue_b로 전파합니다. (스레드 버전)
    """
    thread_name = threading.current_thread().name
    print(f" OCR 스레드 {thread_name} 시작.")
    
    config = "--psm 6 --oem 1"
    
    try:
        while True:
            item = queue_a.get()

            if item is None:
                queue_b.put(None)
                break
            
            image_path, binary_img = item
            
            try:
                pil_img = Image.fromarray(binary_img)
                
                ocr_start_time = time.time()
                # pytesseract 호출은 외부 프로세스를 실행하므로 GIL을 해제합니다.
                # 따라서 스레딩 환경에서도 병렬성 효과를 볼 수 있습니다.
                text = pytesseract.image_to_string(pil_img, lang='eng', config=config)
                ocr_end_time = time.time()

                # Lock을 사용하여 공유된 시간 합계를 안전하게 업데이트합니다.
                with lock:
                    current_total = times_dict.get('stage2_ocr_total', 0)
                    times_dict['stage2_ocr_total'] = current_total + (ocr_end_time - ocr_start_time)
                
                queue_b.put((image_path, text.strip()))
            
            except Exception as ocr_e:
                print(f" 경고: OCR 처리 오류 {image_path}: {ocr_e}")
                queue_b.put((image_path, f"ERROR: {ocr_e}"))

    except Exception as e:
        print(f" 치명적 오류 (S2, {thread_name}): {e}")
    finally:
        print(f" OCR 스레드 {thread_name} 종료.")

# ----------------------------------------------------------------------------
# STAGE 3: 결과 취합 (Consumer)
# - 'queue_b'에서 OCR 결과 튜플을 가져옵니다.
# - 모든 S2 워커가 'None' 신호를 보낼 때까지 결과를 수집합니다.
# - 'results_dict'에 최종 결과를 저장합니다.
# ----------------------------------------------------------------------------
def stage3_save(queue_b, t_s2_ocr_count, results_dict, times_dict):
    """
    S3: queue_b에서 결과를 받아 results_dict에 저장합니다.
    S2 스레드 수만큼 'None'을 받으면 종료합니다. (스레드 버전)
    """
    times_dict['stage3_start'] = time.time()
    thread_name = threading.current_thread().name
    print(f" 결과 취합 스레드 {thread_name} 시작.")
    
    finished_s2_threads = 0
    results_count = 0
    
    try:
        while finished_s2_threads < t_s2_ocr_count:
            item = queue_b.get()
            
            if item is None:
                finished_s2_threads += 1
                continue
                
            image_path, text = item
            results_dict[image_path] = text
            results_count += 1
            
    except Exception as e:
        print(f" 치명적 오류 (S3, {thread_name}): {e}")
    finally:
        print(f" 결과 취합 완료. 총 {results_count}개 결과 수집.")
        times_dict['stage3_end'] = time.time()
        print(f" 결과 취합 스레드 {thread_name} 종료.")

# ----------------------------------------------------------------------------
# MAIN: 파이프라인 설정 및 실행
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    
    # --- 1. [조정 가능] 스레드 할당 파라미터 ---
    T_S1_PREPROCESS = 3  # 전처리 스레드 수
    T_S2_OCR = 6         # OCR 워커 스레드 수
    T_S3_SAVE = 1        # 결과 취합 스레드 수
    
    total_threads = T_S1_PREPROCESS + T_S2_OCR + T_S3_SAVE
    print(f"--- 모델 B (파이프라인 멀티스레딩) 테스트 시작 ---")
    print(f"스레드 할당: S1(전처리)={T_S1_PREPROCESS}, S2(OCR)={T_S2_OCR}, S3(저장)={T_S3_SAVE}")
    print(f"총 사용 스레드: {total_threads}")
    
    QUEUE_MAXSIZE = T_S2_OCR * 2
    
    # --- 2. 이미지 경로 불러오기 ---
    print("이미지 파일 로드 중...")
    project_root = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(project_root, "dataset", "training_data", "images")
    image_paths = sorted(glob.glob(os.path.join(images_dir, "*.png")) + glob.glob(os.path.join(images_dir, "*.jpg")))
    
    if not image_paths:
        print("="*50)
        print("오류: 테스트할 이미지를 찾을 수 없습니다.")
        print(f"지정된 경로 '{images_dir}'에 .png 또는 .jpg 이미지 파일이 있는지 확인해주세요.")
        print("="*50)
        sys.exit(1)
        
    N_IMAGES = len(image_paths)
    print(f"총 {N_IMAGES}개의 이미지 파일을 찾았습니다.")

    # --- 3. 큐(Queue) 및 공유 데이터 설정 ---
    results_dict = {}  # 스레드는 메모리를 공유하므로 일반 dict 사용
    times_dict = {}    # 시간 측정을 위한 공유 딕셔너리
    lock = threading.Lock() # 공유 데이터(times_dict) 접근 제어를 위한 Lock
    
    queue_a = queue.Queue(maxsize=QUEUE_MAXSIZE)
    queue_b = queue.Queue(maxsize=QUEUE_MAXSIZE)
    
    # --- 4. 스레드 생성 및 시작 ---
    pipeline_start_time = time.time()
    
    s1_threads, s2_threads, s3_threads = [], [], []

    # 이미지 경로를 S1 워커들에게 분배
    if T_S1_PREPROCESS > 1:
        chunk_size = (N_IMAGES + T_S1_PREPROCESS - 1) // T_S1_PREPROCESS
        path_chunks = [image_paths[i:i + chunk_size] for i in range(0, len(image_paths), chunk_size)]
    else:
        path_chunks = [image_paths]

    # Stage 1 (전처리) 스레드 생성
    for i in range(T_S1_PREPROCESS):
        worker_paths = path_chunks[i] if i < len(path_chunks) else []
        if not worker_paths: continue
        t1 = threading.Thread(target=stage1_preprocess, args=(worker_paths, queue_a, i, times_dict))
        s1_threads.append(t1)
        t1.start()

    # Stage 2 (OCR) 워커 풀 생성
    for _ in range(T_S2_OCR):
        t2 = threading.Thread(target=stage2_ocr, args=(queue_a, queue_b, times_dict, lock))
        s2_threads.append(t2)
        t2.start()

    # Stage 3 (결과 취합) 스레드 생성
    for _ in range(T_S3_SAVE):
        t3 = threading.Thread(target=stage3_save, args=(queue_b, T_S2_OCR, results_dict, times_dict))
        s3_threads.append(t3)
        t3.start()

    # --- 5. 파이프라인 종료 처리 ---
    
    # S1 스레드들이 모든 이미지 처리를 마칠 때까지 기다립니다.
    # 이 과정에서 S1이 queue_a를 채우고, S2가 queue_a를 비우는 작업이 동시에 일어납니다.
    for t in s1_threads:
        t.join()
    
    # S1이 모두 종료되었으므로, S1 종료 시간을 기록합니다.
    if 'stage1_start' in times_dict:
        times_dict['stage1_end'] = time.time()
    print("모든 S1(전처리) 작업 완료.")

    # S1 작업이 모두 끝났으므로, S2 스레드들에게 종료 신호(None)를 보냅니다.
    print(f"S2 스레드들에게 {T_S2_OCR}개의 종료 신호를 전송합니다.")
    for _ in range(T_S2_OCR):
        queue_a.put(None)

    # S2와 S3 스레드들이 모두 종료될 때까지 기다립니다.
    for t in s2_threads + s3_threads:
        t.join()
    print("모든 S2(OCR) 및 S3(결과 취합) 작업 완료.")

    pipeline_end_time = time.time()
    total_time = pipeline_end_time - pipeline_start_time
    
    # --- 6. 최종 결과 리포트 ---
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

    # (선택 사항) 결과 샘플 5개 출력
    print("\n--- 결과 샘플 (최대 5개) ---")
    count = 0
    for path, text in results_dict.items():
        print(f"FILE: {os.path.basename(path)}")
        print(f"TEXT: {text[:70].replace(chr(10), ' ')}...")
        print("-" * 20)
        count += 1
        if count >= 5:
            break