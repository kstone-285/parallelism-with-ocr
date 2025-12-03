import multiprocessing
import asyncio
import os
import glob
import time
import sys
import cv2
import pytesseract
from PIL import Image
import numpy as np

# ----------------------------------------------------------------------------
# STAGE 1: 전처리 (Producer) - 비동기 버전
# - 이미지 경로를 받아, 디스크 읽기 및 전처리를 비동기적으로 수행합니다.
# - 처리된 데이터를 'queue_a'에 넣습니다.
# ----------------------------------------------------------------------------
async def _process_single_image_for_stage1(image_path):
    """(Helper) 개별 이미지를 비동기 I/O로 읽고 전처리합니다."""
    loop = asyncio.get_running_loop()
    
    # cv2.imread는 블로킹 I/O이므로 executor에서 실행
    img_data = await loop.run_in_executor(None, cv2.imread, image_path)
    if img_data is None:
        print(f" 경고: 이미지를 읽을 수 없습니다. {image_path}")
        return None

    # CPU 집약적 전처리 (이것도 executor에서 실행하여 이벤트 루프 확보)
    def preprocess(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
    
    binary_img = await loop.run_in_executor(None, preprocess, img_data)
    return image_path, binary_img

async def stage1_preprocess_async(image_paths_chunk, queue_a, p_s2_ocr_count, times_dict):
    """S1: 이미지들을 비동기적으로 전처리하여 queue_a에 넣습니다."""
    times_dict['stage1_start'] = time.time()
    print(f" S1-{os.getpid()}: 전처리 작업자 시작. {len(image_paths_chunk)}개 이미지 처리.")

    tasks = [_process_single_image_for_stage1(path) for path in image_paths_chunk]
    results = await asyncio.gather(*tasks)

    for result in results:
        if result:
            queue_a.put(result)

    # 이 프로세스 담당의 모든 이미지가 처리되면, 담당한 S2 워커들에게 종료 신호 전송
    # 분산 환경에서는 종료 신호 관리가 더 복잡해지지만, 여기서는 단순화된 모델을 가정합니다.
    # 여기서는 S1 프로세스가 하나라고 가정하고 종료 신호를 보냅니다.
    # 여러 S1 프로세스를 제대로 지원하려면, 모든 S1이 끝났는지 확인하는 로직이 필요합니다.
    if os.getpid() == times_dict.get('s1_leader_pid', -1):
        print(f" S1-{os.getpid()}: 전처리 완료. S2 작업자들에게 {p_s2_ocr_count}개의 종료 신호 전송.")
        for _ in range(p_s2_ocr_count):
            queue_a.put(None)

    times_dict['stage1_end'] = time.time()
    print(f" S1-{os.getpid()}: 전처리 작업자 종료.")

# ----------------------------------------------------------------------------
# STAGE 2: OCR (Worker Pool) - 비동기 버전
# - 'queue_a'에서 데이터를 비동기적으로 가져와 OCR을 수행합니다.
# - OCR(CPU 집약적)은 executor에서, 큐 작업(I/O 블로킹)도 executor에서 실행합니다.
# ----------------------------------------------------------------------------
async def stage2_ocr_async(queue_a, queue_b, times_dict, lock):
    """S2: queue_a에서 데이터를 받아 OCR을 수행하고 queue_b로 보냅니다."""
    print(f" S2-{os.getpid()}: OCR 작업자 시작.")
    loop = asyncio.get_running_loop()
    config = "--psm 6 --oem 1"
    
    while True:
        # queue.get()은 블로킹 함수이므로 executor에서 비동기적으로 실행
        item = await loop.run_in_executor(None, queue_a.get)

        if item is None:
            await loop.run_in_executor(None, queue_b.put, None)
            break
        
        image_path, binary_img = item

        try:
            pil_img = Image.fromarray(binary_img)
            
            # pytesseract.image_to_string은 CPU 집약적이므로 executor에서 실행
            ocr_start_time = time.time()
            text = await loop.run_in_executor(None, pytesseract.image_to_string, pil_img, 'eng', config)
            ocr_duration = time.time() - ocr_start_time

            with lock:
                times_dict['stage2_ocr_total'] = times_dict.get('stage2_ocr_total', 0) + ocr_duration

            await loop.run_in_executor(None, queue_b.put, (image_path, text.strip()))
        
        except Exception as ocr_e:
            print(f" 경고: S2-{os.getpid()} OCR 처리 오류 {image_path}: {ocr_e}")
            await loop.run_in_executor(None, queue_b.put, (image_path, f"ERROR: {ocr_e}"))

    print(f" S2-{os.getpid()}: OCR 작업자 종료.")

# ----------------------------------------------------------------------------
# STAGE 3: 결과 취합 (Consumer) - 비동기 버전
# ----------------------------------------------------------------------------
async def stage3_save_async(queue_b, p_s2_ocr_count, results_dict, times_dict):
    """S3: queue_b에서 비동기적으로 결과를 받아 results_dict에 저장합니다."""
    times_dict['stage3_start'] = time.time()
    print(f" S3-{os.getpid()}: 결과 취합 작업자 시작.")
    loop = asyncio.get_running_loop()
    
    finished_s2_workers = 0
    while finished_s2_workers < p_s2_ocr_count:
        item = await loop.run_in_executor(None, queue_b.get) 
        
        if item is None:
            finished_s2_workers += 1
            continue
            
        image_path, text = item
        results_dict[image_path] = text
        
    print(f" S3-{os.getpid()}: 결과 취합 완료. 총 {len(results_dict)}개 결과 수집.")
    times_dict['stage3_end'] = time.time()
    print(f" S3-{os.getpid()}: 결과 취합 작업자 종료.")

# ----------------------------------------------------------------------------
# Worker Process Entry Point
# - 각 프로세스에서 asyncio 이벤트 루프를 시작하는 래퍼 함수입니다.
# ----------------------------------------------------------------------------
def run_async_worker(target_func, *args):
    """주어진 비동기 함수를 실행하기 위한 동기적 진입점"""
    asyncio.run(target_func(*args))

# ----------------------------------------------------------------------------
# MAIN: 파이프라인 설정 및 실행
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    
    # --- 1. [조정 가능] 코어 할당 파라미터 ---
    P_S1_PREPROCESS = 1
    P_S2_OCR = 9  
    P_S3_SAVE = 1
    
    total_processes = P_S1_PREPROCESS + P_S2_OCR + P_S3_SAVE
    print(f"--- 모델 B (비동기 파이프라인) 테스트 시작 ---")
    print(f"코어 할당: S1(전처리)={P_S1_PREPROCESS}, S2(OCR)={P_S2_OCR}, S3(저장)={P_S3_SAVE}")
    print(f"총 사용 프로세스: {total_processes}")
    
    QUEUE_MAXSIZE = P_S2_OCR * 2
    
    # --- 2. 이미지 경로 불러오기 및 분배 ---
    print("이미지 파일 로드 중...")
    project_root = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(project_root, "dataset", "training_data", "images")
    all_image_paths = sorted(glob.glob(os.path.join(images_dir, "*.png")) + glob.glob(os.path.join(images_dir, "*.jpg")))
    
    if not all_image_paths:
        print("="*50)
        print(f"오류: '{images_dir}'에서 테스트할 이미지를 찾을 수 없습니다.")
        print("="*50)
        sys.exit(1)
        
    N_IMAGES = len(all_image_paths)
    print(f"총 {N_IMAGES}개의 이미지 파일을 찾았습니다.")
    
    # S1 프로세스들에게 이미지 경로를 분배
    s1_chunks = np.array_split(all_image_paths, P_S1_PREPROCESS)
    s1_chunks = [chunk.tolist() for chunk in s1_chunks if chunk.size > 0]

    # --- 3. 큐(Queue) 및 공유 메모리(Manager) 설정 ---
    manager = multiprocessing.Manager()
    results_dict = manager.dict()
    times_dict = manager.dict()
    lock = manager.Lock()
    
    queue_a = multiprocessing.Queue(maxsize=QUEUE_MAXSIZE)
    queue_b = multiprocessing.Queue(maxsize=QUEUE_MAXSIZE)
    
    # --- 4. 프로세스 생성 및 시작 ---
    pipeline_start_time = time.time()
    processes = []
    
    # Stage 1 (전처리) 프로세스 생성
    leader_pid_set = False
    for chunk in s1_chunks:
        p1 = multiprocessing.Process(
            target=run_async_worker, 
            args=(stage1_preprocess_async, chunk, queue_a, P_S2_OCR, times_dict)
        )
        processes.append(p1)
        p1.start()
        if not leader_pid_set: # 첫번째 S1 프로세스를 리더로 지정하여 종료 신호 전송 담당
            times_dict['s1_leader_pid'] = p1.pid
            leader_pid_set = True

    # Stage 2 (OCR) 워커 풀 생성
    for _ in range(P_S2_OCR):
        p2 = multiprocessing.Process(
            target=run_async_worker, 
            args=(stage2_ocr_async, queue_a, queue_b, times_dict, lock)
        )
        processes.append(p2)
        p2.start()

    # Stage 3 (결과 취합) 프로세스 생성
    for _ in range(P_S3_SAVE):
        p3 = multiprocessing.Process(
            target=run_async_worker, 
            args=(stage3_save_async, queue_b, P_S2_OCR, results_dict, times_dict)
        )
        processes.append(p3)
        p3.start()

    # --- 5. 모든 프로세스 종료 대기 ---
    for p in processes:
        p.join()

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
    # manager.dict()는 바로 items()를 호출할 수 없으므로 일반 dict로 복사
    final_results = dict(results_dict)
    for path, text in final_results.items():
        print(f"FILE: {os.path.basename(path)}")
        print(f"TEXT: {text[:70].replace(chr(10), ' ')}...")
        print("-" * 20)
        count += 1
        if count >= 5:
            break