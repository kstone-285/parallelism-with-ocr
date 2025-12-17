import cv2
import pytesseract
from PIL import Image
import time
import os
import glob
import threading
import queue 
import multiprocessing

# Global: GIL 시뮬레이션 및 공유 리소스
gil_simulation_lock = threading.Lock() # OCR 구간 강제 동기화 (GIL 효과)
shared_results = []
shared_times = {"preprocess": 0, "ocr": 0, "postprocess": 0}
result_lock = threading.Lock() # 결과 집계용 Lock

# Worker: 단일 이미지 처리 (GIL 병목 포함)
def process_single_image(image_path):
    """이미지 처리 및 GIL 병목 시뮬레이션 수행"""
    try:
        # Stage 1: 전처리
        preprocess_start = time.time()
        img = cv2.imread(image_path)
        if img is None: return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binary_img = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        preprocess_time = time.time() - preprocess_start
        
        # Stage 2: OCR (GIL 시뮬레이션)
        ocr_start = time.time()
        pil_img = Image.fromarray(binary_img)
        config = "--psm 6 --oem 1"
        
        # 강제 Lock: 한 번에 한 스레드만 OCR 수행 (순차 처리 유도)
        with gil_simulation_lock:
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
    except Exception:
        return None

# Thread Worker: 큐 소비
def worker_thread(task_queue, worker_id):
    """작업 큐에서 이미지를 가져와 처리"""
    while True:
        try:
            image_path = task_queue.get_nowait()
        except queue.Empty:
            break
            
        result = process_single_image(image_path)
        
        if result:
            with result_lock:
                shared_results.append(result["text"])
                for stage, val in result["times"].items():
                    shared_times[stage] += val
        
        task_queue.task_done()

# Main: 스레드 설정 및 실행
if __name__ == "__main__":
    
    # 1. 이미지 로드
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    images_dir = os.path.join(project_root, "dataset", "training_data", "images")
    image_paths = sorted(glob.glob(os.path.join(images_dir, "*.png")) + glob.glob(os.path.join(images_dir, "*.jpg")))
    
    if not image_paths:
        print("오류: 이미지를 찾을 수 없습니다.")
        exit()

    # 2. 스레드 및 큐 설정
    thread_count = multiprocessing.cpu_count()
    print(f"멀티스레딩 (GIL 병목 시뮬레이션) 시작... (스레드: {thread_count}개)")
    print("주의: OCR 구간에 강제 Lock을 걸어 성능 저하를 유도합니다.")

    task_queue = queue.Queue()
    for path in image_paths:
        task_queue.put(path)

    start_time = time.time()

    # 3. 스레드 시작 및 대기
    threads = []
    for i in range(thread_count):
        t = threading.Thread(target=worker_thread, args=(task_queue, i))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    end_time = time.time()
    total_time = end_time - start_time
    
    # 4. 결과 리포트
    print(f"\n[결과] 멀티스레딩(GIL Sim) 완료.")
    print(f"총 소요 시간: {total_time:.2f} 초")
    print(f"-> Baseline과 비슷하거나 더 느려야 정상입니다.")