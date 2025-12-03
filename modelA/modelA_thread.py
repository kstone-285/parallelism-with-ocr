import cv2
import pytesseract
from PIL import Image
import time
import os
import glob
import threading
import queue 
import multiprocessing

# --- GIL 시뮬레이션용 락 ---
# 이 락을 잡은 스레드만 OCR을 돌릴 수 있습니다. (마치 GIL처럼 동작)
gil_simulation_lock = threading.Lock()

# --- 공유 메모리 ---
shared_results = []
shared_times = {"preprocess": 0, "ocr": 0, "postprocess": 0}
result_lock = threading.Lock() 

def process_single_image(image_path):
    try:
        # Stage 1: Pre-processing (여기는 보통 GIL 영향을 덜 받으므로 놔둡니다)
        preprocess_start = time.time()
        img = cv2.imread(image_path)
        if img is None: return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binary_img = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        preprocess_time = time.time() - preprocess_start
        
        # Stage 2: OCR (여기가 핵심!)
        ocr_start = time.time()
        pil_img = Image.fromarray(binary_img)
        config = "--psm 6 --oem 1"
        
        # [핵심 변경] GIL 병목 시뮬레이션
        # pytesseract가 내부적으로 GIL을 풀어주더라도, 
        # 우리가 파이썬 레벨에서 Lock을 걸어서 한 번에 한 스레드만 실행하게 만듭니다.
        # 결과적으로 "병렬 처리"가 아니라 "순차 처리"가 됩니다.
        with gil_simulation_lock:
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
    except Exception:
        return None

def worker_thread(task_queue, worker_id):
    """일감을 가져와 처리하는 워커 스레드"""
    # print(f"스레드-{worker_id} 시작") # 출력 줄임
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

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(project_root, "dataset", "training_data", "images")
    image_paths = sorted(glob.glob(os.path.join(images_dir, "*.png")) + glob.glob(os.path.join(images_dir, "*.jpg")))
    
    # image_paths = image_paths * 10 

    if not image_paths:
        print("이미지 없음")
        exit()

    thread_count = multiprocessing.cpu_count()
    print(f"멀티스레딩 (GIL 병목 시뮬레이션) 시작... (스레드: {thread_count}개)")
    print("주의: OCR 구간에 강제 Lock을 걸어 성능 저하를 유도합니다.")

    task_queue = queue.Queue()
    for path in image_paths:
        task_queue.put(path)

    start_time = time.time()

    threads = []
    for i in range(thread_count):
        t = threading.Thread(target=worker_thread, args=(task_queue, i))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n[결과] 멀티스레딩(GIL Sim) 완료.")
    print(f"총 소요 시간: {total_time:.2f} 초")
    print(f"-> Baseline과 비슷하거나 더 느려야 정상입니다.")