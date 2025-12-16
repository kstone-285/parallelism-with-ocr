# ⚡ High-Performance OCR Pipeline: Optimizing Inference with Parallelism

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![Tesseract](https://img.shields.io/badge/Tesseract-OCR-000000?logo=tesseract&logoColor=white)
![Multiprocessing](https://img.shields.io/badge/Parallelism-Multiprocessing-orange)
![AsyncIO](https://img.shields.io/badge/Concurrency-AsyncIO-RED)
![MPI](https://img.shields.io/badge/Distributed-MPI-purple)

> **"Maximizing Throughput in CPU-Bound OCR Tasks using Asymmetric Pipelining & Hybrid Concurrency"**

본 프로젝트는 대규모 비정형 데이터(이미지) 처리에 있어 **Tesseract OCR의 연산 병목(CPU-Bound)**과 **I/O 대기 시간(I/O-Bound)**을 동시에 해결하기 위한 **고성능 분산 처리 아키텍처 연구**입니다.

단순한 순차 처리 방식을 넘어, **멀티프로세싱 기반의 데이터 병렬화**와 **비대칭 파이프라인(Asymmetric Pipelining) 전략**을 통해 추론 속도를 약 **7.28배** 가속화했습니다.

---

## 📑 Table of Contents

1. [Project Overview](#-1-project-overview)
2. [Theoretical Background](#-2-theoretical-background)
3. [System Architecture](#-3-system-architecture)
4. [Performance Benchmark](#-4-performance-benchmark)
5. [Project Structure](#-5-project-structure)
6. [How to Run](#-6-how-to-run)
7. [Authors & References](#-7-authors--references)

---

## 📖 1. Project Overview

### 🎯 Core Problem
* **CPU Bottleneck:** Tesseract OCR은 복잡한 LSTM 연산을 수행하는 고비용 CPU 작업
* **I/O Blocking:** 이미지 로드(Load) 및 저장(Save) 시 발생하는 I/O 대기 시간 동안 CPU 자원이 낭비
* **GIL Limitation:** Python의 Global Interpreter Lock으로 인해 멀티스레딩으로는 성능 향상에 한계가 있음

### 🏆 Key Achievements
* **7.28x Speedup:** 단일 코어 Baseline 대비 약 7.28배의 추론 속도 향상 (3073.8s → 422.4s)
* **Sweet Spot Discovery:** 전체 워크로드의 **99%**를 차지하는 OCR 추론 단계에 자원을 집중하는 **1:9:1 (Pre:OCR:Post)** 최적 할당 비율 도출
* **Hybrid Concurrency:** `AsyncIO`와 `Multiprocessing`을 결합하여 유휴 자원(Idle Resource)을 최소화하는 하이브리드 아키텍처 구현

---

## 📚 2. Theoretical Background

본 프로젝트의 설계는 다음의 컴퓨터 공학 이론을 기반으로 합니다.

### 2.1 Why Multiprocessing? (Overcoming GIL)
OCR 추론은 대표적인 **CPU-Bound** 작업이고, 특히 Python은 **GIL**로 인해 멀티스레딩을 사용하더라도 한 시점에 하나의 스레드만 바이트코드를 실행이 가능
* **Solution:** 각 워커가 독립된 메모리 공간과 고유한 GIL을 갖는 **Multiprocessing**을 도입하여 물리적 병렬성을 확보

### 2.2 Why Pipeline? (Resolving Bottleneck)
파이프라인 시스템의 전체 처리량(Throughput)은 가장 느린 단계(Bottleneck Stage)에 의해 결정 (**Amdahl's Law**)
* **Analysis:** 데이터셋 분석 결과, **추론(Inference) 단계가 전체 실행 시간의 99%**를 차지하는 불균형을 확인
* **Solution:** 작업을 **[전처리] → [추론] → [후처리]**로 분리하고, 병목 구간인 [추론] 단계에 가용 코어의 80% 이상을 할당하는 **비대칭 자원 할당 전략**을 적용

### 2.3 Distributed Scalability (MPI)
단일 머신의 코어 한계를 극복하기 위해 `MPI (Message Passing Interface)`를 도입
* **Trade-off:** 분산 환경에서는 데이터 전송에 따른 **직렬화(Serialization) 오버헤드**가 성능의 주요 변수임을 실험적으로 검증

---

## 🏗️ 3. System Architecture

### 🔹 Model A: Data Parallelism (SPMD)
* **Concept:** 전체 데이터셋을 $N$개의 청크(Chunk)로 분할하여 독립적인 워커 프로세스에 할당
* **Pros:** 구현이 간단하고 데이터 의존성이 없을 때 효과적
* **Cons:** I/O와 CPU 작업이 혼재되어 있어, I/O 대기 시간 동안 CPU 유휴 상태 발생

### 🔹 Model B: Asymmetric Pipeline (Producer-Consumer) [Proposed]
* **Concept:** `multiprocessing.Queue`를 통해 데이터를 스트리밍하며 처리
* **Structure:**
    1.  **Stage 1 (Producer):** Image Load & Preprocess (1 Core)
    2.  **Stage 2 (Worker Pool):** OCR Inference (N Cores) - **Allocated Max Resources**
    3.  **Stage 3 (Consumer):** Post-process & Save (1 Core)
* **Result:** **1:9:1 (Pre:OCR:Post)** 비율에서 병목 현상이 해소되며 최대 성능 달성

### 🔹 Model B + Hybrid Concurrency (AsyncIO)
* **Tech:** `Multiprocessing` (Physical Core) + `AsyncIO` (Logical Concurrency)
* **Mechanism:** 워커 프로세스 내부에서 **Event Loop**를 구동하고, `cv2.imread` 등의 Blocking 연산을 `ThreadPoolExecutor`로 위임하여 **Non-blocking Pipeline** 구현

---

## 📊 4. Performance Benchmark

**Environment:** Apple M3 Pro (11-core CPU), Python 3.9  
**Dataset:** NAVER CLOVA CORD-v2 (Train-set) - Real-world receipt images

| Model | Architecture | Core Allocation | Time (s) | Speedup |
| :--- | :--- | :---: | :---: | :---: |
| **Baseline** | Sequential | 1 Core | 3073.8s | 1.0x |
| **Model A** | Data Parallelism (Pool) | All Cores | 486.8s | 6.31x |
| **Model B** | **Pipeline (Optimized)** | **1 / 9 / 1** | **422.4s** | **7.28x** |
| **Model B** | Pipeline (Async I/O) | 1 / 9 / 1 | 428.4s | 7.17x |

> **Insight:** 파이프라인 모델(Model B)이 단순 데이터 병렬화(Model A)보다 우수하며, 특히 **1:9:1 비율**에서 유휴 자원이 최소화되어 가장 높은 Throughput을 기록했습니다.

---

## 📂 5. Project Structure

```text
parallelism-with-ocr/
│
├── dataset/
│   └── training_data/
│       └── images/           # CORD-v2 Dataset (Images)
│
├── baseline.py               # [Baseline] Sequential Processing
├── modelA.py                 # [Model A] Data Parallelism (Pool)
├── modelA_nopool.py          # [Model A] Manual Process Control
├── modelA_mpi.py             # [Model A] Distributed Processing with MPI
├── modelB.py                 # [Model B] Pipeline Parallelism (Sync)
├── modelB_async.py           # [Model B+] Pipeline with AsyncIO & ThreadPool
├── modelB_mpi.py             # [Model B] Distributed Pipeline with MPI
│
└── requirements.txt          # Dependencies
```

---

## 🏃 6. How to Run

Prerequisites

```bash
# 1. Install Tesseract OCR (System Dependency)
brew install tesseract  # macOS
sudo apt install tesseract-ocr  # Linux

# 2. Install Python Packages
pip install -r requirements.txt
```
Execution
1. Recommended Model (Model B - Optimized Pipeline)

```bash
python modelB.py
# Follow the prompt for core allocation:
# S1 (Preprocess): 1
# S2 (OCR): 9  (Adjust based on your CPU cores)
# S3 (Save): 1
2. AsyncIO Hybrid Model
```

```bash
python modelB_async.py
3. Distributed Simulation (MPI)
```

```bash
# Start MPI Cluster (Requires ipyparallel)
ipcluster start -n 7 --engines=mpi

# Run Orchestrator
python modelB_mpi.py
```

---

## 👨‍💻 7. Members & References

* Members: Kyoosuk Hwang (21101239), Chandong Hwang (21101240)

* Institution: Seoul National University of Science and Technology, Dept. of Computer Science

* This project was conducted as part of the Big Data Processing course at Seoul National University of Science and Technology.

📘 Note: 본 프로젝트에 대한 더 자세한 이론적 배경, 실험 설계 과정, 그리고 상세한 결과 분석은 리포지토리에 포함된 **[최종 보고서 (PDF/Docx)]**를 참조해주시기 바랍니다.