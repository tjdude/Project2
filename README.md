# AI 기반 탈모 상태 진단 및 평가 서비스

### 🔬 프로젝트 주제
- AI 기반 두피 이미지를 분석하여, 현재 탈모 상태를 3가지 기준으로 정량화하고 점수화된 진단 결과를 제공하는 탈모 평가 시스템

### 🎯 프로젝트 목표
- 두피 이미지에서 탈모 관련 요소 3가지를 추출
- 각 요소를 정량화하여 점수화된 진단 결과 제공
- 사용자에게 현재 탈모 상태에 대한 객관적인 인식 제공
  
### 📆 프로젝트 기간
2025.02.24 ~ 2025.03.17
---
### 💡 활용 기술
- 인공지능
  - OpenCV, Numpy, YOLO11, Pytorch, TF-Lite, KMeans
 
- 애플리케이션
  - Flask, Jinja2, HTML, base64, Linux
 
- 하드웨어 플랫폼
  - Raspberry Pi 5
--- 
### 🛠 개발 과정
- **인공지능 시스템 구현**
  - AI-Hub에서 두피 이미지 데이터 수집 및 직접 라벨링 작업 수행
  - OpenCV를 활용하여 이미지 전처리 (리사이즈, 색상 변환 등) 진행
  - YOLOv11 객체 탐지 모델을 적용하여 이미지 내 모공 단위 머리카락 수를 탐지하고 바운딩 박스로 시각화
  - Pytorch를 활용하여 CNN 모델을 설계 및 두피 상태 분류 모델 학습 수행
  - 학습된 모델을 평가한 후 성능이 우수한 모델을 TF-Lite 모델로 경량화하여 패키징
  - KMeans Segmentation을 활용하여 두피 및 머리카락 면적 수치화

- **애플리케이션 구현**
  - TF-Lite 모델을 라즈베리파이에 탑재
  - USB 카메라로 촬영한 이미지를 실시간으로 모델에 입력 후 측정
  - Flask 기반 서버 구축으로 웹 애플리케이션 형태로 연동
  - HTML을 활용해 추론 결과를 시각화하는 웹 페이지 제작
---    
### 📁 기능별 Jupyter 노트북 업로드
1. 모공당 머리카락 개수 추출
2. 모델 경량화
3. 두피 & 모발 면적 분석

### 🖥️ 웹 애플리케이션 코드
- app.py : Flask 기반 백엔드 서버
- templates : 사용자 인터페이스를 구성하는 HTML 템플릿

### 🔗 모델 파일 
  [Model(Google Drive)](https://drive.google.com/drive/folders/1zja8ApEzK1q6DGCXXx_MC9F5H7PygK5-?usp=sharing)

---

### 📊 구현 결과
*기능 1,2*
- https://youtu.be/WmfiY4-5Ep0

*기능 3*
- https://youtu.be/4gdbaQSZbZk
 
### 📊 결과

- 시스템 아키텍처
  ![시스템](https://github.com/user-attachments/assets/88b7d52c-7edc-4ef0-9113-08de07647518)

- 모델 아키텍처
  
  ![모델](https://github.com/user-attachments/assets/7400faa9-ca17-48c2-86c5-7b85ef39a658)
