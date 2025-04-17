import zipfile
import os
# 폴더가 존재하지 않으면 경로에 압축해제
if not os.path.exists('clust_img'):
    path = '/content/drive/MyDrive/dataset/clus_img.zip' # 폴더 경로
    f_zip = zipfile.ZipFile(path)
    f_zip.extractall('clust_img')
    f_zip.close()

import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

image_folder = "/content/clust_img/clustering_img"
image_files = glob.glob(os.path.join(image_folder, "*.jpg"))

# 최대 9개의 이미지만 샘플링
num_samples = min(len(image_files), 9)
sample_images = image_files[:num_samples]

# 서브플롯 설정 (3x6 = 9개 이미지 * 원본 & 세그먼트)
fig, axes = plt.subplots(3, 6, figsize=(15, 10))

valid_count = 0  # 정상적으로 로드된 이미지 개수

for img_path in sample_images:
    image = cv2.imread(img_path)

    # 이미지 로드 실패 시 건너뛰기
    if image is None:
        continue

    # 원본 이미지를 RGB로 변환 (OpenCV는 기본적으로 BGR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # LAB 색공간 변환
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # 픽셀을 2D 배열로 변환
    pixels = image_lab.reshape((-1, 3))

    # K-Means 클러스터링 (머리카락=0, 두피=1로 클러스터 2개 설정)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels)

    # 밝기 기준으로 두피(1), 머리카락(0)
    label_0_mean = np.mean(pixels[labels == 0])
    label_1_mean = np.mean(pixels[labels == 1])

    if label_0_mean > label_1_mean:
        labels = np.where(labels == 0, 1, 0)  # 두피 = 1, 머리카락 = 0으로 변환
    else:
        labels = np.where(labels == 1, 1, 0)  # 반대 상황 처리

    # 원본 형태로 복원
    segmented_image = labels.reshape(image.shape[:2])

    # 머리카락과 두피 면적(픽셀 수) 계산
    hair_pixels = np.sum(segmented_image == 0)
    scalp_pixels = np.sum(segmented_image == 1)
    total_pixels = hair_pixels + scalp_pixels

    hair_ratio = hair_pixels / total_pixels * 100  # 머리카락 비율 (%)
    scalp_ratio = scalp_pixels / total_pixels * 100  # 두피 비율 (%)

    # 머리카락 & 두피 비율 출력
    print(f"이미지 이름 : {os.path.basename(img_path)}")
    print(f"머리카락 비율: {hair_ratio:.2f}% ({hair_pixels}개)")
    print(f"두피 비율: {scalp_ratio:.2f}% ({scalp_pixels}개)\n")

    # 원본 이미지 출력
    ax1 = axes[valid_count // 3, (valid_count % 3) * 2]
    ax1.imshow(image_rgb)
    ax1.set_title(f"Original {valid_count+1}")
    ax1.axis("off")

    # K-Means 결과 출력
    ax2 = axes[valid_count // 3, (valid_count % 3) * 2 + 1]
    ax2.imshow(segmented_image, cmap='binary')
    ax2.set_title(f"Segmented {valid_count+1}")
    ax2.axis("off")

    valid_count += 1

    if valid_count >= 9:
        break

plt.tight_layout()
plt.show()

