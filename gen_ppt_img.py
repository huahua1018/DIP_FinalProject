
import cv2
import numpy as np

# 讀取兩張圖片
img1_pt = './root_place365_val_train1/4.png'  # 背景層（主要場景）
img2_pt = './root_place365_val_train1/5.png'  # 反射層（模擬反射）

img1 = cv2.imread(img1_pt)
img2 = cv2.imread(img2_pt)

# 確保兩張圖片大小相同
if img1.shape != img2.shape:
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

# 對反射層進行模糊處理
#reflection_layer = cv2.GaussianBlur(img2, (15, 15), 10)  # 調整模糊核大小和標準差
reflection_layer = img2  # 調整模糊核大小和標準差
# 定義不同的 alpha 值#
alphas = [0,0.1, 0.3, 0.5, 0.7, 0.9,1]
alphas = [0,0.1,0.2,0.3,0.4,0.45]
output_dir = './PPT_reflection_images/'  # 輸出目錄

# 確保輸出目錄存在
import os
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 生成並保存合成圖片
for i, alpha in enumerate(alphas):
    beta = 1 - alpha  # 反射層的權重
    synthetic_reflection = cv2.addWeighted(img1, alpha, reflection_layer, beta, 0)
    
    # 構建輸出檔案名稱
    output_path = os.path.join(output_dir, f'synthetic_alpha_{alpha:.1f}.png')
    
    # 保存圖片
    cv2.imwrite(output_path, synthetic_reflection)
    print(f"Saved: {output_path}")

print("All images saved successfully!")