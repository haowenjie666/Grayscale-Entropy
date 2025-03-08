
import os
import numpy as np
import cv2
import cupy as cp
from openpyxl import Workbook

def calculate_2d_entropy(image, a, b, n):
    # 将图像转为GPU上的数组
    image = cp.asarray(image)  # 使用 CuPy 转换为 GPU 数组
    image = cp.clip(image, a, b)

    # 初始化联合概率矩阵
    joint_prob_matrix = cp.zeros((b-a+1, b-a+1), dtype=cp.float32)

    # 遍历图像计算 n x n 领域内的相邻像素对的联合概率
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            center_pixel = image[i, j] - a
            # 遍历 n x n 领域内的像素
            for di in range(-n//2, n//2 + 1):
                for dj in range(-n//2, n//2 + 1):
                    ni, nj = i + di, j + dj
                    # 确保相邻像素在图像边界内
                    if 0 <= ni < image.shape[0] and 0 <= nj < image.shape[1]:
                        adjacent_pixel = image[ni, nj] - a
                        joint_prob_matrix[center_pixel, adjacent_pixel] += 1

    # 转换为概率
    joint_prob_matrix /= joint_prob_matrix.sum()

    # 计算二维熵
    entropy = -cp.sum(joint_prob_matrix * cp.log2(joint_prob_matrix + 1e-9))  # 加上小量避免对0取对数

    return entropy

def batch_process_images(image_dir, a, b, n):
    results = []

    # 遍历图像目录
    for filename in os.listdir(image_dir):
        if filename.endswith(".tif"):  # 假设图像为tif格式
            image_path = os.path.join(image_dir, filename)
            image = cv2.imread(image_path, -1)  # 读取为16位灰度图像

            # 计算二维熵
            entropy = calculate_2d_entropy(image, a, b, n)
            results.append((filename, entropy))

    return results

def save_results_to_excel(results, excel_path):
    wb = Workbook()
    ws = wb.active
    ws.title = "Entropy Results"
    ws.append(["Image Name", "2D Entropy"])

    for filename, entropy in results:
        ws.append([filename, entropy])

    wb.save(excel_path)

def main():
    image_dir = ""  # Specify the image directory path here
    a, b = a, b  # 16-bit image grayscale range
    n = n  # Define the n x n neighborhood size, n should be an odd number
    excel_path = ""  # Specify the path to save the results Excel file

    results = batch_process_images(image_dir, a, b, n)
    save_results_to_excel(results, excel_path)

    print(f"Processed {len(results)} images. Results are saved to '{excel_path}'.")

if __name__ == "__main__":
    main()
