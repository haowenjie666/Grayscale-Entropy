import os
import numpy as np
import cv2
from openpyxl import Workbook
from multiprocessing import Pool


def calculate_2d_entropy(image, a, b, n):
    image = np.clip(image, a, b)
    joint_prob_matrix = np.zeros((b - a + 1, b - a + 1), dtype=np.float32)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            center_pixel = image[i, j] - a
            for di in range(-n // 2, n // 2 + 1):
                for dj in range(-n // 2, n // 2 + 1):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < image.shape[0] and 0 <= nj < image.shape[1]:
                        adjacent_pixel = image[ni, nj] - a
                        joint_prob_matrix[center_pixel, adjacent_pixel] += 1

    joint_prob_matrix /= joint_prob_matrix.sum()
    entropy = -np.sum(joint_prob_matrix * np.log2(joint_prob_matrix + 1e-9))
    return entropy


def process_image(args):
    image_path, a, b, n = args
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # 保证图像以原始格式读取
    entropy = calculate_2d_entropy(image, a, b, n)
    return (os.path.basename(image_path), entropy)


def batch_process_images(image_dir, a, b, n):
    image_paths = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir) if filename.endswith(".tif")]
    args = [(image_path, a, b, n) for image_path in image_paths]

    with Pool(n) as pool:  # 使用n个进程
        results = pool.map(process_image, args)

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
    image_dir = "xxx"
    a, b = lowerlimit,upperlimit 
    n = xx
    excel_path = "xxx"

    results = batch_process_images(image_dir, a, b, n)
    save_results_to_excel(results, excel_path)

    print(f"Processed {len(results)} images. Results are saved to '{excel_path}'.")


if __name__ == "__main__":
    main()
