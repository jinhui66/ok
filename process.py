import cv2  
import matplotlib.pyplot as plt  
import numpy as np  
import os

def bi(img, circle_kernel, iterations):
    # 应用腐蚀操作  
    eroded_img = cv2.erode(img, circle_kernel, iterations=iterations)  

    # 应用膨胀操作  
    dilated_img = cv2.dilate(eroded_img, circle_kernel, iterations=iterations)
    return dilated_img

def kai(img, circle_kernel, iterations):
    # 应用膨胀操作  
    dilated_img = cv2.dilate(img, circle_kernel, iterations=iterations)
    # 应用腐蚀操作  
    eroded_img = cv2.erode(dilated_img, circle_kernel, iterations=iterations)  
    return eroded_img

def process(img_path, output_dir="./data/processed"):
    # 读取灰度图像  
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  
    # print(img)
    # 检查图像是否成功加载  
    if img is None:  
        print("Error: Could not open or find the image.")  
    else:  
    
        # 对图像进行二值化  
        # 设置阈值和最大值  
        threshold_value = 127  # 你可以根据需要调整这个值  
        max_value = 1
        _, binary_img = cv2.threshold(img, threshold_value, max_value, cv2.THRESH_BINARY)  


        kernel_size1 = (3,3)
        circle_kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size1)

        kernel_size2 = (5,5)
        circle_kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size2)

        # 应用闭运算
        img2 = bi(binary_img, circle_kernel1, iterations=1)
        # 应用开运算
        img3 = kai(img2, circle_kernel2, iterations=1)
        img3 = img3 * 255  # 恢复到原始图像的灰度范围

        output_path = os.path.join(output_dir, os.path.basename(img_path))
        print(output_path)
        cv2.imwrite(output_path, img3)

if __name__ == "__main__":
    input_dir = "./data/origin_2"
    # input_dir = "../png_to_sever/png_to_sever/png_data_OctStruct/00DE272A-6A8C-4E5C-AB5E-14630E664CCA"
    output_dir = "./data/processed"
    os.makedirs(output_dir, exist_ok=True)
    for i in os.listdir(input_dir):
        img_path = os.path.join(input_dir, i)
        process(img_path, output_dir=output_dir)
        print(f"Processed {i} and saved to {output_dir}")