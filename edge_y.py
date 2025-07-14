import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from process import kai, bi
from scipy.interpolate import UnivariateSpline

def sobel_edge_detection(img):
    # Y方向边缘检测（8-bit）
    img_y = cv2.Sobel(img, cv2.CV_8U, dx=0, dy=1)  # x方向不检测，y轴方向检测
    return img_y

def remove_top_pixel(img, pixels):
    img[:pixels, :] = 0
    return img

def extract_top_pixel(img):
    """保留每一列的第一个255像素，其余置0"""
    height, width = img.shape
    result = np.zeros_like(img)  # 全黑图像
    
    for x in range(width):
        col = img[:, x]  # 获取当前列的所有像素
        # 找到第一个255的位置
        first_255_pos = np.argmax(col == 255)  # 返回第一个满足条件的索引
        if col[first_255_pos] == 255:  # 确保确实找到了255
            result[first_255_pos, x] = 255  # 只保留第一个255
    return result

def find_all_white_columns(img, white_ratio_threshold=0.8):
    """
    找到所有白色像素占比超过一定比例的列索引
    :param img: 二值化图像（0和255）
    :param white_ratio_threshold: 白色像素占比阈值（默认80%）
    :return: 符合条件的列索引列表
    """
    height, width = img.shape
    white_columns = []
    
    for x in range(width):
        col = img[:, x]  # 获取当前列的所有像素
        white_pixels = np.count_nonzero(col == 255)  # 统计白色像素数量
        ratio = white_pixels / height  # 计算白色像素占比
        
        if ratio >= white_ratio_threshold:
            white_columns.append(x)
    return white_columns

def remove_columns_around_white(final_edges, white_columns):
    """在final_edges中，将white_columns及其左右50列置0"""
    height, width = final_edges.shape

    mask = np.ones_like(final_edges)  # 初始全1（表示保留）
    
    left = white_columns[0] - 50
    right = white_columns[-1] + 50

    mask[:, left:right+1] = 0  # 置0（表示移除）

    final_edges_cleaned = final_edges * mask  # 应用掩码
    return final_edges_cleaned, left, right

def remove_small_components(img, min_size=10):
    """
    通过连通域分析去除小连通区域（孤立点）
    :param img: 二值化边缘图像
    :param min_size: 最小连通区域大小（像素数）
    :return: 过滤后的图像
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(img)
    output = np.zeros_like(img)
    
    # 跳过背景（标签0）
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            output[labels == i] = 255
    return output

# 迭代删除离群点
def iterly_remove_single_point(top_index, threshold):
    current_index = top_index[0]
    current_threshold = threshold
    max_threshold = 10 * threshold
    # print(top_index[0])
    for i in range(len(top_index)):
        # print(current_index, top_index[i], current_threshold)
        if abs(top_index[i] - current_index) <= current_threshold and top_index[i] - current_index <=0:
            current_index = top_index[i]
        else:
            top_index[i] = -1
            current_threshold += threshold
            if current_threshold >= max_threshold:
                current_threshold = max_threshold

    return top_index

def img2index(edge):
    height, width = edge.shape
    top_index = np.zeros(width, dtype=np.int32)  # Initialize with 0 (no edge found)
    # print(left_index, right_index)
    for x in range(width):
        column = edge[:, x]  # Get the entire column
        y_positions = np.where(column == 255)[0]  # Find all y positions with 255
        
        if y_positions.size > 0:
            top_index[x] = y_positions[0]  # Store the first (topmost) 255 position
        else:
            top_index[x] = -1 # No edge found in this column
    return top_index

def valid_edge(edge, left_index, right_index, threshold=3):
    height, width = edge.shape
    top_index = img2index(edge)
    
    # Separate left and right edges
    left_top = top_index[:left_index][::-1]
    right_top = top_index[right_index+1:]

    # print(left_index.shape, right_index.shape)
    # print(left_index, right_index)
    left_top = iterly_remove_single_point(left_top, threshold)
    right_top = iterly_remove_single_point(right_top, threshold)
    # print(left_top)
    top_index[:left_index] = left_top[::-1]
    top_index[right_index+1:] = right_top
    # print(top_index)

    new_edge = np.zeros_like(edge)
    for x in range(width):
        if top_index[x] != -1:  # If edge exists in this column
            new_edge[top_index[x], x] = 255

    return new_edge

def fit_circle(xs, ys):
    """
    使用最小二乘法拟合圆（x - a)^2 + (y - b)^2 = r^2
    返回：圆心坐标(a, b) 和 半径 r
    """
    A = np.c_[2*xs, 2*ys, np.ones_like(xs)]
    f = xs**2 + ys**2
    C, _, _, _ = np.linalg.lstsq(A, f, rcond=None)  # 解方程 A * C = f
    a, b, c = C
    r = np.sqrt(c + a**2 + b**2)
    return a, b, r

def fill_circle_region(img, left_index, right_index, margin=20):
    """
    使用圆拟合中间缺失区域。
    :param img: 输入边缘图
    :param left_index: 缺口左边界列索引
    :param right_index: 缺口右边界列索引
    :param margin: 缺口两边取点的列宽
    """
    h, w = img.shape
    points = np.column_stack(np.where(img > 0))  # (y, x)
    if len(points) < 3:
        return img
    
    x = points[:, 1]
    y = points[:, 0]
    
    # 选择拟合点：缺口两侧 margin 范围内
    mask = ((x >= left_index - margin) & (x < left_index)) | ((x > right_index) & (x <= right_index + margin))
    x_fit = x[mask]
    y_fit = y[mask]
    
    if len(x_fit) < 3:
        return img  # 不足三点无法拟合圆
    
    # 拟合圆
    a, b, r = fit_circle(x_fit, y_fit)
    
    # 在缺口区域生成x坐标
    x_missing = np.arange(left_index, right_index + 1)
    # 对应的y为圆上的值： (x - a)^2 + (y - b)^2 = r^2 → y = b ± sqrt(r^2 - (x - a)^2)
    y_missing = b + np.sqrt(np.clip(r**2 - (x_missing - a)**2, 0, None))  # 取上半圆（适应图像坐标系）
    y_missing = y_missing.astype(int)
    y_missing = np.clip(y_missing, 0, h - 1)

    # 填充拟合曲线
    img_filled = img.copy()
    for xi, yi in zip(x_missing, y_missing):
        img_filled[yi, xi] = 255

    return img_filled

from sklearn.linear_model import LinearRegression
import numpy as np

def fill_linear_region(img, left_index, right_index, margin=5):
    """
    使用线性拟合中间缺失区域。
    :param img: 输入边缘图
    :param left_index: 缺口左边界列索引
    :param right_index: 缺口右边界列索引
    :param margin: 缺口两边取点的列宽
    """
    h, w = img.shape
    points = np.column_stack(np.where(img > 0))  # (y, x)
    if len(points) < 2:
        return img  # 如果没有足够的点来拟合线，直接返回原图
    
    x = points[:, 1]
    y = points[:, 0]
    
    # 选择拟合点：缺口两侧 margin 范围内
    mask = ((x >= left_index - margin) & (x < left_index)) | ((x > right_index) & (x <= right_index + margin))
    x_fit = x[mask]
    y_fit = y[mask]
    
    if len(x_fit) < 2:
        return img  # 不足两点无法进行线性拟合
    
    # 使用线性回归拟合直线
    reg = LinearRegression().fit(x_fit.reshape(-1, 1), y_fit)
    
    # 获取拟合直线的斜率和截距
    slope = reg.coef_[0]
    intercept = reg.intercept_
    
    # 在缺口区域生成x坐标
    x_missing = np.arange(left_index, right_index + 1)
    y_missing = slope * x_missing + intercept
    y_missing = y_missing.astype(int)
    y_missing = np.clip(y_missing, 0, h - 1)  # 保证 y 在图像范围内

    # 填充拟合曲线
    img_filled = img.copy()
    for xi, yi in zip(x_missing, y_missing):
        img_filled[yi, xi] = 255

    return img_filled

def detect_missing_region(img):
    """
    检测图像中缺失的区间（即纵向无白色像素点的区域）。
    :param img: 输入二值图像（值为0或255）
    :return: 缺失区间的列表，每个区间为一个(left_index, right_index)元组
    """
    h, w = img.shape
    missing_regions = []
    in_missing_region = False
    left_index = None

    # 遍历每一列，检查是否有白色像素点
    for x in range(w):
        if np.sum(img[:, x]) == 0:
            # 当前列为黑色，表示在缺失区域中
            if not in_missing_region:
                left_index = x
                in_missing_region = True
        else:
            if in_missing_region:
                # 结束一个缺失区域
                right_index = x - 1
                missing_regions.append((left_index, right_index))
                in_missing_region = False

    # 如果图像最后一列是缺失区域的右边界
    if in_missing_region:
        missing_regions.append((left_index, w - 1))

    # 可选：排除首尾的缺失区域
    if len(missing_regions) > 2:
        missing_regions = missing_regions[1:-1]

    return missing_regions


def detect_edge(img_path, output_dir="./data/edge_y", bin_dir="./data/bin"):
    # 读取灰度图像  
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  
    
    # 查找全白列
    white_columns = find_all_white_columns(img)
    # print(f"全白列索引: {white_columns}")
    
    # img = get_largest_component(img)

    # Sobel边缘检测
    edges = sobel_edge_detection(img)
    # 顶部的150像素值置为0
    edges = remove_top_pixel(edges, 150)

    # edges = remove_small_components(edges)
    # kernel_size1 = (3,3)
    # circle_kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size1)
    # edges = bi(edges, circle_kernel1, iterations=1)
    
    # 只保留每列第一个255像素
    final_edges = extract_top_pixel(edges)
    
    # 移除全白列及其左右50列
    if white_columns:
        final_edges, left_index, right_index = remove_columns_around_white(final_edges, white_columns)
    
    # 去除离散噪音
    final_edges = valid_edge(final_edges, left_index, right_index)

    # 补全底部空缺区域
    final_edges = fill_circle_region(final_edges, left_index, right_index)


    missing_regions = detect_missing_region(final_edges)

    # 对每个缺失区间应用线性拟合函数补全
    for left_index, right_index in missing_regions:
        final_edges = fill_linear_region(final_edges, left_index, right_index, margin=5)

    final_edges = fill_linear_region(final_edges, left_index, right_index)

    final_index = img2index(final_edges)
    # 保存 final_index 为 bin 文件（如：xxx.bin）
    bin_path = os.path.join(bin_dir, os.path.basename(img_path) + ".bin")
    final_index.tofile(bin_path)

    # 保存结果
    output_path = os.path.join(output_dir, os.path.basename(img_path))
    cv2.imwrite(output_path, final_edges)
    print(f"Processed {img_path} and saved to {output_path}")

if __name__ == "__main__":
    input_dir = "./data/processed"
    output_dir = "./data/edge_png"
    bin_dir = "./data/bin"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(bin_dir, exist_ok=True)

    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        detect_edge(img_path, output_dir, bin_dir)