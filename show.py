import cv2
import matplotlib.pyplot as plt

id = "frame_001.png"
img_path = f"./data/origin/{id}"
edge_path = f"./data/edge_y/{id}"

# 读取图像（原始图像转为三通道，边缘检测结果保持单通道）
img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # 直接读取为三通道彩色图
edge = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)  # 边缘检测结果是单通道

# 将边缘检测结果转为三通道（红色显示）
edges_colored = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
edges_colored[:, :, :2] = 0  # 只保留红色通道（B=0, G=0, R=255）

# 叠加边缘到原图
merged_img = cv2.addWeighted(img, 0.5, edges_colored, 5, 0)

# 显示结果（Matplotlib需要RGB格式，OpenCV是BGR）
plt.imshow(cv2.cvtColor(merged_img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()