import cv2
from diffusers import StableDiffusionGLIGENPipeline

# 加载图像
image = cv2.imread('asset/4.png')

# 定义新大小
new_size = (400, 250)  # 在这里指定新的图像宽度和高度

# 调整图像大小
resized_image = cv2.resize(image, new_size)

# 保存新图像
cv2.imwrite('asset/4.png', resized_image)