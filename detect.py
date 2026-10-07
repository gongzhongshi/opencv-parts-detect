import cv2
import numpy as np

# 读取图片
img = cv2.imread("coin.jpg")
img_copy = img.copy()

# 转灰度 + 模糊降噪
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 0)

# 边缘检测
edge = cv2.Canny(blur, 50, 150)

# 寻找轮廓
contours, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 遍历轮廓，筛选硬币轮廓
for cnt in contours:
    area = cv2.contourArea(cnt)
    # 过滤太小的噪点轮廓，根据你的图片可以微调数值
    if area > 300:
        # 获取包围矩形框
        x,y,w,h = cv2.boundingRect(cnt)
        # 在原图画框
        cv2.rectangle(img_copy, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(img_copy, "Coin", (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

# 保存效果图
cv2.imwrite("result.jpg", img_copy)
print("完成！效果图保存为 result.jpg")
