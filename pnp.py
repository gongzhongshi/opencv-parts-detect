import cv2
import numpy as np

# 读取图片
img = cv2.imread("coin.jpg")
img_show = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5),0)
edge = cv2.Canny(blur,50,150)
contours,_ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 硬币真实物理尺寸：一元硬币直径25mm，定义世界坐标4个角点
world_points = np.array([
    [-12.5, -12.5, 0],
    [12.5, -12.5, 0],
    [12.5, 12.5, 0],
    [-12.5, 12.5, 0]
], dtype=np.float32)

# 相机内参（简易估算，可直接使用）
camera_matrix = np.array([[800,0,320],
                          [0,800,240],
                          [0,0,1]], dtype=np.float32)
dist_coeffs = np.zeros((4,1))

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area>300:
        x,y,w,h = cv2.boundingRect(cnt)
        # 获取矩形四个角点（图像UV坐标）
        img_points = np.array([
            [x,y], [x+w,y], [x+w,y+h], [x,y+h]
        ], dtype=np.float32)
        # PnP求解旋转rvec，平移tvec
        ret, rvec, tvec = cv2.solvePnP(world_points, img_points, camera_matrix, dist_coeffs)
        print("====PnP解算结果====")
        print("旋转向量 rvec：\n",rvec)
        print("平移向量 tvec（单位mm）：\n",tvec)
        # 画框+标记角点
        cv2.rectangle(img_show,(x,y),(x+w,y+h),(0,255,0),2)
        for p in img_points:
            cv2.circle(img_show, (int(p[0]),int(p[1])),4,(0,0,255),-1)

cv2.imwrite("pnp_result.jpg",img_show)
print("PnP效果图已保存 pnp_result.jpg")
