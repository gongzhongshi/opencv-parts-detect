import cv2
import numpy as np

# 相机内参，和PnP代码保持一致
camera_matrix = np.array([[800,0,320],
                          [0,800,240],
                          [0,0,1]], dtype=np.float32)
dist_coeffs = np.zeros((4,1))

# 硬币世界坐标，直径25mm
world_points = np.array([
    [-12.5, -12.5, 0],
    [12.5, -12.5, 0],
    [12.5, 12.5, 0],
    [-12.5, 12.5, 0]
], dtype=np.float32)

# 这里放入多张拍摄的硬币图片，你可以多拍几张，依次命名 img1.jpg img2.jpg
img_list = ["coin.jpg"]

for img_name in img_list:
    img = cv2.imread(img_name)
    if img is None:
        print(f"无法读取图片 {img_name}")
        continue
    img_show = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5),0)
    edge = cv2.Canny(blur,50,150)
    contours,_ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area>300:
            x,y,w,h = cv2.boundingRect(cnt)
            img_points = np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]], dtype=np.float32)
            # PnP求解三维旋转和平移向量
            ret, rvec, tvec = cv2.solvePnP(world_points, img_points, camera_matrix, dist_coeffs)

            print(f"\n====图片 {img_name} 的三维位姿结果====")
            print("旋转向量 rvec：")
            print(rvec)
            print("平移向量 tvec（单位mm，代表相机到硬币三维空间距离）：")
            print(tvec)

            # 在图上绘制角点
            cv2.rectangle(img_show,(x,y),(x+w,y+h),(0,255,0),2)
            for p in img_points:
                cv2.circle(img_show, (int(p[0]),int(p[1])),4,(0,0,255),-1)

    save_name = "3d_result_" + img_name
    cv2.imwrite(save_name, img_show)
    print(f"效果图已保存：{save_name}")

print("\n三维位姿预测程序运行完毕")
