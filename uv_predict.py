import cv2
import numpy as np

cap = cv2.VideoCapture(0) # 打开摄像头
prev_x, prev_y = None, None

while True:
    ret, frame = cap.read()
    if not ret: break
    frame_show = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    edge = cv2.Canny(blur,50,150)
    contours,_ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area>2000:
            x,y,w,h = cv2.boundingRect(cnt)
            cx, cy = x+w//2, y+h//2 # 目标中心点UV坐标
            cv2.rectangle(frame_show,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.circle(frame_show,(cx,cy),5,(0,0,255),-1)

            # 线性预测下一时刻UV位置
            if prev_x is not None:
                pred_x = 2*cx - prev_x
                pred_y = 2*cy - prev_y
                cv2.circle(frame_show, (int(pred_x),int(pred_y)),5,(255,0,0),-1)
                cv2.putText(frame_show,f"Pred UV:({pred_x},{pred_y})",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2)
            prev_x, prev_y = cx, cy

    cv2.imshow("UV预测", frame_show)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
print("UV平面预测程序结束")
