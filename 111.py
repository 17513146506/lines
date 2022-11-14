import time
import glob
import os
import cv2
import numpy as np
from PIL import Image
from unet import Unet
unet = Unet()
mode = "predict"
lower_blue = np.array([110, 50, 50])
upper_blue = np.array([130, 255, 255])
kernel1 = np.ones((13, 13), np.uint8)
kernel2 = np.ones((7, 7), np.uint8)
kernel3 = np.ones((37, 37), np.uint8)
font = cv2.FONT_HERSHEY_SIMPLEX
fourcc = cv2.VideoWriter_fourcc(*'XVID')
i = 0
WSI_MASK_PATH = 'out'  # 存放图片的文件夹路径
paths = glob.glob(os.path.join(WSI_MASK_PATH, '*.jpg'))
paths.sort()
for path in paths:
        frame = cv2.imread(path)
        t1 = time.time()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        closing1 = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel1)
        opening = cv2.morphologyEx(closing1, cv2.MORPH_OPEN, kernel2)
        closing2 = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel3)
        dilation = cv2.dilate(closing2, kernel2, iterations=1)
        roi = closing2[920:1080, 0:1920]
        # cv2.line(frame, (0, 1000), (1920, 1000), (0, 255, 0), 3)
        contours, hierarchy = cv2.findContours(roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        x, y, w, h = cv2.boundingRect(cnt)
        # cv2.rectangle(frame,(x,y+920),(x+w,y+920+h),(0,255,0),2)
        cx=int((x+x+w)/2)
        cy=int((y+920+y+920+h)/2)
        cv2.circle(frame, (cx,cy), 10, (0, 250, 0), 10)
        # cv2.line(frame, (x + int(w / 2), 980), (x + int(w / 2), 1020), (0, 255, 0), 2)
        # 格式转变，BGRtoRGB
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        # 转变成Image
        frame = Image.fromarray(np.uint8(frame))
        # 进行检测
        frame = np.array(unet.detect_image(frame))
        # RGBtoBGR满足opencv显示格式
        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
        offest_top = 0
        cv2.putText(frame, 'central:'+ str(int(w / 2))+'cm',(40, 100 ),font, 5, (0, 255, 0), 5,cv2.LINE_AA)
        cv2.imwrite('321/' + str(i) + '.jpg', frame)
        i += 1
        print(i)
print("保存图片完毕")