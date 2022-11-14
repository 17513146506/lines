import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
from moviepy.editor import VideoFileClip
import glob
def merge_image_to_video(folder_name):
    fps = 25
    firstflag = True
    for f1 in os.listdir(folder_name):
        filename = os.path.join(folder_name, f1)
        frame = cv2.imread(filename)
        if firstflag == True:  # 读取第一张图时进行初始化，尺寸也近照些图
            firstflag = False
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            img_size = (frame.shape[1], frame.shape[0])
            video = cv2.VideoWriter("result.mp4", fourcc, fps, img_size)
        for index in range(fps):
            frame = cv2.imread(filename)
            frame_suitable = cv2.resize(frame, (img_size[0], img_size[1]), interpolation=cv2.INTER_CUBIC)
            video.write(frame_suitable)
    print('保存为mp4成功！')
    video.release()
if __name__ == '__main__':
    folder_name = r"detection"
    merge_image_to_video(folder_name)