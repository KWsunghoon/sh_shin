import cv2
import os
import glob
from isort import file
import natsort
import shutil

image_path = 'C:/Users/Shin/anaconda3/Lip_Reading_in_the_wild/lipread_test'

folder = os.listdir(image_path)

for i in range(len(folder)):
    file_list = os.listdir(image_path + '/' + folder[i])
    for j in range(len(file_list)):
        file = os.listdir(image_path + '/' + folder[i] + '/' + file_list[j])
        if len(file) != 29:
            print(image_path + '/' + folder[i] + '/' + file_list[j])
            shutil.rmtree(image_path + '/' + folder[i] + '/' + file_list[j])
frame_array = []
cnt = 0
for i, path in enumerate(natsort.natsorted(glob.glob(image_path + "/*"))):
    cnt = 0
    for j, word_path in enumerate(natsort.natsorted(glob.glob(path + "/*"))):
        cnt = cnt + 1

        frame_array = []
        for k, word in enumerate(natsort.natsorted(glob.glob(word_path + "/*"))):
            img = cv2.imread(word)
            height, width, layers = img.shape
            size = (width,height)
            frame_array.append(img)

        out = cv2.VideoWriter(path + "/" + str(cnt).zfill(3)+".mp4", cv2.VideoWriter_fourcc(*'DIVX'), 29, size)
        for i in range(len(frame_array)):
            # writing to a image array
            out.write(frame_array[i])
    out.release()