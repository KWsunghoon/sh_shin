import cv2
import numpy as np
import os
import lip_tracking
import load_path
import matplotlib.pyplot as plt
import tensorflow as tf
import glob

class save_frame:
    def pil_to_cv(pil_image):
        """
        Returns a copy of an image in a representation suited for OpenCV
        :param pil_image: PIL.Image object
        :return: Numpy array compatible with OpenCV
        """
        return np.array(pil_image)[:, :, ::-1]

    def mouth_frame(data, data_length, folder):

        fps = 25
        size = (50, 100)
        try:
            out = cv2.VideoWriter(folder + "1.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
            for i in range(data_length):
                # writing to a image array
                out.write(data[i])
            out.release()
            # for frame in data:
            #     writer.writeFrame(frame)

            # writer.close()
        except:
            pass

    def mouth_image(data, data_length, folder):
        try:
            for i in range(data_length):
                cv2.imwrite(folder + '/' + str(i).zfill(2) + ".jpg", data[i])
        except:
            pass
  
    def make_folder(image_path, mp4_class, num):
        try:
            os.makedirs(os.path.join(image_path, mp4_class))
        except:
            print("The folder already exists. ")
        folder = image_path + "/" + mp4_class

        try:
            folder = os.makedirs(os.path.join(folder, str(num).zfill(4)))
        except:
            print("The folder already exists. ")
        folder = image_path + "/" + mp4_class + "/" + str(num).zfill(4)
        return folder

    def mp4_to_img(frames, word_class, image_path):
        # word_label.append(word_class)

        for i in range(np.shape(frames)[0]):
            try:
                mouth = lip_tracking.lip_tracking.face_detection(frames[i])
                data, length = load_path.load_path.set_data(mouth)
                # total_data.append(data)
                folder = save_frame.make_folder(image_path, word_class, i)
                save_frame.mouth_image(data, length, folder)
            except:
                pass

    def dataset(path, num_mp4, list_mp4, image_path, train_path):

        for i, _class in zip(range(num_mp4), list_mp4):
            label_loc = os.path.join(path, _class)
            loc = os.path.join(label_loc, train_path)
            if (len(os.listdir(loc))>0):
                frame = load_path.load_path.load_dir(loc, _class)
            save_frame.mp4_to_img(frame, _class, image_path)

# mp4_path = "Lip_Reading_in_the_wild/lipread_mp4"
# image_path = "Lip_Reading_in_the_wild/lipread_test"
# train = 'test'
# mp4_path, list_mp4 = load_path.load_path.total_path()
# save_frame.dataset(mp4_path, len(list_mp4), list_mp4, image_path, train)

mp4_path = "Lip_Reading_in_the_wild/sub_mp4"
image_path = "Lip_Reading_in_the_wild/lipread_val"
train = 'val'
mp4_path, list_mp4 = load_path.load_path.total_path()
save_frame.dataset(mp4_path, len(list_mp4), list_mp4, image_path, train)