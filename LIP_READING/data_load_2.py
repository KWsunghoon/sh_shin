import cv2
import os
import glob
import tensorflow as tf
import skvideo
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers

skvideo.setFFmpegPath("C:/Users/Shin/anaconda3/ffmpeg/ffmpeg-2022-04-07-git-607ecc27ed-full_build/bin")
import skvideo.io


def fix_gpu():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.InteractiveSession(config=config)


path = 'C:/Users/Shin/anaconda3/Lip_Reading_in_the_wild/lipread_image/'
class data_load:
    def video_list(file_path):
        data_list = glob.glob(file_path + '\\*\\*.mp4')
        label_list = os.listdir(file_path)
        return data_list, label_list
    def image_list(file_path):
        data_list = glob.glob(file_path + '\\*\\*')
        label_list = os.listdir(file_path)
        return data_list, label_list

    def video_read(file):
        # cap = cv2.VideoCapture(file)

        # while (cap.isOpened()):
        #     ret, frame = cap.read()

        # cap.release()
        videogen = skvideo.io.vread(file) / 255
        # try:
        #     if videogen
        # videogen = np.array(videogen)
        # videogen = cv2.resize(videogen, (50, 100))
        return videogen

    def image_read(file):
        image = []
        file_list = os.listdir(file)
        for i in file_list:
            img = cv2.imread(file + '/' + i, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB).astype(np.float64) / 255
            image.append(img)
        return image

    def get_data():
        file, label_list = data_load.video_list(path)
        # file, label_list = data_load.image_list(path)
        for i in file:
            label = str(i.split('\\')[-2])
            # label = label_to_index[label]
            videogen = data_load.video_read(i)
            # image = data_load.image_read(i)
            label = data_load.labeling(label, label_list)
            yield (videogen, label)

    def labeling(label, label_list):
        cnt = 0
        for i in label_list:
            if i == label:
                return cnt
            cnt = cnt + 1

    def scale(image, label):
        image = cv2.resize(image, (112, 112), interpolation =cv2.INTER_AREA)
        return image, label

    def one_hot(image, label):

        data_list, label_list = data_load.video_list(path)

        return image, label
 
    def fixup_shape(images, labels):
        images.set_shape([None, 29, 112, 112, 3])
        labels.set_shape([None])
        return images, labels
# file, label_list = data_load.video_list(path)
# for i in file:
#     label = str(i.split('\\')[-2])
#     # label = label_to_index[label]
#     videogen = data_load.video_read(i)
#     image = data_load.image_read(i)
#     label = data_load.labeling(label, label_list)

# dataset = tf.keras.preprocessing.image_dataset_from_directory(
#         path,
#         validation_split = None,
#         subset = None,
#         #label_mode = 'categorical',
#         shuffle = False,
#         batch_size = 29,
#         color_mode = 'rgb',
#         image_size = (112, 112))

dataset = tf.data.Dataset.from_generator(data_load.get_data, (tf.float64, tf.int64))
file, label_list = data_load.video_list(path)
# file, label_list = data_load.image_list(path)

# for i in file:
#     if skvideo.io.vread(i).shape == (29, 50, 100, 3):
#         pass
#     else:
#         print(i)+
#         os.remove(i)



AUTOTUNE = tf.data.AUTOTUNE

for image_batch, label_batch in dataset.take(1):
    print(image_batch.shape)
    print(label_batch.shape)

checkpoint_path = path + "cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)


dataset = dataset.shuffle(1000)
dataset = dataset.batch(4, drop_remainder =  True).map(data_load.fixup_shape)

model_ = tf.keras.Sequential()

model_.add(layers.InputLayer((29, 112, 112, 3)))

model_.add(layers.Conv3D(48, (3, 3, 3), padding='same'))
model_.add(layers.MaxPooling3D(pool_size=(3, 3, 3), padding='same'))

model_.add(layers.Conv3D(256, (3, 3, 3), padding='same'))
model_.add(layers.MaxPooling3D(pool_size=(3, 3, 3), padding='same'))

model_.add(layers.Conv3D(512, (3, 3, 3), padding='same'))
model_.add(layers.Conv3D(512, (3, 3, 3), padding='same'))
model_.add(layers.Flatten())
# model_.add(layers.Dense(1024, activation='relu'))
model_.add(layers.Dense(500, activation='softmax'))
model_.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_.summary()

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)
fix_gpu()

with tf.device("/device:GPU:0"):                                          
    model_.fit(dataset, epochs=20, steps_per_epoch = len(file) / 4, verbose=1, callbacks=[cp_callback])

model_.save("2022_05_18.h5")

# model_.add(layers.ZeroPadding3D(padding=(2, 2, 2)))
# model_.add(layers.Conv3D(32, (4, 4, 4), strides=(2, 2, 2), activation='relu', padding='same'))
# model_.add(layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same'))
# model_.add(layers.Dropout(0.5))

# model_.add(layers.ZeroPadding3D(padding=(2, 2, 2)))
# model_.add(layers.Conv3D(64, (4, 4, 4), strides=(2, 2, 2), activation='relu', padding='same'))
# model_.add(layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same'))
# model_.add(layers.Dropout(0.5))

# model_.add(layers.ZeroPadding3D(padding=(2, 2, 2)))
# model_.add(layers.Conv3D(96, (4, 4, 4), strides=(2, 2, 2), activation='relu', padding='same'))
# model_.add(layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same'))
# model_.add(layers.Dropout(0.5))
# model_.add(layers.Conv3D(192, (3, 3, 3), strides=(2, 2, 2), activation='relu', padding='same'))
# model_.add(layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same'))
# model_.add(layers.TimeDistributed(layers.Flatten()))
# model_.add(layers.Bidirectional(layers.GRU(256, return_sequences=True, kernel_initializer='Orthogonal'), merge_mode='concat'))
# model_.add(layers.Bidirectional(layers.GRU(256, return_sequences=False, kernel_initializer='Orthogonal'), merge_mode='concat'))
# model_.add(layers.Dense(64, kernel_initializer='he_normal'))
# model_.add(layers.Dense(len(label_list), activation='softmax'))

# model_.add(layers.Activation('softmax', name='softmax'))

# model_.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# model_.summary()

# with tf.device("/device:GPU:0"):
#     model_.fit(dataset, epochs=20, verbose=1)

# model_.save("2022_04_01.h5")

print("!")
