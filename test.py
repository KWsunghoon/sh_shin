import cv2
import os
import glob
import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
import skvideo
skvideo.setFFmpegPath('C:/Users/Shin/anaconda3/ffmpeg/ffmpeg-2022-04-07-git-607ecc27ed-full_build/bin')
import skvideo.io
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
path = 'C:/Users/Shin/anaconda3/Lip_Reading_in_the_Wild/lipread_image/'

class data_load:
    def video_list(file_path):
        data_list = glob.glob(file_path + '\\*\\*.mp4')
        label_list = os.listdir(file_path)
        return data_list, label_list

    def video_read(file):
        # cap = cv2.VideoCapture(file)

        # while (cap.isOpened()):
        #     ret, frame = cap.read()

        # cap.release()
        videogen = skvideo.io.vread(file)
        # try:
        #     if videogen
        # videogen = np.array(videogen)
        # videogen = cv2.resize(videogen, (50, 100))
        return videogen

    def get_data():
        file, label_lis6t = data_load.video_list(path)
        # label_to_index = {
        #     'ABOUT' : 0,
        #     'ABSOLUTELY' : 1
        # }
        for i in file:
            label = str(i.split('\\')[-2])
            # label = label_to_index[label]
            videogen = data_load.video_read(i)
            label = data_load.labeling(label, label_list)
            yield (videogen, label)
    def labeling(label, label_list):
        cnt = 0
        for i in label_list:
            if i == label:
                return cnt
            cnt = cnt + 1
    def scale(image, label):
        image = cv2.resize(image, (50, 100), interpolation =cv2.INTER_AREA)
        return image, label

    def one_hot(image, label):

        data_list, label_list = data_load.video_list(path)

        return image, label

    def fixup_shape(images, labels):
        images.set_shape([None, 29, 50, 100, 3])
        labels.set_shape([None])
        return images, labels
    def CTC(name, args):
        return Lambda(data_load.ctc_lambda_func, output_shape=(1,), name=name)(args)
    def ctc_lambda_func(args):
        y_pred, labels, input_length, label_length = args
        # From Keras example image_ocr.py:
        # the 2 is critical here since the first couple outputs of the RNN
        # tend to be garbage:
        # y_pred = y_pred[:, 2:, :]
        y_pred = y_pred[:, :, :]
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

# data_list, label_list = data_load.video_list(path)
dataset = tf.data.Dataset.from_generator(data_load.get_data, (tf.float64, tf.int64))
file, label_list = data_load.video_list(path)
# for i in file:
#     if skvideo.io.vread(i).shape == (29, 50, 100, 3):
#         pass
#     else:
#         print(i)+
#         os.remove(i)

for image_batch, label_batch in dataset.take(1):
    print(image_batch.shape)
    print(label_batch.shape)

# dataset = dataset.map(data_load.scale)
# data_list = data_load.video_read('C:\\ProgramData\\Anaconda3\\Lip_Reading_in_the_Wild\\sub\\')

dataset = dataset.shuffle(1500)
dataset = dataset.batch(2, drop_remainder = True).map(data_load.fixup_shape)
# dataset = dataset.repeat(5)

model_ = tf.keras.Sequential()

# input_data = layers.Input(name='the_input', shape=(29, 50, 100, 3), dtype='float32')

# zero1 = layers.ZeroPadding3D(padding=(1, 2, 2), name='zero1')(input_data)
# conv1 = layers.Conv3D(32, (3, 5, 5), strides=(1, 2, 2), kernel_initializer='he_normal', name='conv1')(zero1)
# batc1 = layers.BatchNormalization(name='batc1')(conv1)
# actv1 = layers.Activation('relu', name='actv1')(batc1)
# drop1 = layers.SpatialDropout3D(0.5)(actv1)
# maxp1 = layers.MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max1')(drop1)

# zero2 = layers.ZeroPadding3D(padding=(1, 2, 2), name='zero2')(maxp1)
# conv2 = layers.Conv3D(64, (3, 5, 5), strides=(1, 1, 1), kernel_initializer='he_normal', name='conv2')(zero2)
# batc2 = layers.BatchNormalization(name='batc2')(conv2)
# actv2 = layers.Activation('relu', name='actv2')(batc2)
# drop2 = layers.SpatialDropout3D(0.5)(actv2)
# maxp2 = layers.MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max2')(drop2)

# zero3 = layers.ZeroPadding3D(padding=(1, 1, 1), name='zero3')(maxp2)
# conv3 = layers.Conv3D(96, (3, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal', name='conv3')(zero3)
# batc3 = layers.BatchNormalization(name='batc3')(conv3)
# actv3 = layers.Activation('relu', name='actv3')(batc3)
# drop3 = layers.SpatialDropout3D(0.5)(actv3)
# maxp3 = layers.MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max3')(drop3)

# resh1 = layers.TimeDistributed(layers.Flatten())(maxp3)

# gru_1 = layers.Bidirectional(layers.GRU(256, return_sequences=True, kernel_initializer='Orthogonal', name='gru1'), merge_mode='concat')(resh1)
# gru_2 = layers.Bidirectional(layers.GRU(256, return_sequences=True, kernel_initializer='Orthogonal', name='gru2'), merge_mode='concat')(gru_1)

# dense1 = layers.Dense(28, kernel_initializer='he_normal', name='dense1')(gru_2)

# y_pred = layers.Activation('softmax', name='softmax')(dense1)

# labels = layers.Input(name='the_labels', shape=[32], dtype='float32')
# input_length = layers.Input(name='input_length', shape=[1], dtype='int64')
# label_length = layers.Input(name='label_length', shape=[1], dtype='int64')

# loss_out = data_load.CTC('ctc', [y_pred, labels, input_length, label_length])

# model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

# adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
# model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam)

# model.fit_generator(generator = dataset, epochs = 20, verbose = 1)
model_.add(layers.InputLayer((29, 50, 100, 3)))
model_.add(layers.ZeroPadding3D(padding=(2, 2, 2)))
model_.add(layers.Conv3D(32, (4, 4, 4), strides=(2, 2, 2), activation='relu', padding='same'))
model_.add(layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same'))
model_.add(layers.Dropout(0.5))

model_.add(layers.ZeroPadding3D(padding=(2, 2, 2)))
model_.add(layers.Conv3D(64, (4, 4, 4), strides=(2, 2, 2), activation='relu', padding='same'))
model_.add(layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same'))
model_.add(layers.Dropout(0.5))

model_.add(layers.ZeroPadding3D(padding=(2, 2, 2)))
model_.add(layers.Conv3D(96, (4, 4, 4), strides=(2, 2, 2), activation='relu', padding='same'))
model_.add(layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same'))
model_.add(layers.Dropout(0.5))
model_.add(layers.Conv3D(192, (3, 3, 3), strides=(2, 2, 2), activation='relu', padding='same'))
model_.add(layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same'))
model_.add(layers.TimeDistributed(layers.Flatten()))
model_.add(layers.Bidirectional(layers.GRU(256, return_sequences=True, kernel_initializer='Orthogonal'), merge_mode='concat'))
model_.add(layers.Bidirectional(layers.GRU(256, return_sequences=False, kernel_initializer='Orthogonal'), merge_mode='concat'))
model_.add(layers.Dense(64, kernel_initializer='he_normal'))
model_.add(layers.Dense(len(label_list), activation='softmax'))

model_.add(layers.Activation('softmax', name='softmax'))

# model_.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model_.summary()

# with tf.device("/device:GPU:0"):
#     model_.fit(dataset, epochs=20, verbose=1)
# CTC('ctc', [self.y_pred, self.labels, self.input_length, self.label_length])

list_callback = [
    keras.callbacks.EarlyStopping(
        monitor='val_pred_classify_accuracy',
        patience=30
    ),
    keras.callbacks.ModelCheckpoint(
        filepath=os.path.join("0406.h5"),
        monitor='val_pred_classify_accuracy',
        save_best_only=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_pred_detection_loss',
        factor=.1,
        patience=30,
        mode='auto',
        verbose=1
    )
]

opt = keras.optimizers.Adam(learning_rate = 0.001)

model_.compile(
    loss=['sparse_categorical_crossentropy', 'binary_crossentropy'],
    loss_weights=[1., .3],
    optimizer=opt,
    metrics=['accuracy']
)

hist = model_.fit(dataset, epochs = 3, callbacks = list_callback, verbose = 1)


model_.save('0406.h5')




# import keras
# # layer_input = layers.InputLayer((29, 50, 100, 3))

# layer_input = tf.keras.Input(shape = (29, 50, 100, 3))


# layer_conv2d_1 = layers.Conv3D(filters=32,
#                             kernel_size=5,
#                             strides=(1,1,1),
#                             padding='same',
#                             activation='relu',
#                             name="conv2d_1")(layer_input)
# layer_maxpooling_1 = layers.MaxPool3D(pool_size=(2, 2, 2),
#                                     strides=None,
#                                     padding='valid',
#                                     name="maxpool_1")(layer_conv2d_1)
# layer_conv2d_2 = layers.Conv3D(filters=64,
#                             kernel_size=5,
#                             strides=(1,1,1),
#                             padding='same',
#                             activation='relu',
#                             name="conv2d_2")(layer_maxpooling_1)
# layer_maxpooling_2 = layers.MaxPool3D(pool_size=(2, 2, 2),
#                                     strides=None,
#                                     padding='valid',
#                                     name="maxpool_2")(layer_conv2d_2)

# layer_flatten_1 = layers.Flatten()(layer_maxpooling_2)

# layer_dense_2 = layers.Dense(128)(layer_flatten_1)

# layer_drop_1 = layers.Dropout(rate=.2)(layer_dense_2)

# dense_classify = layers.Dense(len(label_list), name="pred_classify", activation='softmax')(layer_drop_1)
# dense_detec = layers.Dense(1., name="pred_detection")(layer_drop_1)


# model = tf.keras.Model(
#     inputs = [layer_input],
#     outputs = [dense_classify, dense_detec])

# list_callback = [
#     keras.callbacks.EarlyStopping(
#         monitor='val_pred_classify_accuracy',
#         patience=30
#     ),
#     keras.callbacks.ModelCheckpoint(
#         filepath=os.path.join("0406.h5"),
#         monitor='val_pred_classify_accuracy',
#         save_best_only=True
#     ),
#     keras.callbacks.ReduceLROnPlateau(
#         monitor='val_pred_detection_loss',
#         factor=.1,
#         patience=30,
#         mode='auto',
#         verbose=1
#     )
# ]

# opt = keras.optimizers.Adam(learning_rate = 0.001)

# model.compile(
#     loss=['sparse_categorical_crossentropy', 'binary_crossentropy'],
#     loss_weights=[1., .3],
#     optimizer=opt,
#     metrics=['accuracy']
# )

# hist = model.fit(dataset, epochs = 3, callbacks = list_callback, verbose = 1)


# model.save('0406.h5')
