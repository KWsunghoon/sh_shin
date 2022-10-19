from operator import contains
from imutils import face_utils
from matplotlib.image import imread
from numpy.lib.utils import source
from tqdm import trange
from keras import backend as K
from matplotlib import pyplot as plt
from tensorflow.keras import layers, models, optimizers, utils
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from tempfile import mkdtemp
import numpy as np
import imutils
import os
import dlib
import cv2
import skimage.transform
import shutil
import json
import re
import sys
import skvideo
import h5py
import skvideo.io
import argparse
import glob
import tensorflow as tf
import keras
skvideo.setFFmpegPath("C:/ProgramData/Anaconda3/ffmpeg/ffmpeg-2021-12-23-git-60ead5cd68-full_build/bin")

print(('TensorFlow version: {0}').format(tf.__version__))
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:

  try:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:

    print(e)


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:

    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:

    print(e)

train = 'train'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def total_path():
    # mp4_path = "Lip_Reading_in_the_wild/lipread_mp4"
    image_path = "Lip_Reading_in_the_wild/lipread_image"
    path = os.getcwd()
    image_path = os.path.join(path, image_path)
    # mp4_path = os.path.join(path, mp4_path)

    # os.chdir(mp4_path)
    list_mp4 = os.listdir(image_path)
    return image_path, list_mp4

def isWaveFile(name):
    if (name.find("mp4") > -1 or name.find(".MP4") > -1):
        return True
    else:
        return False

def isValidDir(loc):
    if (os.path.isdir(loc)):
        return True
    else:
        return False

def load_wav_dir(path, n_class):
    mp4_frames = []
    
    if(isValidDir(path)):
        list_mp4 = os.listdir(path)
        list_mp4 = [s for s in list_mp4 if (".mp4" in s) or (".MP4" in s)]        
        n_files = len(list_mp4)
        list_mp4 = [os.path.join(path, s) for s in list_mp4]

    elif(os.path.isfile(path)):
        file_wav = [path]

    for j in trange(n_files, desc="loading %s" % n_class):
        loc = os.path.join(path, list_mp4[j])

        try:
            frames = skvideo.io.vreader(loc)
            frames = np.array([frame for frame in frames])
            mp4_frames.append(frames)
        except Exception as e:
            print("\t",e,": file, {}".format(loc))
            pass

    return mp4_frames

def face_detection(frames):
    mouth_frames = []
    MOUTH_WIDTH = 100
    MOUTH_HEIGHT = 50
    HORIZONTAL_PAD = 0.19
    normalize_ratio = None

    try:
        for frame in frames:
            dets = detector(frame, 1)
            shape = None
            for k, d in enumerate(dets):
                shape = predictor(frame, d)
                i = -1
            if shape is None:
                return frames
            mouth_points = []
            for part in shape.parts():
                i += 1
                if i < 48: # 입 부분만 검출
                    continue
                mouth_points.append
                mouth_points.append((part.x,part.y))

                np_mouth_points = np.array(mouth_points)

            mouth_centroid = np.mean(np_mouth_points[:, -2:], axis=0)

            if normalize_ratio is None:
                mouth_left = np.min(np_mouth_points[:, :-1]) * (1.0 - HORIZONTAL_PAD)
                mouth_right = np.max(np_mouth_points[:, :-1]) * (1.0 + HORIZONTAL_PAD)

                normalize_ratio = MOUTH_WIDTH / float(mouth_right - mouth_left)

            new_img_shape = (int(frame.shape[0] * normalize_ratio), int(frame.shape[1] * normalize_ratio))
            resized_img = skimage.transform.resize(frame, new_img_shape)

            mouth_centroid_norm = mouth_centroid * normalize_ratio

            mouth_l = int(mouth_centroid_norm[0] - MOUTH_WIDTH / 2)
            mouth_r = int(mouth_centroid_norm[0] + MOUTH_WIDTH / 2)
            mouth_t = int(mouth_centroid_norm[1] - MOUTH_HEIGHT / 2)
            mouth_b = int(mouth_centroid_norm[1] + MOUTH_HEIGHT / 2)

            mouth_crop_image = resized_img[mouth_t:mouth_b, mouth_l:mouth_r]
            mouth_crop_image = np.array(mouth_crop_image)
            mouth_crop_image =  mouth_crop_image.astype(np.float32)
            mouth_crop_image = cv2.cvtColor(mouth_crop_image, cv2.COLOR_BGR2RGB)


            mouth_frames.append(mouth_crop_image)
    except Exception:
        pass
    return mouth_frames


def set_data(frames):
    try:
        data_frames = np.array(frames)
        if (K.image_data_format() == 'channels_first'):
            data_frames = np.rollaxis(data_frames, 3)
        data = data_frames
        length = len(data)
        return data, length
    except:
        pass

def pil_to_cv(pil_image):
    """
    Returns a copy of an image in a representation suited for OpenCV
    :param pil_image: PIL.Image object
    :return: Numpy array compatible with OpenCV
    """
    return np.array(pil_image)[:, :, ::-1]

def mouth_frame(data, data_length, folder):
    writer = skvideo.io.FFmpegWriter(folder + "1.mp4")
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
            image = cv2.convertScaleAbs(data[i], alpha=(255.0))
            cv2.imwrite(folder + str(i) + ".jpg", image)
    except:
        pass

def make_folder(image_path, mp4_class, num):
    try:
        os.makedirs(os.path.join(image_path, mp4_class))
    except:
        print("The folder already exists. ")
    folder = image_path + "/" + mp4_class

    try:
        os.makedirs(os.path.join(folder, str(num)))
    except:
        print("The folder already exists. ")
    folder = folder + "/" + str(num) + "/"

    return folder

def mp4_to_img(frames, word_class, image_path):
    # word_label.append(word_class)

    for i in range(np.shape(frames)[0]):
        try:
            mouth = face_detection(frames[i])
            data, length = set_data(mouth)
            # total_data.append(data)
            folder = make_folder(image_path, word_class, i)
            mouth_image(data, length, folder)
        except:
            pass

def dataset(path, num_mp4, list_mp4, image_path):

    for i, _class in zip(range(num_mp4), list_mp4):
        label_loc = os.path.join(path, _class)
        loc = os.path.join(label_loc, train)
        if (len(os.listdir(loc))>0):
            frame = load_wav_dir(loc, _class)
        mp4_to_img(frame, _class, image_path)

image_path, list_mp4 = total_path()

# list_mp4 = [s for s in list_mp4 if(isValidDir(s) and not isWaveFile(s))]

num_mp4 = len(list_mp4)

# dataset(mp4_path, num_mp4, list_mp4, image_path)
X = []
Y = []

image_h = 112
image_w = 112

# for i, path in enumerate(glob.glob(image_path + "/*")):
#     label = [0 for i in range(4)]
#     buffer_1 = []
#     for j, word_path in enumerate(glob.glob(path + "/*")):
#         buffer_2 = []
#         for k, word in enumerate(glob.glob(word_path + "/*")):
#             img = cv2.imread(word)
#             img = cv2.resize(img, None, fx = image_h/img.shape[1], fy = image_w/img.shape[0])
#             img = img / 255
#             buffer_2.append(img)
#         if (np.shape(buffer_2) == (29, 112, 112, 3)):
#             buffer_1.append(buffer_2)
#         else:
#             pass
#         if len(buffer_1) == 200:
#             break
#         Y.append(label)
#     X.append(buffer_1)
#     Y.append(os.path.basename(path))
#     if i == 3:
#         break
#     print("!.")

def make_path(file_name, directory = '', is_make_temp_dir = False):
    try:
        path = '\\\\223.194.32.78\\Digital_Lab\\Personals\\Sunghoon_Shin\\lipread_data\\'
        path = os.path.join(path)
        if is_make_temp_dir is True:
            directory = mkdtemp(path)
        if len(directory) >= 2 and not os.path.exists(path + directory):
            os.makedirs(path + directory)
    except:
        pass
    return os.path.join(path, directory, file_name)

def make_memap(mem_file_name, np_to_copy):
    memmap_configs = dict()
    np_to_copy = np.array(np_to_copy)
    memmap_configs['shape'] = shape = tuple(np_to_copy.shape)
    memmap_configs['dtype'] = dtype = str(type(np_to_copy))
    json.dump(memmap_configs, open(mem_file_name + '.conf', 'w'))
    mm = np.memmap(mem_file_name, mode = 'w+', shape=shape, dtype='float32')
    mm[:] = np_to_copy[:]
    mm.flush()
    return mm

def read_memmap(mem_file_name):
    with open(mem_file_name + '.conf', 'r') as file:
        memmap_configs = json.load(file)
        return np.memmap(mem_file_name, mode = 'r+', \
                        shape = tuple(memmap_configs['shape']))

cnt = 0

def file_remove(path):
    path = os.path.join(path)
    if (len(os.listdir(path)) == 29):
        pass
    else:
        print(path)

# for i, path in enumerate(glob.glob('C:/ProgramData/Anaconda3/Lip_Reading_in_the_Wild/sub' + "/*")):
#     for j, word_path in enumerate(glob.glob(path + "/*")):
#             file_remove(word_path)


# X_file_name = make_path("X.dat", directory='data')

# XX = read_memmap(X_file_name)

# for folder in os.listdir(image_path):
#     a = 0
#     cnt = cnt + 1
#     X = []
#     Y = []
#     for image_folder in os.listdir(os.path.join(image_path, folder)):
#         sub_X = []
#         sub_Y = []
#         a = a + 1
#         for image in os.listdir(os.path.join(image_path, folder, image_folder)):
#             path_to_image = os.path.join(image_path, folder, image_folder, image)
#             image = cv2.imread(path_to_image)
#             # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#             image = cv2.resize(image, None, fx = image_h/image.shape[1], fy = image_w/image.shape[0])
#             # image = np.expand_dims(image, axis=2)
#             label = re.findall(r'\w+\w+', folder)[0]
#             image = image/255
#             sub_X.append(image)
#             sub_Y.append(label)
#         if (np.shape(sub_X) == (29, 112, 112, 3) and len(sub_Y) == (29)):
#             X.append(sub_X)
#             Y.append(label)
#     # X_file_name = make_path("X_" + str(cnt) + ".dat", directory='data')
#     # Y_file_name = make_path("Y_" + str(cnt) + ".dat", directory='data')
#     # XX = read_memmap(X_file_name)
#     # m_X = make_memap(X_file_name, X)
#     # m_Y = make_memap(Y_file_name, Y)
#     dataset = tf.data.Dataset.from_tensor_slices((X, Y))
#     for i in dataset:
#         print(i.numpy())
#     f = h5py.File('X' + str(1) + '.hdf5', 'w')
#     f.create_dataset('X', data = X)
#     print("!")
#         # if a == 100:
#         #     break
#     # if cnt == 3:
#     #     break
test = 'C:/ProgramData/Anaconda3/Lip_Reading_in_the_Wild/sub/'
path = os.path.join(image_path + '/ABOUT')
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test,# labels = 'inferred',
        validation_split = None,
        subset = None,
        #label_mode = 'categorical',
        shuffle = False,
        batch_size = 29,
        color_mode = 'rgb',
        image_size = (112, 112))

# 클래스 네임 라벨링 하기!!

# for image_batch, labels_batch in train_ds:
#     print(image_batch.shape)
#     print(labels_batch.shape)

class_names = train_ds.class_names

# plt.figure(figsize=(10, 10))

# for images, labels in train_ds.take(100):
#     for i in range(29):
#         plt.imshow(images[i].numpy().astype("uint8"))
#         plt.title(class_names)
#         plt.axis("off")
#         plt.show()

normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)

# train_ds = train_ds.unbatch()
images = list(train_ds.map(lambda x, y: normalization_layer(tf.reshape(x, shape=(-1, 29, 112, 112, 3)))))
labels = list(train_ds.map(lambda x, y: y))


# train_ds = tf.keras.layers.experimental.preprocessing.Resizing(train_ds, (-1,29,112,112,3))
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(tf.reshape(x, shape=(-1, 29, 112, 112, 3))), y))
# image_batch, labels_batch = next(iter(normalized_ds))
batch_size = 29
AUTOTUNE = tf.data.experimental.AUTOTUNE
# dataset_ = tf.data.Dataset.from_tensor_slices((images, labels))
def prepare(ds, shuffle=False):
  # Resize and rescale all datasets
    ds = ds.map(lambda x, y: (normalization_layer(tf.reshape(x, shape=(29, 112, 112, 3))), y), 
                num_parallel_calls=AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(1000)

    # Batch all datasets
    ds = ds.batch(len(class_names))

    # Use buffered prefecting on all datasets
    return ds#.prefetch(buffer_size=AUTOTUNE)

AUTOTUNE = tf.data.AUTOTUNE

# train_ds = normalized_ds.cache().prefetch(buffer_size=AUTOTUNE)
# train_ds = prepare(train_ds, shuffle=False)
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

# Y_train = to_categorical(Y_train, cnt)
# Y_test = to_categorical(Y_test, cnt)

model = tf.keras.Sequential()

model.add(layers.InputLayer((29, 112, 112, 3)))
model.add(layers.Conv3D(48, kernel_size = (3, 3, 3), padding = 'same'))
model.add(layers.MaxPool3D(pool_size = (3, 3, 3), padding = 'same'))

model.add(layers.Conv3D(256, kernel_size = (3, 3, 3), padding = 'same'))
model.add(layers.MaxPool3D(pool_size = (3, 3, 3), padding = 'same'))

model.add(layers.Conv3D(512, kernel_size = (3, 3, 3), padding = 'same'))

model.add(layers.Conv3D(512, kernel_size = (3, 3, 3), padding = 'same'))
model.add(layers.BatchNormalization())
model.add(layers.Flatten())

model.add(layers.Dropout(0.5))

model.add(layers.Dense(32, activation='relu'))

model.add(layers.Dense(2, activation = 'sigmoid'))

model.summary()

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

history = model.fit(normalized_ds, batch_size = 32, epochs = 50, verbose = 1)

model.save('model.h5')

print("!")