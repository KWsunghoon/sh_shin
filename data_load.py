import os
from tempfile import mkdtemp
import json
import numpy as np
import tensorflow as tf
import glob
import natsort
import cv2
import matplotlib.pyplot as plt

class data_load:

    def get_AUTOTUNE():
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        return AUTOTUNE

    def setting_list(path):

        data_list = natsort.natsorted(glob.glob(path))
        data_list = np.array(data_list)
        return data_list

    def get_label_from_path(path):
        return str(path.split('\\')[-3])

    def read_image(path):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = np.array(image)

        return cv2.resize(image, (112,112))#.reshape(112, 112, 3)


    def label_name(path):
        label_name_list = []
        data_list = data_load.setting_list(path)
        for path in data_list:
            label_name_list.append(data_load.get_label_from_path(path))
        unique_label_names = np.unique(label_name_list)
        return label_name_list, unique_label_names

    def onehot_encode_label(path):
        nothing, unique_label_names = data_load.label_name(path)
        onehot_label = unique_label_names == data_load.get_label_from_path(path)
        onehot_label = onehot_label.astype(np.uint8)
        return onehot_label

    def data_list(path):
        batch_size = 29
        data_height = 112
        data_width = 112
        channel_n = 3
        num_classes = 2

        data_list = data_load.setting_list(path)

        batch_image = np.zeros((batch_size, data_height, data_width, channel_n))
        batch_label = np.zeros((batch_size, num_classes))

        label_list = [data_load.onehot_encode_label(path).tolist() for path in data_list]

        # data_list = np.reshape(data_list, (-1, 29))
        # label_list = np.reshape(label_list, (-1, 29, 2))
        return data_list, label_list

    def tensor_data_load(path):
        data_list, label_list = data_load.data_list(path)

        dataset_ = tf.data.Dataset.from_tensor_slices((data_list, label_list))
        dataset_ = dataset_.map(lambda data_list, label_list: (tuple(tf.py_function(data_load._read_py_function, [data_list, label_list], [tf.float32, tf.uint8]))))

        dataset_ = dataset_.map(data_load._resize_function)
        # dataset_ = dataset_.repeat()
        dataset_ = dataset_.batch(29)

        dataset_ = dataset_.map(data_load.tensor_slice)

        dataset_ = dataset_.batch(32)

        dataset_ = dataset_.prefetch(tf.data.AUTOTUNE)

        for image_batch, label_batch in dataset_.take(5):
            print(image_batch.shape)
            print(label_batch.shape)
        return dataset_

    # for n, path in enumerate(data_list[:batch_size]):
    #     image = read_image(path)
    #     onehot_label = onehot_encode_label(path)
    #     batch_image[n, :, :, :] = image
    #     batch_label[n, :] = onehot_label


    def _read_py_function(path, label):
        image = data_load.read_image(path)
        label = np.array(label)
        return image.astype(np.float32), label

    def _resize_function(image_decoded, label):
        image_decoded.set_shape([None, None, None])
        image_resized = tf.image.resize(image_decoded, [112, 112])
        return image_resized, label

    def tensor_slice(image, label):
        label_ = label[1]
        return image, label_




class remove_file:

    def file_remove(path):
        path = os.path.join(path)

        if (len(os.listdir(path)) == 29):
            pass

        else:
            print(path)


class using_memmap:
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

class from_directory:
    def data_from_directory(data_dir):
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
                data_dir,
                validation_split = 0.2,
                subset = "training",
                #label_mode = 'categorical',
                shuffle = False,
                batch_size = 29,
                color_mode = 'rgb',
                image_size = (100, 50))
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
                data_dir,
                validation_split = 0.2,
                subset = "validation",
                #label_mode = 'categorical',
                shuffle = False,
                batch_size = 29,
                color_mode = 'rgb',
                image_size = (100, 50))


        return train_ds, val_ds

    def data_ds(data_dir):
        train_ds, val_ds = from_directory.data_from_directory(data_dir)
        # plt.figure(figsize=(10, 10))
        # for images, labels in train_ds.take(1):
        #     for i in range(9):
        #         ax = plt.subplot(3, 3, i + 1)
        #         plt.imshow(images[i].numpy().astype("uint8"))
        #         plt.title(train_ds.class_names[labels[i]])
        #         plt.axis("off")
        class_names = train_ds.class_names
        print(class_names)
        return train_ds, val_ds, class_names

    def data_mapping(train_ds, class_names):
        # data_list, label_list = data_load.data_list(data_dir)
        normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
        train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

        image_batch, labels_batch = next(iter(train_ds))
        first_image = image_batch[0]
        # Notice the pixels values are now in `[0,1]`.
        print(np.min(first_image), np.max(first_image))
        train_ds = train_ds.cache().prefetch(buffer_size = data_load.get_AUTOTUNE())
        train_ds = train_ds.map(data_load.tensor_slice)
        train_ds = train_ds.batch(1)
        train_ds = train_ds.shuffle(100)
        #train_ds = train_ds.repeat(10)
        # a = glob.glob('C:\\ProgramData\\Anaconda3\\Lip_Reading_in_the_Wild\\sub\\*\\*\\')
        # cnt = -1
        # try:
        #     for x, y in train_ds.take(1000):
        #         cnt = cnt + 1
        #         # print(x.shape)
        #         # print(y.shape)
        #         for i in range(2):

        #             if str(x[i][0][0][0].numpy()) == '[0.70980394 0.43137258 0.34509805]':
        #                 print(str(cnt) + "___" + str(i) + "__1__" + class_names[y[i]])
        #                 a_1 = x[i][0]
        #             if str(x[i][0][0][0].numpy()) == '[0.6509804  0.427451   0.40000004]':
        #                 print(str(cnt) + "___" + str(i) + "__2__" + class_names[y[i]])
        #                 a_2 = x[i][0]
        #             if str(x[i][0][0][0].numpy()) == '[0.6431373  0.5137255  0.43921572]':
        #                 print(str(cnt) + "___" + str(i) + "__3__" +  class_names[y[i]])
        #                 a_3 = x[i][0]
        #             if str(x[i][0][0][0].numpy()) == '[0.25882354 0.20000002 0.10980393]':
        #                 print(str(cnt) + "___" + str(i) + "__4__" + class_names[y[i]])
        #                 a_4 = x[i][0]
        #             if str(x[i][0][0][0].numpy()) == '[0.4156863  0.26666668 0.19215688]':
        #                 print(str(cnt) + "___" + str(i) + "__5__" + class_names[y[i]])
        #                 a_5 = x[i][0]
        #             if str(x[i][0][0][0].numpy()) == '[0.6431373  0.42352945 0.3647059 ]':
        #                 print(str(cnt) + "___" + str(i) + "__6__" + class_names[y[i]])
        #                 a_6 = x[i][0]


        #             if str(x[i][0][0][0].numpy()) == '[0.6117647  0.43137258 0.34117648]':
        #                 print(str(cnt) + "___" + str(i) + "__7__" + class_names[y[i]])
        #                 a_7 = x[i][0]
        #             if str(x[i][0][0][0].numpy()) == '[0.7843138  0.5058824  0.41960788]':
        #                 print(str(cnt) + "___" + str(i) + "__8__" + class_names[y[i]])
        #                 a_8 = x[i][0]
        #             if str(x[i][0][0][0].numpy()) == '[0.04313726 0.01568628 0.6666667 ]':
        #                 print(str(cnt) + "___" + str(i) + "__9__" + class_names[y[i]])
        #                 a_9 = x[i][0]
        #             if str(x[i][0][0][0].numpy()) == '[0.6666667  0.49803925 0.47450984]':
        #                 print(str(cnt) + "___" + str(i) + "__10__" + class_names[y[i]])
        #                 a_10 = x[i][0]

        #         # if claxss_names[y[0]] == 'ABSOLUTELY':
        #         #     print(cnt)
        # except Exception as e:
        #     print(e)
        #     pass
        # print(cnt)
        return train_ds
 
    def show(image, label):
        plt.figure()
        plt.imshow(image)
        plt.title(label.numpy().decode('utf-8'))
        plt.axis('off')

    def label_mapping(path):
        data_list, label_list = data_load.data_list(path)

        label = tf.data.Dataset.from_tensor_slices(label_list)
        #dataset_ = dataset_.map(lambda data_list, label_list: (tuple(tf.py_function(data_load._read_py_function, [data_list, label_list], [tf.float32, tf.uint8]))))
        return label