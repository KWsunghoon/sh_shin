import tensorflow as tf

def image_left_right_flip(image):
    return tf.image.flip_left_right(image)
    # images_list = tf.unstack(video)
    # for i in range(len(images_list)):
    #     images_list[i] = tf.image.flip_left_right(images_list[i])
    # return tf.stack(images_list)


def video_left_right_flip(video):
    return tf.map_fn(image_left_right_flip, video)


def normalize(videos):
    # return videos * (1. / 255.) - 0.5
    return (videos - 127.5) / 50

def get_batch(paths, options):
    """Returns a data split of the RECOLA dataset, which was saved in tfrecords format.
    Args:
        split_name: A train/test/valid split name.
    Returns:
        The raw audio examples and the corresponding arousal/valence
        labels.
    """
    shuffle = options['shuffle']
    batch_size = options['batch_size']
    num_classes = options['num_classes']
    crop_size = options['crop_size']
    horizontal_flip = options['horizontal_flip']

    # root_path = Path(dataset_dir) / split_name
    # paths = [str(x) for x in root_path.glob('*.tfrecords')]

    filename_queue = tf.compat.v1.train.string_input_producer(paths, shuffle=shuffle)

    reader = tf.compat.v1.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = tf.compat.v1.parse_single_example(
        serialized_example,
        features={
            'video': tf.compat.v1.FixedLenFeature([], tf.string),
            'label': tf.compat.v1.FixedLenFeature([], tf.int64)
        }
    )

    video = tf.cast(tf.compat.v1.decode_raw(features['video'], tf.uint8), tf.float32) #/ 255.
    label = features['label']#tf.decode_raw(features['label'], tf.int64)

    # Number of threads should always be one, in order to load samples
    # sequentially.
    videos, labels = tf.compat.v1.train.batch(
        [video, label], batch_size, num_threads=1, capacity=1000, dynamic_pad=True)

    videos = tf.reshape(videos, (batch_size, 29, 118, 118, 1))
    #labels = tf.reshape(labels, (batch_size,  1))
    labels = tf.compat.v1.contrib.layers.one_hot_encoding(labels, num_classes)

    # if is_training:
        # resized_image = tf.image.resize_images(frame, [crop_size, 110])
        # random cropping
    if crop_size is not None:
        videos = tf.compat.v1.random_crop(videos, [batch_size, 29, crop_size, crop_size, 1])
    # random left right flip
    if horizontal_flip:
        sample = tf.compat.v1.random_uniform(shape=[], minval=0, maxval=1, dtype=tf.float32)
        option = tf.less(sample, 0.5)
        videos = tf.cond(option,
                         lambda: tf.map_fn(video_left_right_flip, videos),
                         lambda: tf.map_fn(tf.identity, videos))
            # lambda: video_left_right_flip(videos),
            # lambda: tf.identity(videos))
    videos = normalize(videos) #tf.cast(videos, tf.float32) * (1. / 255.) - 0.5

    return videos, labels