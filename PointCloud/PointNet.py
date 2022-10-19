import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import initializers
from tensorflow.keras import regularizers

def conv_bn(x, filters):
    x = layers.Conv1D(filters, kernel_size = 1, padding = 'valid')(x)
    x = layers.BatchNormalization(momentum = 0.0)(x)
    return layers.Activation('relu')(x)

def dense_bn(x, filters):
    x = layers.Dense(filters)(x)
    x = layers.BatchNormalization(momentum = 0.0)(x)
    return layers.Activation('relu')(x)

def tnet(inputs, num_features):
    bias = initializers.Constant(np.eye(num_features))
    reg = Orthogonal_Regularizer(num_features)

    x = conv_bn(inputs, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 512)
    x = layers.GlobalAveragePooling1D()(x)
    x = dense_bn(x, 256)
    x = dense_bn(x, 128)
    x = layers.Dense(
        num_features * num_features,
        kernel_initializer = "zeros",
        bias_initializer = bias,
        activity_regularizer = reg,
    )(x)
    feat_T = layers.Reshape((num_features, num_features))(x)
    return layers.Dot(axes = (2, 1))([inputs, feat_T])

class Orthogonal_Regularizer(regularizers.Regularizer):
    def __init__(self, num_features, l2reg = 0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = tf.eye(num_features)
    
    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes = (2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))

