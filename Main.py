import WRN
import tensorflow as tf
import numpy as np
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
shape= (x_train.shape[1:])
classes= tf.unique(tf.reshape(y_train, shape=(-1,)))[0].get_shape().as_list()[0]
# WRN params
n, k = 28, 10
# Number of subnetworks
M=3

model = WRN.build_model(shape, classes, n, k, M)
print(model.summary())
