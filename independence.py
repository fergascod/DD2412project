import WRN
import tensorflow as tf
from utils import *
import os
import random
import time
import matplotlib.pyplot as plt
# from keras import backend as K

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

AUTO = tf.data.AUTOTUNE

BATCH_SIZE = 1
M = 3
num_labels = 10
batch_repetition = 1
train_batch_size = int(BATCH_SIZE / batch_repetition)
test_batch_size = int(BATCH_SIZE)
tr_data, test_data, classes, train_dataset_size, test_dataset_size, input_shape = load_CIFAR(train_batch_size, test_batch_size,
                                                                                             num_labels, batch_repetition, M,
                                                                                             AUTO)
n, k = 28, 10

model = WRN.build_model(input_shape, classes, n, k, M)
checkpoint_path="run/Cifar10/0_0_0_1/weights/final_weights.h5"
model.load_weights(checkpoint_path)

# print(model.summary())

layer_name="conv2d_10"

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

def plot_variance_activations(model, layer_name, x_train):
    layer_output=model.get_layer(layer_name).output
    intermediate_model=tf.keras.models.Model(inputs=model.input,outputs=layer_output)

    x_train = list(x_train)
    ex1, ex2 = random.sample(x_train, 2)
    act_1, act_2, act_3 = [], [], []

    for _ in range(3):
        ex3 = random.sample(x_train, 1)[0]
        example1 = [tf.convert_to_tensor(ex3),
                    tf.convert_to_tensor(ex1),
                    tf.convert_to_tensor(ex2)]
        example2 = [tf.convert_to_tensor(ex1),
                    tf.convert_to_tensor(ex3),
                    tf.convert_to_tensor(ex2)]
        example3 = [tf.convert_to_tensor(ex1),
                    tf.convert_to_tensor(ex2),
                    tf.convert_to_tensor(ex3)]
        stack1 = tf.stack(example1)[None, :, :,:]
        stack2 = tf.stack(example2)[None, :, :,:]
        stack3 = tf.stack(example3)[None, :, :,:]

        act_1.append(tf.reshape(intermediate_model(stack1),[-1]))
        act_2.append(tf.reshape(intermediate_model(stack2),[-1]))
        act_3.append(tf.reshape(intermediate_model(stack3),[-1]))


    act_tfs = [tf.stack(act) for act in [act_1, act_2, act_3]]
    var_1, var_2, var_3 = [tf.math.reduce_std(act_tf, axis=0) for act_tf in act_tfs]
    print(var_1, var_2, var_3)
    plt.scatter(var_1, var_2)
    plt.show()

plot_variance_activations(model, layer_name, x_train)
