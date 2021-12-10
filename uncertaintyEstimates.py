import WRN_old as WRN #For previous runs
# import WRN
import tensorflow as tf
from utils import *
import os
import random
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from keras import backend as K
import statistics

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def uncertaintyEstimation(model, x_test , y_test, classes, M):
    probs_wrong=[]
    probs_corrt=[]
    num_ex=50
    corr=0
    exs=0
    for x, y in zip(x_test[:num_ex], y_test[:num_ex]):
        exs+=1
        example=tf.stack([tf.convert_to_tensor(x) for _ in range(M)])[None, :, :, :, :]
        pred=tf.reduce_mean(tf.nn.softmax(model(example)), axis=1)[0]
        prob=float(tf.math.reduce_max(pred).numpy())
        if tf.math.argmax(pred)==y[0]:
            corr+=1
            probs_corrt.append(prob)
        else:
            probs_wrong.append(prob)
    print("Mean corrent:", statistics.mean(probs_corrt))
    print("Mean wrong:", statistics.mean(probs_wrong))
    print("Accuracy:", corr/exs)

def main():
    n, k, M = 28, 10, 3
    classes = 10
    input_shape = [M]+[32, 32, 3]
    model = WRN.build_model(input_shape, classes, n, k, M)
    checkpoint_path="run/Cifar10/0_0_0_1/weights/final_weights.h5"
    model.load_weights(checkpoint_path)
    train_batch_size, test_batch_size, batch_repetitions = 1, 1, 1
    (x_train_10, y_train_10), (x_test_10, y_test_10) = tf.keras.datasets.cifar10.load_data()
    test = list(zip(x_test_10, y_test_10))
    random.shuffle(test)
    x_test_10, y_test_10 = zip(*test)
    uncertaintyEstimation(model, x_test_10 , y_test_10, classes, M)

if __name__ == "__main__":
    main()
