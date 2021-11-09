import WRN
import tensorflow as tf
import numpy as np


class WarmUpPiecewiseConstantSchedule(
    tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Learning rate schedule used in the original paper
    """
    def __init__(self,
                 steps_per_epoch,
                 base_learning_rate,
                 decay_ratio,
                 decay_epochs,
                 warmup_epochs):
        super().__init__()
        self.steps_per_epoch = steps_per_epoch
        self.base_learning_rate = base_learning_rate
        self.decay_ratio = decay_ratio
        self.decay_epochs = decay_epochs
        self.warmup_epochs = warmup_epochs

    def __call__(self, step):
        lr_epoch = tf.cast(step, tf.float32) / self.steps_per_epoch
        learning_rate = self.base_learning_rate
        if self.warmup_epochs >= 1:
            learning_rate *= lr_epoch / self.warmup_epochs
        decay_epochs = [self.warmup_epochs] + self.decay_epochs
        for index, start_epoch in enumerate(decay_epochs):
            learning_rate = tf.where(
                lr_epoch >= start_epoch,
                self.base_learning_rate * self.decay_ratio ** index,
                learning_rate)
        return learning_rate

    def get_config(self):
        return {
            'steps_per_epoch': self.steps_per_epoch,
            'base_learning_rate': self.base_learning_rate,
        }


class ExpectedCalibrationError(tf.keras.metrics.Metric):
    """
    To implement
    """
    def __init__(self, num_bins=15, name='ece', **kwargs):
        super(ExpectedCalibrationError, self).__init__(name=name, **kwargs)

    def update_state(self, labels, probabilities):
        return

    def result(self):
        return

    def reset_states(self):
        return




(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
shape = (x_train.shape[1:])
classes = tf.unique(tf.reshape(y_train, shape=(-1,)))[0].get_shape().as_list()[0]
train_dataset_size = x_train.shape[0]
# WRN params
n, k = 28, 10
# Number of subnetworks
M = 3
model = WRN.build_model(shape, classes, n, k, M)
print(model.summary())

lr_decay_ratio = 0.1
base_lr=0.1
lr_warmup_epochs=1
lr_decay_epochs = [80, 160, 180]


BATCH_SIZE = 512
epochs = 250
batch_repetition = 4
l2_reg = 3e-4


steps_per_epoch = train_dataset_size // BATCH_SIZE
lr_schedule = WarmUpPiecewiseConstantSchedule(
    steps_per_epoch,
    base_lr,
    decay_ratio=lr_decay_ratio,
    decay_epochs=lr_decay_epochs,
    warmup_epochs=lr_warmup_epochs)
optimizer = tf.keras.optimizers.SGD(
    lr_schedule, momentum=0.9, nesterov=True)

metrics = {
    'train/negative_log_likelihood': tf.keras.metrics.Mean(),
    'train/accuracy': tf.keras.metrics.SparseCategoricalAccuracy(),
    'train/loss': tf.keras.metrics.Mean(),
    'train/ece': ExpectedCalibrationError(),
    'test/negative_log_likelihood': tf.keras.metrics.Mean(),
    'test/accuracy': tf.keras.metrics.SparseCategoricalAccuracy(),
    'test/ece': ExpectedCalibrationError(),
}


