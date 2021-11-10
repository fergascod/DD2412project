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
    Expected calibration error (Guo et al., 2017, Naeini et al., 2015)
    """

    def __init__(self, num_bins=15, name='ece', **kwargs):
        super(ExpectedCalibrationError, self).__init__(name=name, **kwargs)
        self.num_bins = num_bins
        self.correct_sums = self.add_weight(
            "correct_sums", shape=(num_bins,), initializer='zeros')
        self.prob_sums = self.add_weight(
            "prob_sums", shape=(num_bins,), initializer='zeros')
        self.counts = self.add_weight(
            "counts", shape=(num_bins,), initializer='zeros')

    def update_state(self, labels, probabilities):
        pred_labels = tf.math.argmax(probabilities, axis=-1)
        pred_probs = tf.math.reduce_max(probabilities, axis=-1)
        correct_preds = tf.math.equal(pred_labels,tf.cast(labels, pred_labels.dtype))
        correct_preds = tf.cast(correct_preds, float)
        custom_binning_score = pred_probs
        bin_indices = tf.histogram_fixed_width_bins(
            custom_binning_score,tf.constant([0., 1.], float), nbins=self.num_bins)
        batch_correct_sums = tf.math.unsorted_segment_sum(
            data=correct_preds,
            segment_ids=bin_indices,
            num_segments=self.num_bins)
        batch_prob_sums = tf.math.unsorted_segment_sum(data=pred_probs,
                                                       segment_ids=bin_indices,
                                                       num_segments=self.num_bins)
        batch_counts = tf.math.unsorted_segment_sum(data=tf.ones_like(bin_indices),
                                                    segment_ids=bin_indices,
                                                    num_segments=self.num_bins)
        batch_counts = tf.cast(batch_counts, float)
        self.correct_sums.assign_add(batch_correct_sums)
        self.prob_sums.assign_add(batch_prob_sums)
        self.counts.assign_add(batch_counts)
        return

    def result(self):
        non_empty = tf.math.not_equal(self.counts, 0)
        correct_sums = tf.boolean_mask(self.correct_sums, non_empty)
        prob_sums = tf.boolean_mask(self.prob_sums, non_empty)
        counts = tf.boolean_mask(self.counts, non_empty)
        accs = correct_sums / counts
        confs = prob_sums / counts
        total_count = tf.reduce_sum(counts)
        return tf.reduce_sum(counts / total_count * tf.abs(accs - confs))

    def reset_states(self):
        tf.keras.backend.batch_set_value([(v, [0., ] * self.num_bins) for v in
                                          self.variables])





AUTO = tf.data.AUTOTUNE
BATCH_SIZE = 512


def load_CIFAR_10(M):

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    batch_repetition=1
    main_shuffle = tf.random.shuffle(tf.tile(tf.range(BATCH_SIZE), [batch_repetition]))
    to_shuffle = tf.shape(main_shuffle)[0]
    shuffle_indices = [
        tf.concat([tf.random.shuffle(main_shuffle[:to_shuffle]),
                   main_shuffle[to_shuffle:]], axis=0)
        for _ in range(M)]

    training_data = (tf.data.Dataset.from_tensor_slices((x_train, y_train))
                     .batch(BATCH_SIZE*M).prefetch(AUTO)
                     .map(lambda x,y:(tf.stack([tf.gather(x, indices, axis=0)
                                                for indices in shuffle_indices], axis=1),
                                      tf.stack([tf.gather(y, indices, axis=0)
                                                for indices in shuffle_indices], axis=1)),
                          num_parallel_calls=AUTO, ))

    test_data = (tf.data.Dataset.from_tensor_slices((x_test, y_test))
                 .shuffle(BATCH_SIZE * 100000)
                 .batch(BATCH_SIZE)
                 .prefetch(AUTO))
    classes = tf.unique(tf.reshape(y_train, shape=(-1,)))[0].get_shape().as_list()[0]
    training_size = x_train.shape[0]
    input_dim = training_data.element_spec[0].shape[1:]
    return training_data, test_data, classes,training_size,input_dim




# Number of subnetworks
M = 3
tr_data, test_data, classes, train_dataset_size,input_shape= load_CIFAR_10(M)
# WRN params
n, k = 28, 10


lr_decay_ratio = 0.1
base_lr = 0.1
lr_warmup_epochs = 1
lr_decay_epochs = [80, 160, 180]

EPOCHS = 250
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
    'train/accuracy': tf.keras.metrics.CategoricalAccuracy(),
    'train/loss': tf.keras.metrics.Mean(),
    'train/ece': ExpectedCalibrationError(),
    'test/negative_log_likelihood': tf.keras.metrics.Mean(),
    'test/accuracy': tf.keras.metrics.CategoricalAccuracy(),
    'test/ece': ExpectedCalibrationError(),
}

model = WRN.build_model(input_shape, classes, n, k, M)
print(model.summary())
for epoch in range(0, EPOCHS):
    iteratorX = iter(tr_data)
    while True:
        try:
            # get the next batch
            batchX = next(iteratorX)
            images = batchX[0]
            labels = tf.squeeze(tf.one_hot(batchX[1], 10))
            with tf.GradientTape() as tape:
                logits = model(images, training=True)
                negative_log_likelihood = tf.reduce_mean(tf.reduce_sum(
                    tf.keras.losses.categorical_crossentropy(
                        labels, logits, from_logits=True), axis=1))
                filtered_variables = []
                # tv= model.trainable_variables
                for var in model.trainable_variables:
                    if ('kernel' in var.name or 'batch_norm' in var.name or
                            'bias' in var.name):
                        filtered_variables.append(tf.reshape(var, (-1,)))
                l2_loss = l2_reg * 2 * tf.nn.l2_loss(tf.concat(filtered_variables, axis=0))
                # tf.nn returns l2 loss divided by 0.5 so we need to double it
                loss = l2_loss + negative_log_likelihood
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            probabilities = tf.nn.softmax(tf.reshape(logits, [-1, classes]))
            metrics['train/ece'].update_state(tf.argmax(tf.reshape(labels, [-1,classes]), axis=-1)
                                              , probabilities)
            metrics['train/loss'].update_state(loss)
            metrics['train/negative_log_likelihood'].update_state(negative_log_likelihood)
            metrics['train/accuracy'].update_state(tf.reshape(labels, [-1]), probabilities)

        except StopIteration:
            # if StopIteration is raised, break from loop
            # print(loss)
            break
