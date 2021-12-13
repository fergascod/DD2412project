import tensorflow as tf
import tensorflow_datasets as tfds


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
        correct_preds = tf.math.equal(pred_labels, tf.cast(labels, pred_labels.dtype))
        correct_preds = tf.cast(correct_preds, float)
        custom_binning_score = pred_probs
        bin_indices = tf.histogram_fixed_width_bins(
            custom_binning_score, tf.constant([0., 1.], float), nbins=self.num_bins)
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


def load_dataset(dataset, tr_batch_size, test_batch_size):
    ds_info = tfds.builder(dataset).info
    training_data = tfds.load(dataset, split='train[20%:]', shuffle_files=True,as_supervised=True, batch_size=-1)
    test_data = tfds.load(dataset, split='test[20%:]', shuffle_files=True,as_supervised=True, batch_size=-1)
    num_labels = ds_info.features['label'].num_classes
    training_size = ds_info.splits['train'].num_examples
    test_size = ds_info.splits['test'].num_examples
    if dataset=='speech_commands':
        input_dim = list(ds_info.features['audio'].shape)
    else:
        input_dim = list(ds_info.features['image'].shape)
    return training_data, test_data, num_labels, training_size, test_size, input_dim




def create_M_structure(tr_set,test_set,M, batch_repetitions,train_batch_size, test_batch_size, audio=False):
    AUTO = tf.data.AUTOTUNE
    main_shuffle = tf.random.shuffle(tf.tile(tf.range(train_batch_size), [batch_repetitions]))
    to_shuffle = tf.shape(main_shuffle)[0]
    shuffle_indices = [
        tf.concat([tf.random.shuffle(main_shuffle[:to_shuffle]),
                   main_shuffle[to_shuffle:]], axis=0)
        for _ in range(M)]

    tr_set= tf.data.Dataset.from_tensor_slices(tr_set).batch(train_batch_size*M,drop_remainder=True).prefetch(AUTO).map(
        lambda x,y:(tf.stack([tf.gather(x, indices, axis=0) for indices in shuffle_indices], axis=1),
                    tf.stack([tf.gather(y, indices, axis=0) for indices in shuffle_indices], axis=1)),
        num_parallel_calls=AUTO, ).shuffle(train_batch_size * 100000)
    if audio:
        test_set = tf.data.Dataset.from_tensor_slices(test_set).batch(test_batch_size, drop_remainder=True).prefetch(
            AUTO).map(
            lambda x, y: (tf.tile(tf.expand_dims(x, 1), [1, M, 1]),
                          y),
            num_parallel_calls=AUTO, ).shuffle(test_batch_size * 100000)
    else:
        test_set = tf.data.Dataset.from_tensor_slices(test_set).batch(test_batch_size,drop_remainder=True).prefetch(AUTO).map(
            lambda x,y:(tf.tile(tf.expand_dims(x, 1), [1, M, 1, 1, 1]),
                        y),
            num_parallel_calls=AUTO, ).shuffle(test_batch_size * 100000)
    return tr_set, test_set


def compute_test_metrics(model, test_data, test_metrics, M, num_labels):
    iteratorX = iter(test_data)
    while True:
        try:
            # get the next batch
            batchX = next(iteratorX)
            images = batchX[0]
            labels = batchX[1]
            labels = tf.squeeze(tf.one_hot(labels, num_labels))
            logits = model(images, training=False)
            logits = tf.squeeze(logits)
            probabilities = tf.nn.softmax(logits)
            if M > 1:
                labels_tiled = tf.tile(
                    tf.expand_dims(labels, 1), [1, M, 1])
                log_likelihoods = -tf.keras.losses.categorical_crossentropy(
                    labels_tiled, logits, from_logits=True)
                negative_log_likelihood = tf.reduce_mean(
                    -tf.reduce_logsumexp(log_likelihoods, axis=[1]) +
                    tf.math.log(float(M)))
                probabilities = tf.math.reduce_mean(probabilities, axis=1)  # marginalize
            else:
                negative_log_likelihood = tf.reduce_mean(
                    tf.keras.losses.categorical_crossentropy(labels, logits, from_logits=True))

            test_metrics['test/ece'].update_state(tf.argmax(tf.reshape(labels, [-1, num_labels]), axis=-1)
                                                  , probabilities)
            # test_ metrics['test/loss'].update_state(loss)
            test_metrics['test/negative_log_likelihood'].update_state(negative_log_likelihood)
            test_metrics['test/accuracy'].update_state(tf.reshape(labels, probabilities.shape), probabilities)
        except StopIteration:
            break
