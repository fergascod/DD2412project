import tensorflow as tf

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


