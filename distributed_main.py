import WRN
import json
import os
import sys
import tensorflow as tf
from utils import *
import os
import pickle
import time
from absl import logging
import tensorflow_datasets as tfds

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
AUTO = tf.data.AUTOTUNE
RUN_ID = '0002'
SECTION = 'Cifar10'
PARENT_FOLDER = os.getcwd()
physical_devices = tf.config.list_physical_devices('GPU')
logging.info("Num GPUs:", len(physical_devices))
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)


def load_CIFAR_10(tr_batch_size, test_batch_size):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    # training on a few examples because it's too slow otherwise, you can remove the [] to train on the full dataset
    """training_data = (tf.data.Dataset.from_tensor_slices((x_train[:], y_train[:]))
                     .batch(tr_batch_size, drop_remainder=True).prefetch(AUTO)
                     .shuffle(tr_batch_size * 100000).repeat())

    test_data = (tf.data.Dataset.from_tensor_slices((x_test[:], y_test[:]))
                 .batch(test_batch_size, drop_remainder=True).prefetch(AUTO)
                 .shuffle(test_batch_size * 100000).repeat())"""
    ds_info = tfds.builder('cifar10').info
    training_data=tfds.load('cifar10', split='train', shuffle_files=True, batch_size=tr_batch_size)
    test_data = tfds.load('cifar10', split='test', shuffle_files=True, batch_size=test_batch_size)
    classes = tf.unique(tf.reshape(y_train, shape=(-1,)))[0].get_shape().as_list()[0]
    training_size = ds_info.splits['train'].num_examples
    test_size = ds_info.splits['test'].num_examples
    input_dim = list(ds_info.features['image'].shape)
    return training_data, test_data, classes, training_size, test_size, input_dim


def main():
    RUN_FOLDER = 'run/{}/'.format(SECTION)
    RUN_FOLDER += '_'.join(RUN_ID)
    if not os.path.exists(RUN_FOLDER):
        os.makedirs(RUN_FOLDER)
        os.mkdir(os.path.join(RUN_FOLDER, 'weights'))
        os.mkdir(os.path.join(RUN_FOLDER, 'metrics'))


    batch_repetitions = 4
    BATCH_SIZE = 256
    strategy = tf.distribute.MirroredStrategy()
    n_cores = 2
    per_core_batch_size = int(BATCH_SIZE / n_cores)
    train_batch_size = int(per_core_batch_size*n_cores / batch_repetitions)
    test_batch_size= int(per_core_batch_size*n_cores)

    # Number of subnetworks (baseline=3)
    M = 3
    tr_data, test_data, classes,\
    train_dataset_size, test_dataset_size, input_shape = load_CIFAR_10(train_batch_size,test_batch_size)
    tr_data = strategy.experimental_distribute_dataset(tr_data)
    test_data = strategy.experimental_distribute_dataset(test_data)
    # WRN params
    n, k = 28, 10

    lr_decay_ratio = 0.2
    base_lr = 0.1 * train_batch_size // 128
    lr_warmup_epochs = 1
    lr_decay_epochs = [80, 160, 180]
    EPOCHS = 250
    l2_reg = 3e-4
    steps_per_epoch = train_dataset_size // train_batch_size
    steps_per_eval = test_dataset_size // test_batch_size

    with strategy.scope():
        lr_schedule = WarmUpPiecewiseConstantSchedule(
            steps_per_epoch,
            base_lr,
            decay_ratio=lr_decay_ratio,
            decay_epochs=lr_decay_epochs,
            warmup_epochs=lr_warmup_epochs)
        optimizer = tf.keras.optimizers.SGD(
            lr_schedule, momentum=0.9, nesterov=True)

        training_metrics = {
            'train/negative_log_likelihood': tf.keras.metrics.Mean(),
            'train/accuracy': tf.keras.metrics.CategoricalAccuracy(),
            'train/loss': tf.keras.metrics.Mean(),
            'train/ece': ExpectedCalibrationError(),
        }

        test_metrics = {
            'test/negative_log_likelihood': tf.keras.metrics.Mean(),
            'test/accuracy': tf.keras.metrics.CategoricalAccuracy(),
            'test/ece': ExpectedCalibrationError(),
        }

        model = WRN.build_model(input_dims=[M] +input_shape,
                                output_dim=classes,
                                n=n,
                                k=k,
                                M=M)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(RUN_FOLDER, 'metrics/logs')
                                                              , update_freq='epoch')
        tensorboard_callback.set_model(model)
        print(model.summary())
        train_metrics_evolution = []
        test_metrics_evolution = []
        checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
        latest_checkpoint = tf.train.latest_checkpoint(RUN_FOLDER)
        initial_epoch = 0
        if latest_checkpoint:
            checkpoint.restore(latest_checkpoint)
            logging.info('Loaded checkpoint %s', latest_checkpoint)
            initial_epoch = optimizer.iterations.numpy() // steps_per_epoch

    @tf.function
    def train_step(iterator):
        """Training step function."""

        def step_fn(inputs):
            """Per-Replica step function."""
            a=0
            images = inputs['image']
            labels= inputs['label']
            BATCH_SIZE = tf.shape(images)[0]

            main_shuffle = tf.random.shuffle(tf.tile(
                tf.range(BATCH_SIZE), [batch_repetitions]))
            to_shuffle = tf.cast(tf.cast(tf.shape(main_shuffle)[0], tf.float32),tf.int32)
            shuffle_indices = [
                tf.concat([tf.random.shuffle(main_shuffle[:to_shuffle]),
                           main_shuffle[to_shuffle:]], axis=0)
                for _ in range(M)]
            images = tf.stack([tf.gather(images, indices, axis=0)
                               for indices in shuffle_indices], axis=1)
            labels = tf.stack([tf.gather(labels, indices, axis=0)
                               for indices in shuffle_indices], axis=1)
            labels = tf.one_hot(labels, 10)
            with tf.GradientTape() as tape:
                logits = model(images, training=True)
                negative_log_likelihood = tf.reduce_mean(tf.reduce_sum(
                    tf.keras.losses.categorical_crossentropy(
                        labels, logits, from_logits=True), axis=1))
                filtered_variables = []
                for var in model.trainable_variables:
                    if ('kernel' in var.name or 'batch_norm' in var.name or
                            'bias' in var.name):
                        filtered_variables.append(tf.reshape(var, (-1,)))
                l2_loss = l2_reg * 2 * tf.nn.l2_loss(tf.concat(filtered_variables, axis=0))
                # tf.nn returns l2 loss divided by 0.5 so we need to double it
                loss = l2_loss + negative_log_likelihood
                scaled_loss = loss / strategy.num_replicas_in_sync

            grads = tape.gradient(scaled_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            probabilities = tf.nn.softmax(tf.reshape(logits, [-1, classes]))
            flat_labels = tf.reshape(labels, [-1])
            training_metrics['train/ece'].update_state(tf.argmax(tf.reshape(labels, [-1, classes]), axis=-1)
                                              , probabilities)
            training_metrics['train/loss'].update_state(loss)
            training_metrics['train/negative_log_likelihood'].update_state(negative_log_likelihood)
            training_metrics['train/accuracy'].update_state(flat_labels, probabilities)

        try:
            strategy.run(step_fn, args=(next(iterator),))
        except (StopIteration, tf.errors.OutOfRangeError):
            return
            # if StopIteration is raised, break from loop
                # print("end of dataset")


    @tf.function
    def test_step(iterator):
        """Evaluation StepFn."""

        def step_fn(inputs):
            """Per-Replica StepFn."""
            images = inputs['image']
            labels= inputs['label']
            images = tf.tile(
                tf.expand_dims(images, 1), [1, M, 1, 1, 1])
            labels = tf.one_hot(labels, 10)
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

            test_metrics['test/ece'].update_state(tf.argmax(tf.reshape(labels, [-1, classes]), axis=-1)
                                                  , probabilities)
            # test_ metrics['test/loss'].update_state(loss)
            test_metrics['test/negative_log_likelihood'].update_state(negative_log_likelihood)
            test_metrics['test/accuracy'].update_state(tf.reshape(labels, probabilities.shape), probabilities)
        try:
            strategy.run(step_fn, args=(next(iterator),))
        except (StopIteration, tf.errors.OutOfRangeError):
            return
            # if StopIteration is raised, break from loop
            # print("end of dataset")

    train_iterator = iter(tr_data)
    for epoch in range(initial_epoch, EPOCHS):
        logging.info("Epoch: {}".format(epoch))
        t1 = time.time()
        for step in range(steps_per_epoch):
            train_step(train_iterator)
            current_step = epoch * steps_per_epoch + (step + 1)
            max_steps = steps_per_epoch * EPOCHS
        t2 = time.time()
        if (epoch + 1) % 50 == 0:
            checkpoint.save(
                os.path.join(RUN_FOLDER, 'checkpoint'))
            model.save_weights(os.path.join(RUN_FOLDER, 'weights/weights_%d.h5' % epoch))
        train_metric = {}
        for name, metric in training_metrics.items():
            train_metric[name] = metric.result().numpy()
            print("{} : {}".format(name, metric.result().numpy()))
            metric.reset_states()
        train_metrics_evolution.append(train_metric)

        test_iterator = iter(test_data)
        t3 = time.time()
        for step in range(steps_per_eval):
            test_step(test_iterator)
        t4 = time.time()
        test_metric = {}
        for name, metric in test_metrics.items():
            test_metric[name] = metric.result().numpy()
            print("{} : {}".format(name, metric.result().numpy()))
            metric.reset_states()
        test_metrics_evolution.append(test_metric)
        print(f"Epoch took {t4 - t1}s. Training took {t2 - t1} s and testing {t4 - t3} s\n")

    model.save_weights(os.path.join(RUN_FOLDER, 'weights/final_weights.h5'))
    final_checkpoint_name = checkpoint.save(
    os.path.join(RUN_FOLDER, 'checkpoint'))
    logging.info('Saved last checkpoint to %s', final_checkpoint_name)
    metrics_evo = (train_metrics_evolution, test_metrics_evolution)
    with open(os.path.join(RUN_FOLDER, 'metrics/metrics_evo.pickle'), 'wb') as f:
        pickle.dump(metrics_evo, f)

if __name__ == '__main__':
    main()

