import WRN_alternative
import json
import os
import sys
import tensorflow as tf
from utils import *
import os
import pickle
import time
from absl import logging
import matplotlib.pyplot as plt

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
AUTO = tf.data.AUTOTUNE
RUN_ID = '0002'
SECTION = 'Cifar10'
PARENT_FOLDER = os.getcwd()
physical_devices = tf.config.list_physical_devices('GPU')
logging.info("Num GPUs:", len(physical_devices))
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)



def main():
    RUN_FOLDER = 'run/distributed{}/'.format(SECTION)
    RUN_FOLDER += '_'.join(RUN_ID)
    if not os.path.exists(RUN_FOLDER):
        os.makedirs(RUN_FOLDER)
        os.mkdir(os.path.join(RUN_FOLDER, 'weights'))
        os.mkdir(os.path.join(RUN_FOLDER, 'metrics'))


    batch_repetitions = 1
    global_batch_size = 256
    strategy = tf.distribute.MirroredStrategy()
    n_cores = 2
    per_core_batch_size = int(global_batch_size / n_cores)
    train_batch_size = int(per_core_batch_size*n_cores / batch_repetitions)
    test_batch_size= int(per_core_batch_size*n_cores)

    # Number of subnetworks (baseline=3)
    M = 3
    tr_data, test_data, classes, train_dataset_size, test_dataset_size, input_shape = load_dataset('cifar10',train_batch_size,test_batch_size)
    tr_data, test_data= create_M_structure(tr_data, test_data, M, batch_repetitions, train_batch_size, test_batch_size)
    input_shape= [M]+input_shape
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

        model = WRN_alternative.build_model(input_dims=input_shape,
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


    def train_step(next_batch):
        """Training step function."""

        def step_fn(inputs):
            """Per-Replica step function."""
            images = inputs[0]
            labels= inputs[1]
            labels= tf.one_hot(labels, classes)
            pre_shuffle_im= tf.reshape(images,(-1,images.shape[2],images.shape[3],images.shape[4]))
            pre_shuffle_lab= tf.reshape(labels,(-1,labels.shape[2]))
            main_shuffle = tf.random.shuffle(tf.range(global_batch_size*M))
            shuffled_im = tf.gather(pre_shuffle_im,main_shuffle,axis=0)
            shufffled_lab = tf.gather(pre_shuffle_lab,main_shuffle,axis=0)
            images= tf.reshape(shuffled_im,(images.shape))
            labels = tf.reshape(shufffled_lab,(labels.shape))
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
                #  FUNDAMENTAL TO SCALE THE LOSS
                scaled_loss = tf.nn.scale_regularization_loss(loss)

            grads = tape.gradient(scaled_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            probabilities = tf.nn.softmax(tf.reshape(logits, [-1, classes]))
            flat_labels = tf.reshape(labels, [-1, classes])
            training_metrics['train/ece'].update_state(tf.argmax(flat_labels, axis=-1), probabilities)
            training_metrics['train/loss'].update_state(loss)
            training_metrics['train/negative_log_likelihood'].update_state(negative_log_likelihood)
            training_metrics['train/accuracy'].update_state(flat_labels, probabilities)


        try:
            strategy.run(step_fn, args=(next_batch,))
        except (StopIteration, tf.errors.OutOfRangeError):
            print("end of dataset")
            return
            # if StopIteration is raised, break from loop


    def test_step(next_batch):
        """Evaluation StepFn."""

        def step_fn(inputs):
            """Per-Replica StepFn."""
            images = inputs['image']
            labels= inputs['label']
            labels = tf.squeeze(tf.one_hot(labels, classes))
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
            test_metrics['test/accuracy'].update_state(labels, probabilities)
        try:
            strategy.run(step_fn, args=(next_batch,))
        except (StopIteration, tf.errors.OutOfRangeError):
            print("end of dataset")
            return
            # if StopIteration is raised, break from loop

    train_iterator = iter(tr_data)
    for epoch in range(initial_epoch, EPOCHS):
        logging.info("Epoch: {}".format(epoch))
        t1 = time.time()
        for step in range(steps_per_epoch):
            batch = next(train_iterator)
            train_step(batch)
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
            test_batch = next(test_iterator)
            test_step(test_batch)
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
    metric = "negative_log_likelihood"
    metric_evo_train = []
    metric_evo_test = []
    with (open(os.path.join(RUN_FOLDER, 'metrics/metrics_evo.pickle'), "rb")) as f:
            metrics_train, metrics_test = pickle.load(f)

    epochs = [i for i in range(len(metrics_train))]

    for metric_train, metric_test in zip(metrics_train, metrics_test):
        metric_evo_train.append(metric_train["train/"+metric])
        metric_evo_test.append(metric_test["test/"+metric])

    plt.plot(epochs, metric_evo_train,label='training')
    plt.plot(epochs, metric_evo_test,label='testing')
    plt.legend()
    plt.title("Evolution of "+metric+" during training")
    plt.savefig(os.path.join(RUN_FOLDER,'nll-evolution'))


if __name__ == '__main__':
    main()

