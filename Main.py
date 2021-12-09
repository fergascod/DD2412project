import WRN
import WRN_alternative
import tensorflow as tf
from utils import *
import os
import pickle
import time
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

global_batch_size = 256  # 512
# Number of subnetworks (baseline=3)
M = 1
batch_repetitions = 1
l2_reg = 3e-4

RUN_ID = '0003'
SECTION = 'Cifar10'
PARENT_FOLDER = os.getcwd()
RUN_FOLDER = 'run/{}/'.format(SECTION)
RUN_FOLDER += '_'.join(RUN_ID)
if not os.path.exists(RUN_FOLDER):
    os.makedirs(RUN_FOLDER)
    os.mkdir(os.path.join(RUN_FOLDER, 'weights'))
    os.mkdir(os.path.join(RUN_FOLDER, 'metrics'))

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)


def train(tr_dataset, model, optimizer, metrics, num_labels):
    iteratorX = iter(tr_dataset)
    while True:
        try:
            # get the next batch
            batchX = next(iteratorX)
            images = batchX[0]
            labels= batchX[1]
            labels= tf.one_hot(labels, num_labels)
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

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            probabilities = tf.nn.softmax(tf.reshape(logits, [-1, num_labels]))
            flat_labels = tf.reshape(labels, [-1, num_labels])
            metrics['train/ece'].update_state(tf.argmax(flat_labels,axis=-1), probabilities)
            metrics['train/loss'].update_state(loss)
            metrics['train/negative_log_likelihood'].update_state(negative_log_likelihood)
            metrics['train/accuracy'].update_state(flat_labels, probabilities)

        except (StopIteration, tf.errors.OutOfRangeError):
            # if StopIteration is raised, break from loop
            # print("end of dataset")
            break



def main():

    train_batch_size = int(global_batch_size / batch_repetitions)
    test_batch_size = int(global_batch_size)

    # loading function parameter: 'cifar10','cifar100','imagenet', 'speech_commands' (for now)
    tr_data, test_data, num_labels, train_dataset_size, test_dataset_size, input_shape = load_dataset('cifar10', train_batch_size, test_batch_size)

    tr_data, test_data= create_M_structure(tr_data, test_data, M, batch_repetitions, train_batch_size, test_batch_size)
    input_shape= [M]+input_shape
    # WRN params
    n, k = 28, 10

    lr_decay_ratio = 0.2
    base_lr = 0.1 * train_batch_size / 128
    lr_warmup_epochs = 1
    EPOCHS = 250
    decay_epochs = [80, 160, 180]
    lr_decay_epochs = [(int(start_epoch_str) * EPOCHS) // 200 for start_epoch_str in decay_epochs]

    steps_per_epoch = train_dataset_size // global_batch_size
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
                                        output_dim=num_labels,
                                        n=n,
                                        k=k,
                                        M=M)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(RUN_FOLDER, 'metrics/logs')
                                                          , update_freq='epoch')
    tensorboard_callback.set_model(model)
    print(model.summary())
    train_metrics_evolution = []
    test_metrics_evolution = []

    for epoch in range(0, EPOCHS):
        print("Epoch: {}".format(epoch))
        t1 = time.time()
        train(tr_data, model, optimizer, training_metrics,num_labels)
        t2 = time.time()
        if (epoch + 1) % 50 == 0:
            model.save_weights(os.path.join(RUN_FOLDER, 'weights/weights_%d.h5' % epoch))
        train_metric = {}
        for name, metric in training_metrics.items():
            train_metric[name] = metric.result().numpy()
            print("{} : {}".format(name, metric.result().numpy()))
            metric.reset_states()
        train_metrics_evolution.append(train_metric)
        t3 = time.time()
        compute_test_metrics(model, test_data, test_metrics, M, num_labels)
        t4 = time.time()
        test_metric = {}
        for name, metric in test_metrics.items():
            test_metric[name] = metric.result().numpy()
            print("{} : {}".format(name, metric.result().numpy()))
            metric.reset_states()
        test_metrics_evolution.append(test_metric)
        print(f"Epoch took {t4 - t1}s. Training took {t2 - t1}s and testing {t4 - t3}s\n")

    model.save_weights(os.path.join(RUN_FOLDER, 'weights/final_weights.h5'))
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

    plt.plot(epochs, metric_evo_train)
    plt.plot(epochs, metric_evo_test)
    plt.title("Evolution of "+metric+" during training")
    plt.show()


if __name__ == '__main__':
    main()

