import os
import matplotlib.pyplot as plt
import pickle

RUN_ID = '0001'
SECTION = 'Cifar10'
PARENT_FOLDER= os.getcwd()
RUN_FOLDER = 'run/{}/'.format(SECTION)
RUN_FOLDER += '_'.join(RUN_ID)
RUN_FOLDER += "/metrics/metrics_evo.pickle"

metric = "negative_log_likelihood"
metric_evo_train = []
metric_evo_test = []
with (open(RUN_FOLDER, "rb")) as f:
        metrics_train, metrics_test = pickle.load(f)

epochs = [i for i in range(len(metrics_train))]

for metric_train, metric_test in zip(metrics_train, metrics_test):
    metric_evo_train.append(metric_train["train/"+metric])
    metric_evo_test.append(metric_test["test/"+metric])

plt.plot(epochs, metric_evo_train)
plt.plot(epochs, metric_evo_test)
plt.title("Evolution of "+metric+" during training")
plt.show()
