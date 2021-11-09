import tensorflow as tf
import keras

#import pip

# # # Run of these two for installation
#pip.main(['install','keras_wrn'])  # pip install keras_wrn --user
import keras_wrn


shape, classes = (32, 32, 3), 10
n, k = 28, 2

model = keras_wrn.build_model(shape, classes, n, k)

print(model.summary())
