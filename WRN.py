import functools

import tensorflow as tf
import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Dense, Dropout, Add, Input, Activation
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten
from keras.regularizers import l2

from MultioutDense import *

BatchNormalization = functools.partial(  # pylint: disable=invalid-name
    tf.keras.layers.BatchNormalization,
    epsilon=1e-5,  # using epsilon and momentum defaults from original paper
    momentum=0.9)

def group(inputs,filters, strides, dropout):
    x_alt=inputs
    x_res=inputs
    x_res = BatchNormalization()(x_res)
    x_res = tf.keras.layers.Activation('relu')(x_res)
    x_res = Conv2D(filters, kernel_size=3, strides=strides, padding="same",
                   use_bias=False,
                   kernel_initializer='he_normal')(x_res)
    # Apply dropout if given
    if dropout:
        x_res = Dropout(dropout)(x_res)
    # Second part
    x_res = BatchNormalization()(x_res)
    x_res = Activation('relu')(x_res)
    x_res = Conv2D(filters, kernel_size=3, padding="same",use_bias=False,
                   kernel_initializer='he_normal')(x_res)
    # Alternative branch
    if not x_alt.shape.is_compatible_with(x_res.shape):
        x_alt = Conv2D(filters, kernel_size=1, strides=strides,use_bias=False,
                       kernel_initializer='he_normal')(x_alt)
    # Merge Branches
    x = Add()([x_res, x_alt])
    return x



def main_block(x, filters, n, strides, dropout):
    x = group(x,filters,strides,dropout=False)
    for i in range(n - 1):
        x = group(x,filters=filters,strides=1,dropout=dropout)
    return x


def build_model(input_dims, output_dim, n, k, M, dropout=False):
    """ Builds the model. Params:
            - n: number of layers. WRNs are of the form WRN-N-K
                 It must satisfy that (N-4)%6 = 0
            - k: Widening factor. WRNs are of the form WRN-N-K
                 It must satisfy that K%2 = 0
            - input_dims: input dimensions for the model.
                The input input_shape must be (ensemble_size, width,height, channels).
            - output_dim: output dimensions for the model
            - dropout: dropout rate - default=0 (not recomended >0.3)
            - act: activation function - default=relu. Build your custom
                   one with keras.backend (ex: swish, e-swish)
    """
    # Ensure n & k are correct
    assert (n - 4) % 6 == 0
    assert k % 2 == 0
    n = (n - 4) // 6
    # This returns a tensor input to the model
    inputs = Input(shape=(input_dims))
    x = tf.keras.layers.Permute([2, 3, 4, 1])(inputs)
    x = tf.keras.layers.Reshape(input_dims[1:-1] +
                                [input_dims[-1] * M])(x)
    # Head of the model
    x = Conv2D(filters=16,
               kernel_size=3,
               strides=1,
               padding="same",
               use_bias=False,
               kernel_initializer='he_normal')(x)
    # 3 Blocks (normal-residual)
    x = main_block(x, filters=16 * k, n=n, strides=1, dropout=dropout)  # 0
    x = main_block(x, filters=32 * k, n=n, strides=2, dropout=dropout)  # 1
    x = main_block(x, filters=64 * k, n=n, strides=2, dropout=dropout)  # 2
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    # Final part of the model
    x = AveragePooling2D(pool_size=8)(x)
    x = Flatten()(x)


    # noinspection PyCallingNonCallable
    outputs = MultioutDense(output_dim,M=M,
                            kernel_initializer='he_normal',
                            activation=None,
                            )(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model
