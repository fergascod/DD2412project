import functools
import tensorflow as tf
import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Dense, Dropout, Add, Input, Activation
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten

from MultioutDense import *

BATCHNORM_L2 = 3e-4

BatchNormalization = functools.partial(  # pylint: disable=invalid-name
    tf.keras.layers.BatchNormalization,
    epsilon=1e-5,  # using epsilon and momentum defaults from original paper
    momentum=0.9,
    beta_regularizer=tf.keras.regularizers.l2(BATCHNORM_L2),
    gamma_regularizer=tf.keras.regularizers.l2(BATCHNORM_L2)\
)

l1_l2 = tf.keras.regularizers.l1_l2


def main_block(x, filters, n, strides, dropout, l1=0., l2=0.):
    # Normal part
    x_res = Conv2D(filters, (3, 3), strides=strides, padding="same",kernel_regularizer=l1_l2(l1=l1, l2=l2))(x)  # , kernel_regularizer=l2(5e-4)
    x_res = BatchNormalization()(x_res)
    x_res = Activation('relu')(x_res)
    x_res = Conv2D(filters, (3, 3), padding="same",kernel_regularizer=l1_l2(l1=l1, l2=l2))(x_res)
    # Alternative branch
    x = Conv2D(filters, (1, 1), strides=strides,kernel_regularizer=l1_l2(l1=l1, l2=l2))(x)
    # Merge Branches
    x = Add()([x_res, x])

    for i in range(n - 1):
        # Residual connection
        x_res = BatchNormalization()(x)
        x_res = Activation('relu')(x_res)
        x_res = Conv2D(filters, (3, 3), padding="same",kernel_regularizer=l1_l2(l1=l1, l2=l2))(x_res)
        # Apply dropout if given
        if dropout: x_res = Dropout(dropout)(x)
        # Second part
        x_res = BatchNormalization()(x_res)
        x_res = Activation('relu')(x_res)
        x_res = Conv2D(filters, (3, 3), padding="same",kernel_regularizer=l1_l2(l1=l1, l2=l2))(x_res)
        # Merge branches
        x = Add()([x, x_res])

    # Inter block part
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def build_model(input_dims, output_dim, n, k, M, dropout=None, l1=0., l2=0.):
    """ Builds the model. Params:
            - n: number of layers. WRNs are of the form WRN-N-K
                 It must satisfy that (N-4)%6 = 0
            - k: Widening factor. WRNs are of the form WRN-N-K
                 It must satisfy that K%2 = 0
            - input_dims: input dimensions for the model
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

    scaled_l2 = l2 * M
    scaled_l1 = l1 * M
    kernel_regularizer = l1_l2(l1=scaled_l1, l2=scaled_l2)
    # Head of the model
    x = Conv2D(16, (3, 3), padding="same",kernel_regularizer=kernel_regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # 3 Blocks (normal-residual)
    x = main_block(x, 16 * k, n, (1, 1), dropout, l2=l2, l1=l1)  # 0
    x = main_block(x, 32 * k, n, (2, 2), dropout, l2=l2, l1=l1)  # 1
    x = main_block(x, 64 * k, n, (2, 2), dropout, l2=l2, l1=l1)  # 2

    # Final part of the model
    x = AveragePooling2D((8, 8))(x)
    x = Flatten()(x)
    # noinspection PyCallingNonCallable
    outputs = MultioutDense(output_dim,M=M,
                            kernel_initializer='he_normal',
                            activation=None,
                            kernel_regularizer=l1_l2(l1=scaled_l1, l2=scaled_l2),
                            bias_regularizer=l1_l2(l1=scaled_l1, l2=scaled_l2)
                            )(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model
