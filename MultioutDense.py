import tensorflow as tf
import keras

class MultioutDense(keras.layers.Dense):
    """Multiheaded output layer."""

    def __init__(self,
                 units,
                 M,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super().__init__(
            units=units * M,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)
        self.ensemble_size = M

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        # NOTE: This restricts this layer from being called on tensors of ndim > 2.
        outputs = super().call(inputs)
        outputs = tf.reshape(
            outputs,
            [batch_size, self.ensemble_size, self.units // self.ensemble_size])
        return outputs


