import tensorflow as tf


class TraceCCA(object):
    def __init__(self, tensor_x, tensor_y):
        """
        :param tensor_x: (..., channel_x, sample)
        :param tensor_y: (..., channel_y, sample)
        """
        self.x = tensor_x
        self.y = tensor_y

        self.mean_x = tf.reduce_mean(tensor_x, axis=-1, keepdims=True)
        self.mean_y = tf.reduce_mean(tensor_y, axis=-1, keepdims=True)

        x_ = self.x - self.mean_x
        y_ = self.y - self.mean_y

        s_xx = tf.matmul(x_, tf.matrix_transpose(x_))
        s_yy = tf.matmul(y_, tf.matrix_transpose(y_))
        s_xy = tf.matmul(x_, tf.matrix_transpose(y_))
        s_yx = tf.matmul(y_, tf.matrix_transpose(x_))

        self.M = tf.linalg.inv(s_xx) @ s_xy @ tf.linalg.inv(s_yy) @ s_yx

        self.rho = tf.linalg.trace(self.M)  # \sum\rho^2

