"""Custom eval metrics for YouTube-8M model."""

import tensorflow as tf
from tensorflow.python.ops import init_ops


class AverageNClass(tf.keras.metrics.Metric):

    def __init__(self, name="average_n_class", **kwargs):
        super(tf.keras.metrics.Metric, self).__init__(name=name, **kwargs)
        self.n_example = self.add_weight(
            "n_example",
            shape=(),
            dtype=tf.float32,
            initializer=init_ops.zeros_initializer)
        self.n_predicted_class = self.add_weight(
            "n_predicted_class",
            shape=(),
            dtype=tf.float32,
            initializer=init_ops.zeros_initializer)

    def update_state(self, y_true, y_pred, **kwargs):
        # Accumulate sample size.
        batch_size = tf.cast(len(y_true), tf.float32)
        self.n_example.assign_add(batch_size)
        # Accumulate number of predicted classes.
        batch_n_class = tf.reduce_sum(tf.cast(y_pred > .5, tf.float32))
        self.n_predicted_class.assign_add(batch_n_class)

    def result(self):
        return self.n_predicted_class / self.n_example


class HitAtOne(tf.keras.metrics.Metric):

    def __init__(self, name="hit_at_one", **kwargs):
        super(tf.keras.metrics.Metric, self).__init__(name=name, **kwargs)
        self.n_example = self.add_weight(
            "n_example",
            shape=(),
            dtype=tf.float32,
            initializer=init_ops.zeros_initializer)
        self.hit_at_one = self.add_weight(
            "hit_at_one",
            shape=(),
            dtype=tf.float32,
            initializer=init_ops.zeros_initializer)

    def update_state(self, y_true, y_pred, **kwargs):
        # Accumulate sample size.
        batch_size = tf.cast(len(y_true), tf.float32)
        self.n_example.assign_add(batch_size)
        # Count number of hit@one.
        tops = tf.math.argmax(y_pred, axis=1, output_type=tf.int32)
        top_idx = tf.stack([tf.range(len(y_true)), tops], axis=1)
        hits = tf.gather_nd(y_true, indices=top_idx)
        self.hit_at_one.assign_add(tf.reduce_sum(hits))

    def result(self):
        return self.hit_at_one / self.n_example
