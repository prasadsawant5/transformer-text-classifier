import tensorflow as tf
from tensorflow.keras.optimizers.schedules import LearningRateSchedule


class CustomSchedule(LearningRateSchedule):
    def __init__(self, dModel, warmupSteps=4000):
        super().__init__()

        # define the dmodel and warmup steps
        self.dModel = dModel
        self.dModel = tf.cast(self.dModel, tf.float32)
        self.warmupSteps = warmupSteps

    def __call__(self, step):
        # build the custom schedule logic
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmupSteps**-1.5)

        return tf.math.rsqrt(self.dModel) * tf.math.minimum(arg1, arg2)