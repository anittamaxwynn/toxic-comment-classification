from typing import TypeAlias

import keras
import tensorflow as tf

Dataset: TypeAlias = tf.data.Dataset
Vectorizer: TypeAlias = keras.layers.TextVectorization

Model: TypeAlias = keras.models.Model
Sequential: TypeAlias = keras.models.Sequential
Metric: TypeAlias = keras.metrics.Metric
Callback: TypeAlias = keras.callbacks.Callback
History: TypeAlias = keras.callbacks.History
