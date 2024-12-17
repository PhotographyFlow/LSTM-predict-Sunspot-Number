import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow_datasets as tfds
import tensorflow as tf


#load dataset
(ds_train,ds_test),ds_info=tfds.load(
  'mnist',  #use MNIST dataset from tensorflow_datasets
  split=['train','test'],
  shuffle_files=True, #shuffle them when training
  as_supervised=True, #returns a tuple (img, label) instead of a dictionary {'image': img, 'label': label}
  with_info=True,
)


#build training pipeline
def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.map(lambda x, y: (x, tf.one_hot(y, 10)))
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)


#build evaluation pipeline
ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.map(lambda x, y: (x, tf.one_hot(y, 10)))
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)


#create model
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001,epsilon=1e-08,use_ema=True),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=[tf.keras.metrics.F1Score()],
)

model.fit(
    ds_train,
    epochs=30,
    validation_data=ds_test,
)
     

#Save the weight for reproduction
model.save_weights('./weights.weights.h5')
#Save model
model.save('./LSTM_model.keras')