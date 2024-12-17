import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow_datasets as tfds
import tensorflow as tf


#load dataset
(ds_test),ds_info=tfds.load(
  'mnist',  #use MNIST dataset from tensorflow_datasets
  split='test',
  shuffle_files=True, 
  as_supervised=True, #returns a tuple (img, label) instead of a dictionary {'image': img, 'label': label}
  with_info=True,
)


#build evaluation pipeline
def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label


ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.map(lambda x, y: (x, tf.one_hot(y, 10)))
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)


#import model
model =tf.keras.models.load_model('./LSTM_model.keras')

#print result
result = model.evaluate(ds_test)
dict(zip(model.metrics_names,result))
     
