# Modified from https://groups.google.com/d/msg/youtube8m-users/yEDzH7EqUf8/EfW0WO3jAgAJ

import tensorflow as tf
from tensorflow.python.platform import gfile


def Dequantize(feat_vector, max_quantized_value=2, min_quantized_value=-2):
  """Dequantize the feature from the byte format to the float format.
  Args:
    feat_vector: the input 1-d vector.
    max_quantized_value: the maximum of the quantized value.
    min_quantized_value: the minimum of the quantized value.
  Returns:
    A float vector which has the same shape as feat_vector.
  """
  assert max_quantized_value >  min_quantized_value
  quantized_range = max_quantized_value - min_quantized_value
  scalar = quantized_range / 255.0
  bias = (quantized_range / 512.0) + min_quantized_value
  return feat_vector * scalar + bias


class YouTube8MFrameFeatureReader:
  def __init__(self,
               num_classes=4816,
               feature_size=1024,
               feature_name="inc3",
               max_frames=300,
               sequence_data=True):
    self.num_classes = num_classes
    self.feature_size = feature_size
    self.feature_name = feature_name
    self.max_frames = max_frames
    self.sequence_data = sequence_data

  def prepare_reader(self,
                     filename_queue,
                     max_quantized_value=2,
                     min_quantized_value=-2):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    context_features, sequence_features = {
        "video_id": tf.FixedLenFeature([], tf.string),
        "labels": tf.VarLenFeature(tf.int64),
    }, None
    if self.sequence_data:
      sequence_features = {
          self.feature_name: tf.FixedLenSequenceFeature([], dtype=tf.string),
      }
    else:
      context_features[self.feature_name] = tf.FixedLenFeature(self.feature_size, tf.float32)

    contexts, features = tf.parse_single_sequence_example(
        serialized_example,
        context_features=context_features,
        sequence_features=sequence_features)

    labels = (tf.cast(
        tf.sparse_to_dense(contexts["labels"].values, (self.num_classes,), 1),
        tf.bool))

    if self.sequence_data:
      decoded_features = tf.reshape(
          tf.cast(
              tf.decode_raw(features[self.feature_name], tf.uint8), tf.float32),
          [-1, self.feature_size])
      num_frames = tf.minimum(tf.shape(decoded_features)[0], self.max_frames)
      video_matrix = Dequantize(decoded_features, max_quantized_value,
                                min_quantized_value)
    else:
      video_matrix = contexts[self.feature_name]
      num_frames = tf.constant(-1)

    # Pad or truncate to 'max_frames' frames.
    video_matrix = tf.cond(tf.shape(video_matrix)[0] <= self.max_frames,
                          lambda: tf.pad(video_matrix,[[0, self.max_frames - tf.shape(video_matrix)[0]], [0, 0]]),
                          lambda: tf.slice(video_matrix, [0,0], [self.max_frames, -1]))
    return contexts["video_id"], video_matrix, labels, num_frames


def prepared_reader(level, files_pattern):
  data_files = gfile.Glob(files_pattern)
  print(data_files)
  
  filename_queue = tf.train.string_input_producer(
      data_files, num_epochs=1, shuffle=False)

  if level == 'frame':
    reader = YouTube8MFrameFeatureReader(feature_name="inc3")
  elif level == 'video':
    reader = YouTube8MFrameFeatureReader(feature_name="mean_inc3", sequence_data=False)
  vals = reader.prepare_reader(filename_queue)
  return vals