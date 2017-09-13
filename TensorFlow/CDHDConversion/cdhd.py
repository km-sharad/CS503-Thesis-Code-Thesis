"""
Contains CDHD Model
"""

import tensorflow as tf
from cdhd_global import CDHDGlobals
import cdhd_input
import os
import numpy as np

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 192,"""Number of images to process in a batch.""")
tf.app.flags.DEFINE_boolean('use_fp16', False, """Train the model using fp16.""")
tf.app.flags.DEFINE_boolean('steps', 3, """number of columns for steps in paper""")
tf.app.flags.DEFINE_boolean('transition_dist', 1, """transition_dist""")
tf.app.flags.DEFINE_boolean('loc_pred_scale', 1, """loc_pred_scale""")
tf.app.flags.DEFINE_boolean('offset_pred_weight', 0.1, """offset_pred_weight""")
tf.app.flags.DEFINE_boolean('pred_factor', 50, """pred_factor""")
tf.app.flags.DEFINE_boolean('nfc', 128, """number of filter channels""")
tf.app.flags.DEFINE_boolean('grid_size', 50, """grid size""")
tf.app.flags.DEFINE_boolean('grid_stride', 25, """grid stride""")

def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var

def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      tf.random_normal_initializer(stddev=stddev, dtype=dtype))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

#def distorted_inputs(stats_dict,batch_size=FLAGS.batch_size):
def distorted_inputs(stats_dict):
  """Construct distorted input for CDHD training using the Reader ops.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.    
  """

  images, meta = cdhd_input.distorted_inputs(stats_dict, FLAGS.batch_size)

  images = tf.cast(images, tf.float32)
  return images, meta

def inference(images):
  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 3, 32],
                                         stddev=1,	#check if this is right
                                         wd=0.0)
    conv = tf.nn.conv2d(images, kernel, [1, 2, 2, 1], padding='VALID')
    biases = _variable_on_cpu('biases', [32], tf.constant_initializer(1.0))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(pre_activation, name=scope.name)
    
  # check if activation_summary is required
  #check if normalization is required (https://www.tensorflow.org/api_docs/python/tf/nn/local_response_normalization) 

  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 32, 64],
                                         stddev=1,
                                         wd=0.0)
    conv = tf.nn.conv2d(conv1, kernel, [1, 2, 2, 1], padding='VALID')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(1.0))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(pre_activation, name=scope.name)

  # conv3
  with tf.variable_scope('conv3') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 64, 64],
                                         stddev=1,
                                         wd=0.0)
    conv = tf.nn.conv2d(conv2, kernel, [1, 2, 2, 1], padding='VALID')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(1.0))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv3 = tf.nn.relu(pre_activation, name=scope.name)    

  # conv4
  with tf.variable_scope('conv4') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 64, 64],
                                         stddev=1,
                                         wd=0.0)
    conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(1.0))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv4 = tf.nn.relu(pre_activation, name=scope.name)        

  # conv5
  with tf.variable_scope('conv5') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 64, 128],
                                         stddev=1,
                                         wd=0.0)
    conv = tf.nn.conv2d(conv4, kernel, [1, 2, 2, 1], padding='VALID')
    biases = _variable_on_cpu('biases', [128], tf.constant_initializer(1.0))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv5 = tf.nn.relu(pre_activation, name=scope.name)    

  # conv6
  with tf.variable_scope('conv6') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 128, 128],
                                         stddev=1,
                                         wd=0.0)
    conv = tf.nn.conv2d(conv5, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [128], tf.constant_initializer(1.0))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv6 = tf.nn.relu(pre_activation, name=scope.name)  

    doForward(conv6)
    
  
def doForward(x):
  grid_x = np.arange(-FLAGS.grid_size, FLAGS.grid_size + 1, FLAGS.grid_stride)
  grid_y = np.arange(-FLAGS.grid_size, FLAGS.grid_size + 1, FLAGS.grid_stride)

  offset_grid_list = []
  for xi in xrange(grid_x.shape[0]):
    for yi in xrange(grid_y.shape[0]):
      offset_grid_list.append((grid_x[xi], grid_y[yi]))

  offset_grid = np.asarray(offset_grid_list) 
  num_out_filters = offset_grid.shape[0] + 1;

  '''
  gt_loc = layer.org_gt_coords ./ loc_pred_scale;
  shared_layers = layer.shared_layers;
  shared_prefix = layer.shared_prefix;
  '''

  x_shape = x.get_shape().as_list()
  n = x_shape[1] * x_shape[2]
  xa = tf.cast(tf.divide(tf.ones([x_shape[0], x_shape[1],x_shape[2],1], tf.int32), n), tf.float32)
  #xa = tf.cast(xa, tf.float32)

  x = tf.concat(3, [xa,x])    #IN NEWER VERSION CORRECT COMMAND IS: tf.concat([xa,x], 3)
  x_shape = x.get_shape().as_list()
  print('x shape after concat: ', x_shape)

  res_steps = np.zeros((1,FLAGS.steps))
  all_preds = np.zeros((FLAGS.steps, 2, 1, x_shape[0]))
  all_cents = np.zeros((FLAGS.steps, 2, 1, x_shape[0]))

  for i in xrange(FLAGS.steps):
    doForwardPass(x, i, num_out_filters)
    
def doForwardPass(x, i, num_out_filters):
  columnActivation(x, i, num_out_filters)


def columnActivation(x, column_num, num_out_filters):
    with tf.variable_scope('col' + str(column_num) + '1') as scope:
      kernel = _variable_with_weight_decay('weights',
                                           shape=[5, 5, FLAGS.nfc + 1, FLAGS.nfc],
                                           stddev=1,  #check if this is right
                                           wd=0.0)
      conv = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], padding='SAME')
      biases = _variable_on_cpu('biases', [FLAGS.nfc], tf.constant_initializer(1.0))
      a = tf.nn.bias_add(conv, biases)

      a_with_negatives_set_to_zero = tf.nn.relu(a)
      a = tf.multiply(a, a_with_negatives_set_to_zero)

    with tf.variable_scope('col' + str(column_num) + '2') as scope:
      kernel = _variable_with_weight_decay('weights',
                                           shape=[5, 5, FLAGS.nfc, FLAGS.nfc],
                                           stddev=1,  #check if this is right
                                           wd=0.0)
      conv = tf.nn.conv2d(a, kernel, [1, 1, 1, 1], padding='SAME')
      biases = _variable_on_cpu('biases', [FLAGS.nfc], tf.constant_initializer(1.0))
      a = tf.nn.bias_add(conv, biases)

      a_with_negatives_set_to_zero = tf.nn.relu(a)
      a = tf.multiply(a, a_with_negatives_set_to_zero)      

    with tf.variable_scope('col' + str(column_num) + '3') as scope:
      kernel = _variable_with_weight_decay('weights',
                                           shape=[5, 5, FLAGS.nfc, num_out_filters],
                                           stddev=1,  #check if this is right
                                           wd=0.0)
      conv = tf.nn.conv2d(a, kernel, [1, 1, 1, 1], padding='SAME')
      biases = _variable_on_cpu('biases', [num_out_filters], tf.constant_initializer(1.0))
      a = tf.nn.bias_add(conv, biases)

      print('a3 shape: ', a.get_shape().as_list())      
      
      #w = //slice a
      #print('w sp: ', w.get_shape().as_list())

      

