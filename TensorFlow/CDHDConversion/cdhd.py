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
tf.app.flags.DEFINE_boolean('sigma', 15, """RBF sigma""")

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

def inference(images, meta):
  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 3, 32],
                                         stddev=1,  #check if this is right
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

    doForwardPass(conv6, meta['out_locs'])
    
def doForwardPass(x, out_locs):
  grid_x = np.arange(-FLAGS.grid_size, FLAGS.grid_size + 1, FLAGS.grid_stride)
  grid_y = np.arange(-FLAGS.grid_size, FLAGS.grid_size + 1, FLAGS.grid_stride)

  offset_grid_list = []
  for xi in xrange(grid_x.shape[0]):
    for yi in xrange(grid_y.shape[0]):
      offset_grid_list.append((grid_x[xi], grid_y[yi]))

  offset_grid = np.asarray(offset_grid_list) 
  offset_grid = tf.convert_to_tensor(offset_grid)
  offset_grid_shape = offset_grid.get_shape().as_list()
  offset_grid = tf.reshape(offset_grid, [1, offset_grid_shape[1],offset_grid_shape[0]])

  num_out_filters = offset_grid.get_shape().as_list()[2] + 1;

  x_shape = x.get_shape().as_list()
  n = x_shape[1] * x_shape[2]
  xa = tf.cast(tf.divide(tf.ones([x_shape[0], x_shape[1],x_shape[2],1], tf.int32), n), tf.float32)

  x = tf.concat(3, [xa,x])    #IN NEWER VERSION OF TF CORRECT COMMAND IS: tf.concat([xa,x], 3)
  x_shape = x.get_shape().as_list()

  res_steps = np.zeros((1,FLAGS.steps))
  all_preds = np.zeros((FLAGS.steps, 2, 1, x_shape[0]))
  all_cents = np.zeros((FLAGS.steps, 2, 1, x_shape[0]))

  fwd_dict = {}
  fwd_dict['num_out_filters'] = num_out_filters
  fwd_dict['out_locs'] = out_locs
  fwd_dict['offset_grid'] = offset_grid

  assert(out_locs.shape[0] == x.get_shape().as_list()[1] * x.get_shape().as_list()[2]), \
              "assertion error in forward pass"

  aug_x = [x, None, None, None, 0]

  for i in xrange(FLAGS.steps):
    columnActivation(aug_x, i, fwd_dict)  
    
def columnActivation(aug_x, column_num, fwd_dict):
  prev_pred = aug_x[1]
  prev_loss = aug_x[4]
  prev_offsets = aug_x[2]
  prev_nw = aug_x[3]
  x = aug_x[0]

  chained = True
  if(prev_pred == None):
    chained = False

  num_out_filters = fwd_dict['num_out_filters']
  out_locs = fwd_dict['out_locs']

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
    print('a shape: ', a.get_shape().as_list())      

    w = a[:, :, :, 0:1]
    print('w shape: ', w.get_shape().as_list())

    nw = getNormalizedLocationWeightsFast(w)    
    nw_shape = nw.get_shape().as_list()
    print('nw shape: ', nw_shape)
    nw_reshape = tf.reshape(nw, [nw_shape[0], nw_shape[1] * nw_shape[2]])

    out_locs_rs = tf.convert_to_tensor(out_locs)
    out_locs_rs = tf.cast(out_locs_rs, tf.float32)

    #Predict the centroid.
    pc = tf.multiply(nw_reshape[:,:,None], out_locs_rs[None,:,:])
    print('out_loc_rs shape: ', out_locs_rs.get_shape().as_list())
    print('nw shape: ', nw_reshape.get_shape().as_list())
    print('pc shape 1: ', pc.get_shape().as_list())
    pc_shape = pc.get_shape().as_list()
    pc = tf.reshape(pc, [pc_shape[0], pc_shape[1], pc_shape[2], 1])
    pc = tf.reduce_sum(pc, axis=1)
    pc_shape = pc.get_shape().as_list()
    pc = tf.reshape(pc, [pc_shape[0], 1, pc_shape[1], pc_shape[2]])
    print('pc shape: ', pc.get_shape().as_list())

    #Predict the offset.
    #Use the offset grid to compute the offsetd.

    offset_grid = fwd_dict['offset_grid']
    num_offset_channels = offset_grid.get_shape().as_list()[2]
    offset_channels = tf.convert_to_tensor(np.arange(FLAGS.grid_stride) + 1)
    num_chans = FLAGS.grid_stride + 1

    offset_wts = a[:, :, :, 1: (FLAGS.grid_stride + 1)]
    offset_max = tf.reduce_max(offset_wts, axis=3)

    offset_wts = tf.subtract(offset_wts, offset_max[:,:,:,None])
    offset_wts = tf.exp(offset_wts)
    sum_offset_wts = tf.reduce_sum(offset_wts, axis=3)
    offset_wts = tf.divide(offset_wts, sum_offset_wts[:,:,:,None])
    offset_grid = tf.reshape(offset_grid, [2, 1, 1, FLAGS.grid_stride])
    
    of_x = tf.multiply(tf.cast(offset_grid[0,:,:,:], tf.float32), offset_wts)
    of_y = tf.multiply(tf.cast(offset_grid[1,:,:,:], tf.float32), offset_wts)

    of_x = tf.reduce_sum(of_x,axis=3)
    of_y = tf.reduce_sum(of_y,axis=3)

    po = tf.stack([of_x, of_y])
    po_shape = po.get_shape().as_list()
    po = tf.reshape(po, [po_shape[1], po_shape[2], po_shape[3], po_shape[0]])

    poc = tf.reduce_sum(tf.multiply(po,nw), axis=(1,2))
    poc_shape = poc.get_shape().as_list()
    poc = tf.reshape(poc, [poc_shape[0], 1, poc_shape[1], 1])

    feat_size = a.get_shape().as_list()
    feat_size[3] = 1
    sigma = tf.cast(FLAGS.sigma, tf.float32)
    offset_gauss = doOffset2GaussianForward(pc + poc, out_locs_rs, sigma, feat_size)
    print('gauss: ', offset_gauss.get_shape().as_list())

    if chained:
      computePredictionLossSL1(prev_pred, pc, FLAGS.transition_dist)
    else:
      cent_residue = pc * 0;
      cent_loss = 0;
      offs_residue = 0;
      offs_loss = 0;      

def getNormalizedLocationWeightsFast(w):
  #Softmax
  a = tf.reduce_max(w, axis=(1,2))
  ew = tf.exp(tf.subtract(w, a[:,None,None,:]))
  sew = tf.reduce_sum(ew, axis=(1,2))
  nw = tf.divide(ew,sew[:,None,None,:])
  return nw

def doOffset2GaussianForward(offset, locs, sigma, feat_size):
  #based on: https://en.wikipedia.org/wiki/Radial_basis_function_kernel
  feat_denom = tf.reduce_sum(tf.square(tf.subtract(offset, locs[None,:,:,None])), axis=2)
  feat = tf.divide((feat_denom/2), tf.square(sigma))
  feat_shape = feat.get_shape().as_list()
  feat = tf.reshape(feat, [feat_shape[0], feat_shape[1], feat_shape[2], 1]) 
  feat = tf.exp(-tf.reshape(feat, feat_size))
  return feat

def computePredictionLossSL1(pred, target, transition_dist):
  residue = tf.subtract(pred, target)
  dim_losses = tf.abs(residue)

  comparator_lt = tf.less(dim_losses, tf.constant(transition_dist)) 
  lt_tensor = dim_losses.assign(tf.where 
                                    (comparator_lt, tf.zeros_like(dim_losses), 
                                      tf.divide(tf.square(dim_losses),2)
                                    )
                                  )

  comparator_gte = tf.greater_equal(dim_losses, tf.constant(transition_dist)) 
  gte_tensor = dim_losses.assign(tf.where 
                                    (comparator_gte, tf.zeros_like(dim_losses), 
                                      tf.divide(tf.subtract(dim_losses, transition_dist),2)
                                    )
                                  )  

  loss = tf.concat(0, [lt_tensor, gte_tensor])
  return loss, residue


