"""
Contains CDHD Model
"""

import tensorflow as tf
from cdhd_global import CDHDGlobals
import cdhd_input
import os
import numpy as np
import pdb

# Basic model parameters.
steps = 3                           # number of columns for steps in paper
transition_dist = 1
offset_pred_weight = 0.1
pred_factor = 50
nfc = 128                           # number of filter channels
grid_size = 50
grid_stride = 25
prev_pred_weight = 0.1

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
    var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32, trainable=True)
    tf.add_to_collection(name, var)

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
  # dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      tf.random_normal_initializer(stddev=stddev, dtype=tf.float32))
  # if wd is not None:
  #   weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
  #   tf.add_to_collection('losses', weight_decay)
  return var

def getNormalizedLocationWeightsFast(w):
  #Softmax
  a = tf.reduce_max(w, axis=(1,2))
  ew = tf.exp(tf.subtract(w, a[:,None,None,:]))
  sew = tf.reduce_sum(ew, axis=(1,2))
  nw = tf.divide(ew,sew[:,None,None,:])
  return nw

def doOffset2GaussianForward(offset, locs, sigma):
  #based on: https://en.wikipedia.org/wiki/Radial_basis_function_kernel
  feat_denom = tf.reduce_sum(tf.square(tf.subtract(offset, locs[None,:,:,None])), axis=2)
  feat = tf.divide((tf.divide(feat_denom,2)), tf.square(sigma))
  feat = tf.exp(-feat)
  return feat  

def computePredictionLossSL1(pred, target, transition_dist):
  residue = tf.subtract(pred, target)
  dim_losses = tf.abs(residue)

  comparator_lt = tf.less(dim_losses, transition_dist) 

  loss = tf.where(comparator_lt, 
                        tf.divide(tf.square(dim_losses),2),
                        tf.divide(tf.subtract(dim_losses, transition_dist),2))   

  loss = tf.reduce_sum(loss, axis=2)
  loss = tf.reshape(loss, [tf.shape(loss)[0],tf.shape(loss)[1],1,1])
  return loss, residue    

def columnActivation(aug_x, column_num, fwd_dict):
  prev_pred = aug_x[1]        #pc + poc
  prev_loss = aug_x[4]        #loss
  prev_offsets = aug_x[2]     #indiv_preds
  prev_nw = aug_x[3]          #indiv_nw
  x = aug_x[0]                #xx

  res = {}                #DELETE

  chained = True
  if(prev_pred == None):
    chained = False

  num_out_filters = fwd_dict['num_out_filters']
  out_locs = fwd_dict['out_locs']

  with tf.variable_scope('col' + str(column_num) + '1') as scope:
    kernel = _variable_with_weight_decay('weights_col' + str(column_num),
                                         shape=[5, 5, nfc + 1, nfc],
                                         stddev=1,  #check if this is right
                                         wd=0.0)
    kernel = tf.multiply(kernel, 0.0249)      #line 327-334 in warpTrainCNNCDHDCentroidChainGridPredSharedRevFastExp3
    conv = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases_col' + str(column_num), [nfc], tf.constant_initializer(0.1))
    a = tf.nn.bias_add(conv, biases)
    a = tf.nn.relu(a, name=scope.name)

  with tf.variable_scope('col' + str(column_num) + '2') as scope:
    kernel = _variable_with_weight_decay('weights_col' + str(column_num),
                                         shape=[5, 5, nfc, nfc],
                                         stddev=1,  #check if this is right
                                         wd=0.0)
    kernel = tf.multiply(kernel, 0.0250)      #line 327-334 in warpTrainCNNCDHDCentroidChainGridPredSharedRevFastExp3
    conv = tf.nn.conv2d(a, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases_col' + str(column_num), [nfc], tf.constant_initializer(0.1))
    a = tf.nn.bias_add(conv, biases)
    a = tf.nn.relu(a, name=scope.name)

  with tf.variable_scope('col' + str(column_num) + '3') as scope:
    kernel = _variable_with_weight_decay('weights_col' + str(column_num),
                                         shape=[5, 5, nfc, num_out_filters],
                                         stddev=1,  #check if this is right
                                         wd=0.0)
    kernel = tf.multiply(kernel, 0.0250)      #line 327-334 in warpTrainCNNCDHDCentroidChainGridPredSharedRevFastExp3
    conv = tf.nn.conv2d(a, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases_col' + str(column_num), [num_out_filters], tf.constant_initializer(0.1))
    a = tf.nn.bias_add(conv, biases)

    #e.g.: a shape: [10, 42, 32, 26], w shape: [10, 42, 32, 1]
    w = a[:, :, :, 0:1]

    #getNormalizedLocationWeightsFast() returns softmax of the activation w
    nw = getNormalizedLocationWeightsFast(w)

    # nw[0] = batch size, nw[1] = rows, nw[2] = columns 
    nw_reshape = tf.reshape(nw, [tf.shape(nw)[0], -1])

    out_locs_rs = tf.convert_to_tensor(out_locs)
    out_locs_rs = tf.cast(out_locs_rs, tf.float32)

    pc = tf.multiply(nw_reshape[:,:,None], out_locs_rs[None,:,:])

    pc = tf.reduce_sum(pc, axis=1)
    pc = tf.reshape(pc, [tf.shape(pc)[0], 1, tf.shape(pc)[1], 1])

    #Predict the offset (section 3.3 of paper).
    #Use the offset grid to compute the offset.
    offset_grid = fwd_dict['offset_grid']
    num_offset_channels = offset_grid.get_shape().as_list()[2]
    offset_channels = tf.convert_to_tensor(np.arange(grid_stride) + 1)
    num_chans = grid_stride + 1

    offset_wts = a[:, :, :, 1: (grid_stride + 1)]

    # Softmax
    offset_max = tf.reduce_max(offset_wts, axis=3)
    offset_wts = tf.subtract(offset_wts, offset_max[:,:,:,None])
    offset_wts = tf.exp(offset_wts)
    sum_offset_wts = tf.reduce_sum(offset_wts, axis=3)
    offset_wts = tf.divide(offset_wts, sum_offset_wts[:,:,:,None])  #o(j) from section 3.3 of paper

    # offset_wts = tf.nn.softmax(offset_wts)

    offset_grid = tf.reshape(offset_grid, [2, 1, 1, grid_stride])

    of_x = tf.multiply(tf.cast(offset_grid[0,0,0,:], tf.float32), offset_wts)
    of_y = tf.multiply(tf.cast(offset_grid[1,0,0,:], tf.float32), offset_wts)

    of_x = tf.reduce_sum(of_x,axis=3)
    of_y = tf.reduce_sum(of_y,axis=3)

    of_x = tf.reshape(of_x, [tf.shape(of_x)[0], tf.shape(of_x)[1], tf.shape(of_x)[2], 1])
    of_y = tf.reshape(of_y, [tf.shape(of_y)[0], tf.shape(of_y)[1], tf.shape(of_y)[2], 1])

    #po = p(i) of eq 2 from section 3.3 of paper
    po = tf.concat(3, [of_x, of_y], name="conct_of_x_of_y_into_po")

    #poc = P of eq 1 from section 3.2 of paper
    poc = tf.reduce_sum(tf.multiply(po,nw), axis=(1,2))
    poc = tf.reshape(poc, [tf.shape(poc)[0], 1, tf.shape(poc)[1], 1])

    sigma = tf.cast(15, tf.float32)   # RBF sigma

    offset_gauss = doOffset2GaussianForward(tf.add(pc, poc), out_locs_rs, sigma)

    res['offset_gauss'] = tf.shape(offset_gauss)    #DELETE
    offset_gauss = tf.reshape(offset_gauss, [tf.shape(a)[0], tf.shape(a)[1], tf.shape(a)[2], 1])

    if chained:
      cent_loss, cent_residue = computePredictionLossSL1(prev_pred, pc, transition_dist)

      offs_loss, offs_residue = computePredictionLossSL1(prev_offsets, pc, transition_dist)
      offs_loss = tf.reduce_sum(tf.multiply(offs_loss, prev_nw), axis=1)
      offs_loss = tf.reshape(offs_loss, [tf.shape(offs_loss)[0],1,1,1])
    else:
      cent_residue = pc * 0;
      cent_loss = 0;
      offs_residue = 0;
      offs_loss = 0;

    #indiv_preds
    po = tf.reshape(po, [tf.shape(po)[0], tf.shape(po)[1] * tf.shape(po)[2], tf.shape(po)[3], 1])
    indiv_preds = tf.add(out_locs_rs[None, :, :, None], po)

    #indiv_nw
    indiv_nw = tf.reshape(nw, [tf.shape(nw)[0], tf.shape(nw)[1] * tf.shape(nw)[2], tf.shape(nw)[3], 1])

    #TODO: check if it's ok to multiply current iteration's preds with ptrv iter weights
    #TODO: if learning does not converge, check how close this formula is to formula (4) in paper
    loss = prev_loss + (cent_loss + offs_loss * offset_pred_weight) * prev_pred_weight;    

    xx = offset_gauss;
    out_x = [xx, pc + poc, indiv_preds, indiv_nw, loss]

    # res = {}
    res['x'] = out_x
    res['pred'] = pc;
    res['w'] = w;
    res['pc'] = pc
    res['po'] = po
    res['poc'] = poc
    res['nw'] = nw
    res['dwn'] = nw     #TODO: check if this is right
    res['offset_wts'] = offset_wts
    # res['offs_loss'] = offs_loss * offset_pred_weight * prev_pred_weight
    res['cent_residue'] = cent_residue
    res['offs_residue'] = offs_residue
    res['nzw_frac'] = 0 #TODO: what is this?
    res['interim'] = a  #TODO: check if this is right

    return res  

def doForwardPass(x, out_locs, gt_loc):

  res_aux = {}    #DELETE

  grid_x = np.arange(-grid_size, grid_size + 1, grid_stride)
  grid_y = np.arange(-grid_size, grid_size + 1, grid_stride)

  offset_grid_list = []
  for xi in xrange(grid_x.shape[0]):
    for yi in xrange(grid_y.shape[0]):
      offset_grid_list.append((grid_x[xi], grid_y[yi]))

  offset_grid = np.asarray(offset_grid_list) 
  offset_grid = tf.convert_to_tensor(offset_grid)
  offset_grid = tf.transpose(offset_grid)
  offset_grid_shape = offset_grid.get_shape().as_list()
  offset_grid = tf.reshape(offset_grid, [1, offset_grid_shape[0],offset_grid_shape[1]])

  #25 + 1 filters: 25 for the offsets and 1 for the gt coordinate
  num_out_filters = offset_grid.get_shape().as_list()[2] + 1; 

  n = tf.shape(x)[1] * tf.shape(x)[2]
  
  xa = tf.multiply(tf.cast(tf.divide(tf.ones([tf.shape(x)[0], \
                tf.shape(x)[1],tf.shape(x)[2],1], tf.int32), n), tf.float32), pred_factor)

  x = tf.concat(3, [xa,x])    #IN NEWER VERSION OF TF CORRECT COMMAND IS: tf.concat([xa,x], 3)

  res_steps = []

  fwd_dict = {}
  fwd_dict['num_out_filters'] = num_out_filters
  fwd_dict['out_locs'] = out_locs
  fwd_dict['offset_grid'] = offset_grid

  # pdb.set_trace()

  tf.assert_equal(tf.convert_to_tensor(tf.shape(out_locs)[0]), \
                  tf.multiply(tf.convert_to_tensor(tf.shape(x)[1]),\
                              tf.convert_to_tensor(tf.shape(x)[2])))                        

  aug_x = [x, None, None, None, 0]

  all_preds = None
  all_cents = None

  res_step = None
  for i in xrange(steps):
    res_step = columnActivation(aug_x, i, fwd_dict)
    res_steps.append(res_step)
    out_x = res_step['x']

    #Output of step 1 goes as input to step 2 and output of step 2 goes as input to step 3
    #TODO: check size of x_sans_xa after data is fed. Note: checked on 12/3/17 - passed

    x_sans_xa = tf.slice(aug_x[0], [0,0,0,1], [tf.shape(aug_x[0])[0], tf.shape(aug_x[0])[1], \
                    tf.shape(aug_x[0])[2], -1])

    # out_x[0] = output of previous layer
    out_x[0] = tf.reshape(out_x[0], [tf.shape(x_sans_xa)[0], tf.shape(x_sans_xa)[1], \
                    tf.shape(x_sans_xa)[2], 1])

    #combining activation of sixth convolution with output of previous layer
    #TODO: not required for last column
    aug_x[0] = tf.concat(3, [out_x[0],x_sans_xa], name='concat_x_and_a_sans_xa') #xx

    aug_x[1] = out_x[1]     #pc + poc
    aug_x[2] = out_x[2]     #indiv_preds
    aug_x[3] = out_x[3]     #indiv_nw
    aug_x[4] = out_x[4]     #loss

    if(i == 0):
      all_preds = aug_x[1]
      all_cents = res_step['pc']
    else:  
      all_preds = tf.concat(1,[all_preds, tf.cast(aug_x[1], tf.float32)])
      all_cents = tf.concat(1,[all_cents, tf.cast(res_step['pc'], tf.float32)])

  gt_loc = tf.convert_to_tensor(gt_loc)
  gt_loc = tf.cast(gt_loc, tf.float32)
  gt_loc = tf.reshape(gt_loc, [tf.shape(gt_loc)[0],1, tf.shape(gt_loc)[1],1])

  #Compute the loss.
  target_loss, target_residue = computePredictionLossSL1(res_step['x'][1], gt_loc, transition_dist)
  offs_loss, offs_residue = computePredictionLossSL1(res_step['x'][2], gt_loc, transition_dist)

  offs_loss = tf.reduce_sum(tf.multiply(offs_loss, res_step['x'][3]), axis=1)
  # offs_loss = tf.reshape(offs_loss, [offs_loss.get_shape().as_list()[0],1,1,1])
  offs_loss = tf.reshape(offs_loss, [tf.shape(offs_loss)[0],1,1,1])

  loss = tf.add(tf.add(target_loss, res_step['x'][4]),
                tf.multiply(offs_loss, offset_pred_weight))
  
  pred = res_step['x'][1]

  # res_aux = {}
  # res = res_ip1;
  res_aux['loss'] = loss    # == res.x = loss; line 64 of 'centroidChainGrid9LossLayer()'
  res_aux['pred'] = pred  
  res_aux['all_preds'] = all_preds  
  res_aux['all_cents'] = all_cents  
  res_aux['res_steps'] = res_steps  
  # res.aux.nzw_frac = 0;
  res_aux['target_residue'] = target_residue  
  res_aux['offs_residue'] = offs_residue  
  res_aux['offs_loss'] = tf.multiply(offs_loss, offset_pred_weight)  

  return res_aux

def inference(images,out_locs,org_gt_coords):
  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 3, 32],
                                         stddev=1,  #check if this is right
                                         wd=0.0)
    kernel = tf.multiply(kernel, 0.2722)        #line 321-325 in warpTrainCNNCDHDCentroidChainGridPredSharedRevFastExp3
    conv = tf.nn.conv2d(images, kernel, [1, 2, 2, 1], padding='VALID')
    biases = _variable_on_cpu('biases', [32], tf.constant_initializer(1.0))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(pre_activation, name=scope.name)
    
  #TODO: check if activation_summary is required
  #TODO: check if normalization is required (https://www.tensorflow.org/api_docs/python/tf/nn/local_response_normalization) 

  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 32, 64],
                                         stddev=1,
                                         wd=0.0)
    kernel = tf.multiply(kernel, 0.0833)        #line 321-325 in warpTrainCNNCDHDCentroidChainGridPredSharedRevFastExp3
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
    kernel = tf.multiply(kernel, 0.0589)        #line 321-325 in warpTrainCNNCDHDCentroidChainGridPredSharedRevFastExp3
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
    kernel = tf.multiply(kernel, 0.0589)        #line 321-325 in warpTrainCNNCDHDCentroidChainGridPredSharedRevFastExp3
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
    kernel = tf.multiply(kernel, 0.0589)        #line 321-325 in warpTrainCNNCDHDCentroidChainGridPredSharedRevFastExp3
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
    kernel = tf.multiply(kernel, 0.0250)        #line 321-325 in warpTrainCNNCDHDCentroidChainGridPredSharedRevFastExp3
    conv = tf.nn.conv2d(conv5, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [128], tf.constant_initializer(1.0))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv6 = tf.nn.relu(pre_activation, name=scope.name)  

  res_aux = doForwardPass(conv6, out_locs, org_gt_coords)
  return res_aux

def train(res_aux, global_step):

  # print(tf.get_collection('weights_col2')[0].get_shape().as_list())

  ret_dict = {}
  ret_dict['loss'] = res_aux['loss']
  col_2_loss = res_aux['loss']

  # col_2_loss = np.sum(res_aux['loss'], axis=0)[0,0,0]

  # a_optimizer_col_2.minimize(col_2_loss, var_list=tf.get_collection(tf.GraphKeys.VARIABLES))

  # a_optimizer_col_2 = tf.train.AdamOptimizer()
  # a_optimizer_col_2.__init__(
  #   learning_rate=0.00001,
  #   beta1=0.9,
  #   beta2=0.999,
  #   epsilon=1e-08,
  #   use_locking=False,
  #   name='Adam_2')

  # a_optimizer_col_2 = tf.train.GradientDescentOptimizer(learning_rate=0.001) 
  a_optimizer_col_2 = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.0003) 

  var_list_2 = []
  var_list_2 = var_list_2 + tf.get_collection('weights_col2')
  var_list_2 = var_list_2 + tf.get_collection('biases_col2')
  var_list_2 = var_list_2 + tf.get_collection('weights') 
  var_list_2 = var_list_2 + tf.get_collection('biases')

  #sanity check
  assert tf.get_collection('weights_col2')[0].graph is tf.get_default_graph(), " weights from default graph"

  # grad_var_2 = a_optimizer_col_2.compute_gradients(col_2_loss, var_list_2)
  # grad_var_2 = a_optimizer_col_2.compute_gradients(col_2_loss, var_list=tf.get_collection('weights_col2'))
  # grad_var_2 = a_optimizer_col_2.compute_gradients(col_2_loss, \
  #                     var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
  # grad_var_2 = a_optimizer_col_2.compute_gradients(col_2_loss)  
  
  # ret_dict['grad_var'] = grad_var_2
  # ret_dict['weights_col2_before'] = tf.get_collection('weights_col2')
  # ret_dict['var_list_2'] = var_list_2  
  
  # app_grad = a_optimizer_col_2.apply_gradients(grad_var_2, global_step=global_step)
  # ret_dict['app_grad'] = app_grad

  minimizer_col2 = a_optimizer_col_2.minimize(col_2_loss, var_list=var_list_2, global_step=global_step)
  ret_dict['minimizer_col2'] = minimizer_col2

  # ret_dict['weights_col2_after'] = tf.get_collection('weights_col2')
  # ret_dict['global_step'] = global_step

  col_1_loss = (res_aux['res_steps'][2])['x'][4]
  # a_optimizer_col_1 = tf.train.AdamOptimizer()
  # a_optimizer_col_1.__init__(
  #   learning_rate=0.00001,
  #   beta1=0.9,
  #   beta2=0.999,
  #   epsilon=1e-08,
  #   use_locking=False,
  #   name='Adam_1')

  # a_optimizer_col_1 = tf.train.GradientDescentOptimizer(learning_rate=0.001)
  a_optimizer_col_1 = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.0003)  

  var_list_1 = []
  var_list_1 = var_list_1 + tf.get_collection('weights_col1')
  var_list_1 = var_list_1 + tf.get_collection('biases_col1')
  var_list_1 = var_list_1 + tf.get_collection('weights') 
  var_list_1 = var_list_1 + tf.get_collection('biases')

  # grad_var_1 = a_optimizer_col_1.compute_gradients(col_1_loss, var_list=tf.get_collection('weights_col1'))
  # a_optimizer_col_1.apply_gradients(grad_var_1, global_step=global_step)  
  minimizer_col1 = a_optimizer_col_1.minimize(col_1_loss, var_list=var_list_1, global_step=global_step)
  ret_dict['minimizer_col1'] = minimizer_col1

  col_0_loss = (res_aux['res_steps'][1])['x'][4]
  # a_optimizer_col_0 = tf.train.AdamOptimizer()
  # a_optimizer_col_0.__init__(
  #   learning_rate=0.00001,
  #   beta1=0.9,
  #   beta2=0.999,
  #   epsilon=1e-08,
  #   use_locking=False,
  #   name='Adam_0')

  # a_optimizer_col_0 = tf.train.GradientDescentOptimizer(learning_rate=0.001) 
  a_optimizer_col_0 = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.0003) 
  
  var_list_0 = []
  var_list_0 = var_list_0 + tf.get_collection('weights_col0')
  var_list_0 = var_list_0 + tf.get_collection('biases_col0')
  var_list_0 = var_list_0 + tf.get_collection('weights') 
  var_list_0 = var_list_0 + tf.get_collection('biases')

  # grad_var_0 = a_optimizer_col_0.compute_gradients(col_0_loss, var_list=tf.get_collection('weights_col0'))
  # a_optimizer_col_0.apply_gradients(grad_var_0, global_step=global_step)  
  minimizer_col0 = a_optimizer_col_0.minimize(col_0_loss, var_list=var_list_0, global_step=global_step)
  ret_dict['minimizer_col0'] = minimizer_col0

  # for op in tf.get_default_graph().get_operations():
  #   print str(op.name) 

  return ret_dict


def buildModelAndTrain(images,out_locs,org_gt_coords, global_step):
  res_aux = inference(images,out_locs,org_gt_coords)
  ret_dict = train(res_aux, global_step)

  ret_dict['loss'] = res_aux['loss']
  # ret_dict['conv6'] = tf.as_string(conv6, scientific=None)
  ret_dict['offset_gauss'] = tf.as_string(res_aux['res_steps'][steps - 3]['offset_gauss'], scientific=None)
  # ret_dict['all_preds_1'] = tf.as_string(res_aux['all_preds_1'])

  return ret_dict

