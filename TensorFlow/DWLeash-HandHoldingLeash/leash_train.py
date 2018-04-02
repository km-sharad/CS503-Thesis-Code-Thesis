import leash
import random
import tensorflow as tf
from PIL import Image
from math import sqrt
import numpy as np
import time
import leash_input
from tensorflow.python import debug as tf_debug
import sys
from scipy.misc import imresize

data_dir = 'dog_walking_dataset/'
total_visible_training_images = 250     # Number of training images where car door handle is visible
total_visible_validation_images = 60    # Number of validation images where car door handle is visible
max_steps = 5000                        # Number of batches to run
stats_sample_size = 100                 # Number of images to calculate mean and sd
batch_size = 10                         # Number of images to process in a batch
max_im_side = 500

def computeNormalizationParameters():

  all_train_visible_idx = [x for x in xrange(0,total_visible_training_images)]
  random.shuffle(all_train_visible_idx)
  stats_sample_indexes = all_train_visible_idx[0:stats_sample_size]

  anno_file_batch_rows = []
  anno_file = open('leash_anno_training_data.txt')
  anno_file_lines = anno_file.readlines()

  for x in stats_sample_indexes:
    anno_file_batch_rows.append(anno_file_lines[x])
 
  mean_pixel = np.zeros(3);
  mean_pixel_sq = np.zeros(3);
  pixel_covar = np.zeros((3, 3));

  num_pixel = 0
  for image_idx in xrange(stats_sample_size):
    image_filename = anno_file_batch_rows[image_idx].split('|')[4]
    image = Image.open(data_dir + image_filename)

    im = np.array(image, dtype=np.uint64)
    if(len(im.shape) == 2):
      #monochrome image, add the third channel
      im = np.stack((image,)*3)
      print('monochrome image for mean calc: ', image_filename)      

    #scale
    scale = round(max_im_side/float(np.amax(im.shape)),4)
    im = imresize(im, scale, interp='bilinear')
    im = im.astype(np.uint64)

    #reshape
    im = im.reshape(im.shape[0] * im.shape[1],im.shape[2])
    npix = im.shape[0]

    mean_pixel = mean_pixel * (float(num_pixel)/(float(num_pixel + npix))) \
                    + np.sum(im, axis=0)/((float(num_pixel + npix)))
    mean_pixel_sq = mean_pixel_sq * (float(num_pixel)/(float(num_pixel + npix))) \
                    + np.sum(np.square(im), axis=0)/(float(num_pixel + npix))

    num_pixel = num_pixel + npix;

  epsilon = 0.001;
  std_pixel = np.sqrt(mean_pixel_sq - np.square(mean_pixel)) + epsilon
  stats_dict = {'mean_pixel': mean_pixel, 'std_pixel': std_pixel}

  #store values so that there's no need to compute next time    
  

  return stats_dict

def getImageMetaRecords(all_train_visible_idx):
  # all_train_visible_idx = [x for x in xrange(0,total_visible_training_images)]
  random.shuffle(all_train_visible_idx)  

  anno_file_batch_rows = []
  anno_file = open('leash_anno_training_data.txt')
  anno_file_lines = anno_file.readlines()

  for x in all_train_visible_idx:
    anno_file_batch_rows.append(anno_file_lines[x])

  return anno_file_batch_rows

def getValidationImageMetaRecords():
  all_validation_visible_idx = [x for x in xrange(0, total_visible_validation_images)]
  random.shuffle(all_validation_visible_idx)

  validation_anno_file_batch_rows = []
  validation_anno_file = open('leash_anno_val_data.txt')
  validation_anno_file_lines = validation_anno_file.readlines()  

  for x in all_validation_visible_idx:
    validation_anno_file_batch_rows.append(validation_anno_file_lines[x])  

  return validation_anno_file_batch_rows[0:batch_size]  

def calculteNormalizedValidationDistance(pred, original, bbox_heights):
  total_normalized_distance = 0
  for i in xrange(original.shape[0]): 
    print(pred[i][0][0][0],pred[i][0][1][0],original[i][0], original[i][1], bbox_heights[i])
    mse = sqrt(pow((pred[i][0][0][0] - original[i][0]), 2) + pow((pred[i][0][1][0] - original[i][1]), 2))
    normalized_dist = mse/float(bbox_heights[i])    
    # print(normalized_dist)
    total_normalized_distance = total_normalized_distance + normalized_dist

  # return average normalized distance for the batch of 10 images
  return total_normalized_distance/float(original.shape[0])

global_step = tf.Variable(0, trainable=False, dtype=tf.int32)
images = tf.placeholder(dtype=tf.float32, shape=[batch_size, None, None, 3])
out_locs = tf.placeholder(dtype=tf.float32, shape=[None, 2])
org_gt_coords = tf.placeholder(dtype=tf.float32, shape=[batch_size, 2])   

stats_dict = computeNormalizationParameters() 

res_aux = leash.inference(images,out_locs,org_gt_coords)

ret_dict = leash.train(res_aux, global_step)

val_dict = leash.test(res_aux, global_step)

init = tf.global_variables_initializer()

saver = tf.train.Saver(max_to_keep=20)

with tf.Session() as sess:
# with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:  
  writer = tf.summary.FileWriter('./graphs', sess.graph)
  sess.run(init)

  # Restore variables from disk.
  saver.restore(sess, "./ckpt/model4999.ckpt")
  print("Model restored.")

  # Following two lines are for debugging
  # Use <code> python leash_train.py --debug </code> command to debug
  # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
  # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

  all_train_visible_idx = [x for x in xrange(0,total_visible_training_images)]
  #random.shuffle(all_train_visible_idx)  

  for epoch in xrange(max_steps):
    start_time = time.time()
    anno_file_batch_rows = getImageMetaRecords(all_train_visible_idx) 
    print('epoch: ', epoch)

    for batch in xrange(len(anno_file_batch_rows)/batch_size):
      distorted_images, meta = leash_input.distorted_inputs(stats_dict, batch_size, \
              anno_file_batch_rows[batch * batch_size : (batch * batch_size) + batch_size])

      out_dict = sess.run(ret_dict, feed_dict=
                            {images: distorted_images, 
                            out_locs: meta['out_locs'],
                            org_gt_coords: meta['org_gt_coords']})

      avg_train_normalized_dist = calculteNormalizedValidationDistance(out_dict['pred_coord'], 
                                            meta['org_gt_coords'],
                                            meta['bbox_heights'])      
      out_f_train = open('train_file.txt', 'a+')
      out_f_train.write(str(epoch) + ' ' + str(batch) + ' ' + str(avg_train_normalized_dist) + '\n')
      out_f_train.close()      
      print('*** loss: ', out_dict['loss'])

      out_f = open('out_file.txt', 'a+')
      out_f.write(str(epoch) + ' ' + str(batch) + ' ' + str(out_dict['loss']) + '\n')
      out_f.close()

    # Save the variables to disk.
    ckpt_file = './ckpt/model' + str(epoch) + '.ckpt'
    save_path = saver.save(sess, ckpt_file)
    print("Model saved in file: %s" % save_path)

    duration = time.time() - start_time
    print('time taken for epoch ' + str(epoch) + ': ' + str(duration))

    out_f_epoch = open('out_epoch.txt', 'a+')
    out_f_epoch.write(str(epoch) + ' ' + str(out_dict['loss']) + '\n')
    out_f_epoch.close()    

    # Validation step
    if((epoch % 2) == 0):
      validation_images, validation_meta = leash_input.distorted_inputs(stats_dict, batch_size, getValidationImageMetaRecords())
      validation_dict = sess.run(val_dict, feed_dict =
                                    {images: validation_images, 
                                    out_locs: validation_meta['out_locs'],
                                    org_gt_coords: validation_meta['org_gt_coords']})  
      avg_normalized_dist = calculteNormalizedValidationDistance(validation_dict['pred_coord'], 
                                            validation_meta['org_gt_coords'],
                                            validation_meta['bbox_heights'])
      out_f_validation = open('validation_epoch.txt', 'a+')
      out_f_validation.write(str(epoch) + ' ' + str(avg_normalized_dist) + '\n')
      out_f_validation.close()         
    
  writer.close() 
