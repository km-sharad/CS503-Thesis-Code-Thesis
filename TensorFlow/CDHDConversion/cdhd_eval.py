import tensorflow as tf
import cdhd_input
import cdhd
import random
from math import sqrt
import numpy as np 
import time
import sys

total_visible_testing_images = 1200     # Number of testing images where car door handle is visible
batch_size = 10                         # Number of images to process in a batch
max_steps = 200

def computeNormalizationParameters():

  mean_pixel = np.zeros(3);
  mean_pixel[0] = 118.1365
  mean_pixel[1] = 114.5391
  mean_pixel[2] = 111.4741

  # mean_pixel_sq = np.zeros(3);
  
  mean_pixel_sq = np.zeros(3);
  mean_pixel_sq[0] = 19350.7498
  mean_pixel_sq[1] = 18537.0203
  mean_pixel_sq[2] = 18291.5741  

  pixel_covar = np.zeros((3, 3));

  mean_pixel_113 = np.zeros((1,1,3))
  mean_pixel_113[0][0][0] = mean_pixel[0]
  mean_pixel_113[0][0][1] = mean_pixel[1]
  mean_pixel_113[0][0][2] = mean_pixel[2]

  mean_pixel_sq_113 = np.zeros((1,1,3))
  mean_pixel_sq_113[0][0][0] = mean_pixel_sq[0]
  mean_pixel_sq_113[0][0][1] = mean_pixel_sq[1]
  mean_pixel_sq_113[0][0][2] = mean_pixel_sq[2]

  std_pixel = np.sqrt(mean_pixel_sq_113 - (mean_pixel_113 ** 2))
  stats_dict = {'mean_pixel': mean_pixel_113, 'std_pixel': std_pixel, 'pixel_covar': pixel_covar}
  
  return stats_dict

def getTestImageMetaRecords():
  all_testing_visible_idx = [x for x in xrange(0,total_visible_testing_images)]
  random.shuffle(all_testing_visible_idx)

  testing_anno_file_batch_rows = []
  testing_anno_file = open('cdhd_anno_testing_data.txt')
  testing_anno_file_lines = testing_anno_file.readlines()  

  for x in all_testing_visible_idx:
    testing_anno_file_batch_rows.append(testing_anno_file_lines[x])  

  return testing_anno_file_batch_rows

def calculteNormalizedValidationDistance(pred, original, bbox_heights):
  total_normalized_distance = 0
  for i in xrange(original.shape[0]): 
    # print(pred[i][0][0][0],pred[i][0][1][0],original[i][0], original[i][1], bbox_heights[i])
    mse = sqrt(pow((pred[i][0][0][0] - original[i][0]), 2) + pow((pred[i][0][1][0] - original[i][1]), 2))
    normalized_dist = mse/float(bbox_heights[i])    
    # print(normalized_dist)
    total_normalized_distance = total_normalized_distance + normalized_dist

  # return average normalized distance for the batch of 10 images
  return total_normalized_distance/float(original.shape[0])  

def evaluate():
  global_step = tf.Variable(0, trainable=False, dtype=tf.int32)
  images = tf.placeholder(dtype=tf.float32, shape=[batch_size, None, None, 3])
  out_locs = tf.placeholder(dtype=tf.float32, shape=[None, 2])
  org_gt_coords = tf.placeholder(dtype=tf.float32, shape=[batch_size, 2])  

  stats_dict = computeNormalizationParameters()

  test_out_dict = cdhd.inference(images, out_locs, org_gt_coords)

  ret_dict = cdhd.train(test_out_dict, global_step)

  saver = tf.train.Saver()

  with tf.Session() as sess:
	# Load weights from disk.
	saver.restore(sess, "./ckpt/model237.ckpt")
	print("Model loaded.")	 

	sess.run(tf.global_variables_initializer())

	for epoch in xrange(max_steps):
	    start_time = time.time()
	    anno_file_batch_rows = getTestImageMetaRecords() 
	    print('epoch: ', epoch)	    

	    for batch in xrange(len(anno_file_batch_rows)/batch_size):
      		distorted_images, meta = cdhd_input.distorted_inputs(stats_dict, batch_size, \
              anno_file_batch_rows[batch * batch_size : (batch * batch_size) + batch_size])			

      		output = sess.run(ret_dict, {
		  						images: distorted_images,
		  						out_locs: meta['out_locs'],
		              			org_gt_coords: meta['org_gt_coords']})

      		avg_normalized_dist = calculteNormalizedValidationDistance(output['pred_coord'], 
		                                          meta['org_gt_coords'],
		                                          meta['bbox_heights'])

      		print('epoch: ', str(epoch) + ' batch: ' + str(batch) + ' ' + str(avg_normalized_dist))

      		out_f_test = open('out_test_batch.txt', 'a+')
      		out_f_test.write(str(epoch) + ' ' + str(batch) + ' ' + str(avg_normalized_dist) + '\n')
      		out_f_test.close()

      		np.set_printoptions(suppress=True)
      		print('actual coord: ', meta['org_gt_coords'])
      		print('pred coord: ', output['pred_coord'])
      		sys.exit()

    	duration = time.time() - start_time
    	print('time taken for epoch ' + str(epoch) + ': ' + str(duration))	  	

def main(argv=None):  
  # start_time = time.time()
  # anno_file_batch_rows = getImageMetaRecords() 

  # for batch in xrange(len(anno_file_batch_rows)/batch_size):	
  evaluate()


if __name__ == '__main__':
  tf.app.run()