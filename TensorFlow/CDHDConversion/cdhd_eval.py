import tensorflow as tf
import cdhd_input
import cdhd
import random
import numpy as np 

total_visible_testing_images = 1200     # Number of testing images where car door handle is visible
batch_size = 10                         # Number of images to process in a batch

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

  return testing_anno_file_batch_rows[0:10]

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
	saver.restore(sess, "./ckpt/model1488.ckpt")
	print("Model loaded.")	 

	# meta = cdhd_input.getTestImages(getTestImageMetaRecords()) 
	distorted_images, meta = cdhd_input.distorted_inputs(stats_dict, batch_size, getTestImageMetaRecords())	  	 	

  	sess.run(tf.global_variables_initializer())
  	output = sess.run(ret_dict, {
  						images: distorted_images,
  						out_locs: meta['out_locs'],
              			org_gt_coords: meta['org_gt_coords']})

  	np.set_printoptions(suppress=True)
  	print('actual coord: ', output['actual_coord'])
  	print('pred coord: ', output['pred_coord'])

def main(argv=None):  
  # start_time = time.time()
  # anno_file_batch_rows = getImageMetaRecords() 

  # for batch in xrange(len(anno_file_batch_rows)/batch_size):	
  evaluate()


if __name__ == '__main__':
  tf.app.run()