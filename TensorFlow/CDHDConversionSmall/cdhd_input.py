import tensorflow as tf
import random
import numpy as np
from PIL import Image
from numpy import array
from math import log
from math import exp
from scipy.misc import imresize
from scipy.ndimage import zoom

data_dir = '../../../../car_dataset/'
max_im_side = 500
init_padding = 32
min_side = 64
start_offset = 16
output_stride = 16
scaling_factor = round(500/float(3), 4)

def distorted_inputs(stats_dict, batch_size, anno_file_batch_rows):
  """Construct distorted input for CIFAR training using the Reader ops.

  Args:
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """

  images = []
  target_locs = []
  infos = []

  for image_idx in xrange(batch_size):
    #read and convert images into numpy array
    meta_rec = anno_file_batch_rows[image_idx].split('|')
    [image, target_loc, info] = getImage(meta_rec, stats_dict)

    images.append(image)
    target_locs.append(target_loc)
    infos.append(info)

  im_sizes = np.asarray([info['im_size'] for info in infos])
  padding = np.asarray([info['padding'] for info in infos])
  aug_target_loc = np.asarray([info['aug_target_loc'] for info in infos])
  final_scale = np.asarray([info['final_scale'] for info in infos])
  im_org = np.asarray([info['im_org'] for info in infos])

  padded_images, cat_padding = concatWithPadding(np.asarray(images), np.asarray(im_sizes))
  padding = padding + cat_padding

  x = np.arange(start_offset, padded_images.shape[1], output_stride)
  y = np.arange(start_offset, padded_images.shape[2], output_stride)


  out_locs_list = []
  for xi in xrange(x.shape[0]):
    for yi in xrange(y.shape[0]):
      out_locs_list.append((x[xi], y[yi]))

  out_locs = np.asarray(out_locs_list)

  gt_coords = np.round(np.divide(np.subtract(np.asarray(target_locs),start_offset), 
                  float((output_stride + 1))),2)
  org_gt_coords = np.asarray(target_locs)

  '''
  out_locs = out_locs ./ opts.scaling_factor;
  org_gt_coords = org_gt_coords ./ opts.scaling_factor;
  aug_target_loc = aug_target_loc ./ opts.scaling_factor;
  '''

  out_locs = np.round(np.divide(out_locs, scaling_factor),4)
  org_gt_coords = np.round(np.divide(org_gt_coords, scaling_factor),4)
  aug_target_loc = np.round(np.divide(aug_target_loc, scaling_factor),4)

  meta_dict = {}
  meta_dict['margins'] = []
  meta_dict['out_locs'] = out_locs
  meta_dict['out_locs_width'] = x.shape[0]
  meta_dict['out_locs_height'] = y.shape[0]
  meta_dict['scale'] = final_scale
  meta_dict['gt_coords'] = gt_coords
  meta_dict['org_gt_coords'] = org_gt_coords
  meta_dict['aug_target_loc'] = aug_target_loc;
  meta_dict['im_org'] = im_org;
  #meta_dict['im_org_scaled'] = ?;
  #meta_dict['torso_height'] = ?;

  # return images_tensor, meta_dict
  return padded_images, meta_dict

def getImage(meta_rec, stats_dict):
    im_meta_dict = {}
    im_meta_dict['gt_x_coord'] = int(float(meta_rec[0]))
    im_meta_dict['gt_y_coord'] = int(float(meta_rec[1]))
    im_meta_dict['img_size'] = [int(i) for i in meta_rec[3].split(',')]
    im_meta_dict['bbox'] = [int(i) for i in meta_rec[8][0:len(meta_rec[8]) - 2].split(',')]    

    image = Image.open(data_dir + meta_rec[2])

    # print('image name: ', meta_rec[2])
    # print('im_meta_dict[img_size]: ', im_meta_dict['img_size'][0], im_meta_dict['img_size'][1])
    # print('image size: ', image.size[0], image.size[1])
    # print('co-ords: x-y', im_meta_dict['gt_x_coord'], im_meta_dict['gt_y_coord'])

    try:
      im = np.array(image.getdata()).reshape(image.size[0], image.size[1], 3)
    except ValueError:
      im = np.dstack([np.array(image.getdata()).reshape(image.size[0], image.size[1])]*3)

    im_org = im      

    #Compute the default scaling
    scale = round(max_im_side/float(np.amax(im.shape)),4)    

    [im, target_loc, aug_scale] = getAugmentedImage(im, im_meta_dict)

    aug_target_loc = target_loc
    im_org_scaled = imresize(im, scale,interp='bilinear')
    im = im_org_scaled
    target_loc[0] = int(round(target_loc[0] * scale,0))    
    target_loc[1] = int(round(target_loc[1] * scale,0))    

    im = np.subtract(im, stats_dict['mean_pixel'])
    im = np.divide(im, stats_dict['std_pixel'])

    im_size = np.asarray(im.shape)

    padding = np.zeros(4) #T/B/L/R - top, bottom, left, right
    padding = padding + init_padding;

    target_loc = target_loc + init_padding
    im_size[0] = im_size[0] + (2 * init_padding)
    im_size[1] = im_size[1] + (2 * init_padding)

    #Pad if smaller than minimum size
    min_side_padding = np.maximum(np.subtract(min_side - im_size[0:2],np.zeros(2)),np.zeros(2))
    padding[1] = padding[1] + min_side_padding[0]
    padding[3] = padding[3] + min_side_padding[1]
    im_size[0] = im_size[0] + min_side_padding[0]
    im_size[1] = im_size[1] + min_side_padding[1]

    size_padding = computePaddingForImage(im_size, output_stride)
    padding[1] = padding[1] + size_padding[0]
    padding[3] = padding[3] + size_padding[1]
    im_size[0] = im_size[0] + size_padding[0]
    im_size[1] = im_size[1] + size_padding[1]    

    padding = padding.astype(int)

    padding_tuple = ((padding[0],padding[1]), (padding[2],padding[3]), (0,0))
    im = np.pad(im, padding_tuple,'edge')

    assert(np.array_equal(im_size, np.asarray(im.shape))), "assertion error in image preprocessing"

    final_scale = aug_scale * scale

    info = {}
    info['aug_scale'] = aug_scale
    info['aug_target_loc'] = aug_target_loc
    info['im_org'] = im_org
    info['im_size'] = im_size
    info['final_scale'] = final_scale
    #info.torso_height = ?
    #info.im_org_scaled = ?
    info['padding'] = padding

    return [im, target_loc, info]

def getAugmentedImage(im, im_meta_dict):
  (im, im_meta_dict) = lrFlipCDHDDataRecord(im, im_meta_dict)
  targets = [im_meta_dict['gt_x_coord'], im_meta_dict['gt_y_coord']]  

  #add random scale jitter
  scale_range = [round(log(0.6),4), round(log(1.25),4)]
  scale = round(exp(random.uniform(0, 1) * (scale_range[1] - scale_range[0]) + scale_range[0]),2)

  im = imresize(im, scale,interp='bilinear')
  targets_np = np.zeros(2)
  targets_np[0] = int(round(targets[0] * scale,0))
  targets_np[1] = int(round(targets[1] * scale,0))
  
  #TO-DO: random crop jitter

  return [im, targets_np, scale]

def lrFlipCDHDDataRecord(im, im_meta_dict):
  if random.uniform(0, 1) > 0.5:
    im = np.fliplr(im)

    #change x-coordinate of ground truth after flipping image
    im_meta_dict['gt_x_coord'] = im.shape[0] - im_meta_dict['gt_x_coord']

    #change x-coordinate of bounding box after flipping image
    temp = im_meta_dict['bbox'][0]
    im_meta_dict['bbox'][0] = im.shape[0] - im_meta_dict['bbox'][2]
    im_meta_dict['bbox'][2] = im.shape[0] - temp

  return im, im_meta_dict

def concatWithPadding(images, im_sizes):
  max_dim = np.amax(im_sizes, axis=0)[0:2]
  paddings = np.zeros([len(images), 4])
  padded_images = []

  #TODO: find a way to perform following two steps in one operations
  paddings[:,0::3] = np.subtract(max_dim, im_sizes[:,:2])
  paddings[:, 0], paddings[:, 1] = paddings[:, 1], paddings[:, 0].copy()

  paddings = paddings.astype(int)

  for idx in xrange(len(images)):
    padding_tuple = (((paddings[idx])[0], (paddings[idx])[1]), ((paddings[idx])[2], (paddings[idx])[3]), (0,0))
    padded_images.append(np.pad(images[idx], padding_tuple,'edge'))

  return np.asarray(padded_images), paddings

def computePaddingForImage(im_size, stride):
  padding = np.remainder(im_size[0:2],stride)
  padding = np.remainder(np.subtract(stride, padding[0:2]), stride)
  return padding
