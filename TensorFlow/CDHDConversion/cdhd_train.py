import cdhd
import random
import tensorflow as tf

'''
FLAGS = tf.app.flags.FLAGS
all_training_data = []
training_rows_indexes = []
batches = []
total_batches = 0
batch_idx = 0
train_dir = 'tbd'
epochs = 2600
batch_size = 10
'''

'''
#todo: fix this path
tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")

tf.app.flags.DEFINE_integer('epochs', 2600,
                            """Number of epochs to run.""")

tf.app.flags.DEFINE_integer('batch_size', 10, """Number of images in a batch.""")

#todo: fix this path
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
'''

tf.app.flags.DEFINE_integer('max_steps', 26000,
                            """Number of batches to run.""")

def train():

  """
  tf.Graph().as_default():
  This method should be used if you want to create multiple graphs in the same process. 
  For convenience, a global default graph is provided, and all ops will be added to this graph 
  if you do not create a new graph explicitly. Use this method with the with keyword to specify 
  that ops created within the scope of a block should be added to this graph.
  """

  with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False)

    # Get images and labels for CDHD.
    images, labels = cdhd.distorted_inputs()

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = cdhd.inference(images)

  # Calculate loss. 
    loss = cdhd.loss(logits, labels)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = cdhd.train(loss, global_step)      

    #example input: 163.3941|99.1765|car_ims/000002.jpg|219,460,3|1|0|1|1|48,24,441,202
    
    #global all_training_data
    CDHDGlobals.all_training_data = []

    for step in xrange(FLAGS.max_steps):
      start_time = time.time()
      _, loss_value = sess.run([train_op, loss])
      duration = time.time() - start_time

def main(argv=None):  # pylint: disable=unused-argument
  train()


if __name__ == '__main__':
  tf.app.run()    