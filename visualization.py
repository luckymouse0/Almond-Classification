# Tensorflow: visualize convolutional filters (conv1)
# https://gist.github.com/kukuruza/03731dc494603ceab0c5
#
# Authors:
# kukuruza(Original version)
# luckymouse0 (Modified version)

import tensorflow as tf
from math import sqrt


def conv1_filters():
    weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'conv1/kernel')[0]
    grid = _put_kernels_on_grid (weights, pad = 0)
    tf.summary.image('conv1/filters', grid, max_outputs=1)
    grid = _put_kernels_on_grid (weights)
    tf.summary.image('conv1/filters/separated', grid, max_outputs=1)

def act_histograms(*act):
    for i in range (len(act)):
        tf.summary.histogram("visualization/act"+str(i+1)+'/histogram', act[i])
    
def histograms():
    weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'conv1/kernel')[0]
    tf.summary.histogram('conv1/histogram', weights)    
    bias = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'conv1/bias')[0]
    tf.summary.histogram('conv1/bias/histogram', bias)
    
    weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'conv2/kernel')[0]
    tf.summary.histogram('conv2/histogram', weights)    
    bias = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'conv2/bias:0')
    tf.summary.histogram('conv2/bias/histogram', bias)
    
    weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'conv3/kernel')[0]
    tf.summary.histogram('conv3/histogram', weights)    
    bias = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'conv3/bias:0')
    tf.summary.histogram('conv3/bias/histogram', bias)
    
    weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'conv4/kernel')[0]
    tf.summary.histogram('conv4/histogram', weights)    
    bias = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'conv4/bias:0')
    tf.summary.histogram('conv4/bias/histogram', bias)
    
    weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'conv5/kernel')[0]
    tf.summary.histogram('conv5/histogram', weights)    
    bias = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'conv5/bias:0')
    tf.summary.histogram('conv5/bias/histogram', bias)
    
    weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'conv6/kernel')[0]
    tf.summary.histogram('conv6/histogram', weights)    
    bias = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'conv6/bias:0')
    tf.summary.histogram('conv6/bias/histogram', bias)
    
    weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'conv7/kernel')[0]
    tf.summary.histogram('conv7/histogram', weights)    
    bias = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'conv7/bias:0')
    tf.summary.histogram('conv7/bias/histogram', bias)
    
    weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'conv8/kernel')[0]
    tf.summary.histogram('conv8/histogram', weights)    
    bias = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'conv8/bias:0')
    tf.summary.histogram('conv8/bias/histogram', bias)
    
    weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'conv9/kernel')[0]
    tf.summary.histogram('conv9/histogram', weights)    
    bias = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'conv9/bias:0')
    tf.summary.histogram('conv9/bias/histogram', bias)
    
    weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'conv10/kernel')[0]
    tf.summary.histogram('conv10/histogram', weights)    
    bias = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'conv10/bias:0')
    tf.summary.histogram('conv10/bias/histogram', bias)
    
    weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'conv11/kernel')[0]
    tf.summary.histogram('conv11/histogram', weights)    
    bias = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'conv11/bias:0')
    tf.summary.histogram('conv11/bias/histogram', bias)
    
    weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'conv12/kernel')[0]
    tf.summary.histogram('conv12/histogram', weights)    
    bias = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'conv12/bias:0')
    tf.summary.histogram('conv12/bias/histogram', bias)
    
    weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'conv13/kernel')[0]
    tf.summary.histogram('conv13/histogram', weights)    
    bias = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'conv13/bias:0')
    tf.summary.histogram('conv13/bias/histogram', bias)
    

def make_summary(name, value, summary_writer, global_step):
    '''Create a summary from a name and value and add it to summary_writer
    Author: Jean Pouget-Abadie (https://stackoverflow.com/users/5094722/jean-pouget-abadie)
    '''    
    summary = tf.Summary()
    summary.value.add(tag=name, simple_value=value)
    summary_writer.add_summary(summary, global_step)
    

def _put_kernels_on_grid (kernel, pad = 1):

    '''Visualize conv. filters as an image (mostly for the 1st layer).
    Arranges filters into a grid, with some paddings between adjacent filters.
    Args:
    kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
    pad:               number of black pixels around each filter (between them)
    Return:
    Tensor of shape [1, (Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels].
    '''
  # get shape of the grid. NumKernels == grid_Y * grid_X
    def _factorization(n):
        for i in range(int(sqrt(float(n))), 0, -1):
            if n % i == 0:
                if i == 1: print('Who would enter a prime number of filters')
                return (i, int(n / i))
                
    (grid_Y, grid_X) = _factorization(kernel.get_shape()[3].value)
    #print ('grid: %d = (%d, %d)' % (kernel.get_shape()[3].value, grid_Y, grid_X))

    x_min = tf.reduce_min(kernel)
    x_max = tf.reduce_max(kernel)
    kernel = (kernel - x_min) / (x_max - x_min)

    # pad X and Y
    x = tf.pad(kernel, tf.constant( [[pad,pad],[pad, pad],[0,0],[0,0]] ), mode = 'CONSTANT')

    # X and Y dimensions, w.r.t. padding
    Y = kernel.get_shape()[0] + 2 * pad
    X = kernel.get_shape()[1] + 2 * pad

    channels = kernel.get_shape()[2]

    # put NumKernels to the 1st dimension
    x = tf.transpose(x, (3, 0, 1, 2))
    # organize grid on Y axis
    x = tf.reshape(x, tf.stack([grid_X, Y * grid_Y, X, channels]))

    # switch X and Y axes
    x = tf.transpose(x, (0, 2, 1, 3))
    # organize grid on X axis
    x = tf.reshape(x, tf.stack([1, X * grid_X, Y * grid_Y, channels]))

    # back to normal order (not combining with the next step for clarity)
    x = tf.transpose(x, (2, 1, 3, 0))

    # to tf.image_summary order [batch_size, height, width, channels],
    #   where in this case batch_size == 1
    x = tf.transpose(x, (3, 0, 1, 2))

    # scaling to [0, 255] is not necessary for tensorboard
    return x
