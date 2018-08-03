# Created on Wed May 31 14:48:46 2017
# https://kratzert.github.io/2017/06/15/example-of-tensorflows-new-input-pipeline.html
#
# Authors:
# Frederik Kratzert (Original version)
# luckymouse0 (Modified version)

"""Containes a helper class for image input pipelines in tensorflow."""

import tensorflow as tf
import numpy as np
import os

from tensorflow.python.data import Dataset
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor



class ImageDataGenerator(object):
    """Wrapper class around the new Tensorflows dataset pipeline.

    Requires Tensorflow >= version 1.12rc0
    """

    def __init__(self, dataset_path, batch_size, num_classes, channel=3, shuffle=True, buffer_size=1000, num_threads=4, img_size=None, data_aug=False, crop=None, resize=None):
        """Create a new ImageDataGenerator.

        Args:
            dataset_path: Path to dataset (train or test) directory.
            batch_size: Number of images per batch.
            num_classes: Number of classes in the dataset.
            shuffle: Whether or not to shuffle the data in the dataset and the initial file list.
            buffer_size: Number of images used as buffer for TensorFlows shuffling of the dataset. Should be > number of images per class.
            img_size: The size of the dataset's images, None means different sizes.
            data_aug: Whether or not to do random data augmentation.
            crop: Which crop to be used: "central", "random" or None.
            resize: Which resize to be used: "keep" to keep aspect ratio, "nokeep" to fit the image to the specific size (could get distorted)or None.

        """
        
        self.num_classes = num_classes
        
        self.img_size = img_size
        
        self.data_aug = data_aug
        
        self.channel = channel
        
        self.crop = crop
        
        self.resize = resize

        # retrieve the data from path
        self._read_dataset(dataset_path)

        # number of samples in the dataset
        self.data_size = len(self.indices)

        # initial shuffling of the file and label lists (together!)
        if shuffle:
            self._shuffle_lists()

        # convert lists to TF tensor
        self.img_paths = convert_to_tensor(self.img_paths, dtype=dtypes.string)
        self.indices = convert_to_tensor(self.indices, dtype=dtypes.int32)

        # create dataset
        dataset = Dataset.from_tensor_slices((self.img_paths, self.indices))

        # call the parsing functions
        data = dataset.map(self._parse_function, num_parallel_calls=num_threads).prefetch(100*batch_size)        

        # shuffle the first `buffer_size` elements of the dataset
        if shuffle:
            data = data.shuffle(buffer_size=buffer_size)

        # create a new dataset with batches of images        
        if(img_size is not None):
            data = data.batch(batch_size)   #same sizes
        else:
            data = data.padded_batch(batch_size, padded_shapes=([None,None,3],[None]))   #different sizes

        self.data = data

    def _read_dataset(self, dataset_path):
        """Read the content of the dataset path"""
        self.img_paths = []
        self.labels = []
        self.indices = []
        
        index = -1
        
        for directory in os.listdir(dataset_path):
            index = index + 1                
            self.labels.append(directory)         
            for image in os.listdir(dataset_path+"/"+directory):
                self.img_paths.append(dataset_path+"/"+directory+"/"+image)
                self.indices.append(index)

    def _shuffle_lists(self):
        """Conjoined shuffling of the list of paths and labels."""
        path = self.img_paths
        indices = self.indices
        permutation = np.random.permutation(self.data_size)
        self.img_paths = []
        self.indices = []
        for i in permutation:
            self.img_paths.append(path[i])
            self.indices.append(indices[i])
    
    def _parse_function(self, filename, label):
        """
        Input parser for samples of the training set.
        """        
        
        # convert label number into one-hot-encoding
        one_hot = tf.one_hot(label, self.num_classes)

        # load and preprocess the image
        img_string = tf.read_file(filename)
        img_decoded = tf.image.decode_png(img_string, channels=self.channel)   #JPG/PNG -> RGB
        
        # crop or resize the image to the given size
        if(self.img_size is not None):
            if(self.crop is not None):
                if(self.crop == "central"):
                    img_decoded = tf.image.resize_image_with_crop_or_pad(img_decoded, self.img_size, self.img_size) 
                elif(self.crop == "random"):
                    img_decoded = _random_crop_with_pad(img_decoded, self.img_size, self.channel)
                    #tf.random_crop(img_decoded, [self.img_size, self.img_size, self.channel])
            
            if(self.resize is not None):
                if(self.resize == "keep"):                
                    img_decoded = _resize_image_keep_aspect(img_decoded, self.img_size)
                elif(self.resize == "nokeep"):
                    img_decoded = tf.image.resize_images(img_decoded, [self.img_size, self.img_size])
        
        """
        Image manipulation operations
        """
        
        if(self.data_aug):
            img_decoded = tf.image.random_flip_left_right(img_decoded)   #random left-right flip
            img_decoded = tf.image.random_flip_up_down(img_decoded)   #random up-down flip
            
            img_decoded = tf.image.random_brightness(img_decoded, max_delta=0.2)   #random brightness, max_delta = [0,1)
            img_decoded = tf.image.random_contrast(img_decoded, lower=0.3, upper=1.0)   #random contrast, [lower, upper]
            
            #img_decoded = tf.image.random_hue(img_decoded, max_delta)   #random hue, max_delta = [0,0.5]
            #img_decoded = tf.image.random_saturation(img_decoded, lower, upper)   #random saturation, [lower, upper]
        
        
        # RGB -> BGR
        img_bgr = img_decoded[:, :, ::-1]

        return img_bgr, one_hot

def _random_crop_with_pad(image, size, channel):
    ''' Randomly crop images and pad if crop size is bigger
    based on: https://stackoverflow.com/questions/42147427/tensorflow-how-to-randomly-crop-input-images-and-labels-in-the-same-way
    Author: @Allen Lavoie (https://stackoverflow.com/users/6824418/allen-lavoie)
    '''
    image_shape = tf.shape(image)
    image_pad = tf.image.pad_to_bounding_box(image, 0, 0, tf.maximum(size, image_shape[0]), tf.maximum(size, image_shape[1]))
    image = tf.random_crop(image_pad, [size, size, channel])
    
    return image

def _resize_image_keep_aspect(image, size):
    '''Resize image keeping aspect ratio 
    https://github.com/tensorflow/tensorflow/issues/14213
    
    Authors: 
    Kayoku(Original version)
    luckymouse0 (Modified version)
    '''
    
    # Take width/height
    initial_width = tf.shape(image)[0]
    initial_height = tf.shape(image)[1]

    # Function for resizing 
    def _resize(x, y):
        # Take the greater value, and use it for the ratio 
        max_ = tf.maximum(initial_width, initial_height)
        ratio = tf.to_float(max_) / tf.constant(size, dtype=tf.float32)

        new_width = tf.to_float(initial_width) / ratio
        new_height = tf.to_float(initial_height) / ratio

        return tf.to_int32(new_width), tf.to_int32(new_height)

    # Useless function for the next condition
    def _useless(x, y):
        return x, y

    new_w, new_h = tf.cond(tf.logical_or(
        tf.greater(initial_width, tf.constant(size)),
        tf.greater(initial_height, tf.constant(size))
        ),
        lambda: _resize(initial_width, initial_height),
        lambda: _useless(initial_width, initial_height)
        )

    image = tf.image.resize_images(image, [new_w, new_h])
    
    # Add the zero-padding to the image to keep image size
    image = tf.image.pad_to_bounding_box(image, 0, 0, tf.maximum(size, new_w), tf.maximum(size, new_h))
    
    # float32 -> uint8
    image = tf.cast(image, tf.uint8)
    
    return image
    
    
    