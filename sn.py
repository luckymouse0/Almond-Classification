import os
import numpy as np
import tensorflow as tf

from datagenerator import ImageDataGenerator
import visualization

from datetime import datetime
from tensorflow.python.data import Iterator


# CONSTANTS

LEARNING_RATE = 0.001  #values between 0.001 and 0.00001
WEIGHT_DECAY = 0.005  #L2 regularization 

BATCH_SIZE = 32  #depends on the GPU memory
EPOCH = 1000  #0 for testing
VALIDATION_EPOCH = 5  #number of epochs until validation (0 for no validation)

IMAGE_SIZE = 128  #IMAGE_SIZE x IMAGE_SIZE, None = Different sizes
CHANNEL = 3  #3 = RGB, 1 = grayscale

NUM_CLASSES = 2

SAVE_EPOCH = 10  #number of epochs until saving a checkpoint
PRINT_EPOCH = 5  #number of epochs until printing accuracy

MODEL_NAME = "simplenet-aug-128-crop-random"
TRAIN_PATH = "dataset/train"
VALIDATION_PATH = "dataset/val"
TEST_PATH = "dataset/test"
LOGS_PATH = "./logs/" + MODEL_NAME
SAVE_PATH = "./checkpoints/" + MODEL_NAME

AUG = True
RESIZE = None
CROP = "random"

TEST = "keep"


# GLOBAL VARIABLES

# x - input image
x = tf.placeholder("float32", shape=[None, IMAGE_SIZE, IMAGE_SIZE, CHANNEL], name="x")

# y correct image class, using one-hot encoding
y_true = tf.placeholder("float32", shape=[None, NUM_CLASSES], name="y_true")


def CreateGraph():    
    """ 
    Implementation of SimpleNet, based on the paper [arXiv:1608.06037]:    
    "Lets keep it simple, Using simple architectures to outperform deeper and more complex architectures"
    
    Seyyed Hossein Hasanpour'1, Mohammad Rouhani'2, Mohsen Fayyaz'3, Mohammad Sabokrou'4
    1 Islamic Azad University, Science and Research branch, Iran
    2 Computer Vision Researcher, Technicolor R&I, Rennes, France
    3 Sensifai, Belgium
    4 Institute for Research in Fundamental Sciences (IPM), Iran
    """
    
    # Network Parameters
    
    if(EPOCH == 0):  #dropout at training time only
        conv_dropout = 0
    else:
        conv_dropout = 0.2  #between 0.1 and 0.3, batch normalization reduces the needs of dropout
    
    convStrides = 1   #stride 1 allows us to leave all spatial down-sampling to the POOL layers
    poolStrides = 2
    
    convKernelSize = 3
    convKernelSize1 = 1
    poolKernelSize = 2
    
    filterSize1 = 64
    filterSize = 128
    
    bn_decay = 0.95
    
    # Network Architecture
    
    conv1 = tf.layers.conv2d(x, filterSize1, kernel_size=[convKernelSize, convKernelSize], strides=[convStrides, convStrides], padding="SAME", kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.constant_initializer(0.1), kernel_regularizer=tf.contrib.layers.l2_regularizer(WEIGHT_DECAY), name='conv1')   
    bn1 = tf.contrib.layers.batch_norm(conv1, epsilon=0.001, decay = bn_decay, updates_collections=None)
    act1 = tf.nn.leaky_relu(bn1, alpha=0.1, name='act1')
    drop1 = tf.layers.dropout(act1, rate=conv_dropout, training=True)
    
    conv2 = tf.layers.conv2d(drop1, filterSize, kernel_size=[convKernelSize, convKernelSize], strides=[convStrides, convStrides], padding="SAME", kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.constant_initializer(0.1), kernel_regularizer=tf.contrib.layers.l2_regularizer(WEIGHT_DECAY), name='conv2')   
    bn2 = tf.contrib.layers.batch_norm(conv2, epsilon=0.001, decay = bn_decay, updates_collections=None)
    pool2 = tf.nn.max_pool(bn2, ksize = [1, poolKernelSize, poolKernelSize, 1], strides = [1, poolStrides, poolStrides, 1], padding = "SAME")    
    act2 = tf.nn.leaky_relu(pool2, alpha=0.1, name='act2')
    drop2 = tf.layers.dropout(act2, rate=conv_dropout, training=True)
    
    conv3 = tf.layers.conv2d(drop2, filterSize, kernel_size=[convKernelSize, convKernelSize], strides=[convStrides, convStrides], padding="SAME", kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.constant_initializer(0.1), kernel_regularizer=tf.contrib.layers.l2_regularizer(WEIGHT_DECAY), name='conv3')   
    bn3 = tf.contrib.layers.batch_norm(conv3, epsilon=0.001, decay = bn_decay, updates_collections=None)
    pool3 = tf.nn.max_pool(bn3, ksize = [1, poolKernelSize, poolKernelSize, 1], strides = [1, poolStrides, poolStrides, 1], padding = "SAME")    
    act3 = tf.nn.leaky_relu(pool3, alpha=0.1, name='act3')
    drop3 = tf.layers.dropout(act3, rate=conv_dropout, training=True)
    
    conv4 = tf.layers.conv2d(drop3, filterSize, kernel_size=[convKernelSize, convKernelSize], strides=[convStrides, convStrides], padding="SAME", kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.constant_initializer(0.1), kernel_regularizer=tf.contrib.layers.l2_regularizer(WEIGHT_DECAY), name='conv4')   
    bn4 = tf.contrib.layers.batch_norm(conv4, epsilon=0.001, decay = bn_decay, updates_collections=None)
    pool4 = tf.nn.max_pool(bn4, ksize = [1, poolKernelSize, poolKernelSize, 1], strides = [1, poolStrides, poolStrides, 1], padding = "SAME")    
    act4 = tf.nn.leaky_relu(pool4, alpha=0.1, name='act4')
    drop4 = tf.layers.dropout(act4, rate=conv_dropout, training=True)    

    conv5 = tf.layers.conv2d(drop4, filterSize, kernel_size=[convKernelSize, convKernelSize], strides=[convStrides, convStrides], padding="SAME", kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.constant_initializer(0.1), kernel_regularizer=tf.contrib.layers.l2_regularizer(WEIGHT_DECAY), name='conv5')   
    bn5 = tf.contrib.layers.batch_norm(conv5, epsilon=0.001, decay = bn_decay, updates_collections=None)
    act5 = tf.nn.leaky_relu(bn5, alpha=0.1, name='act5')
    drop5 = tf.layers.dropout(act5, rate=conv_dropout, training=True)
    
    conv6 = tf.layers.conv2d(drop5, filterSize, kernel_size=[convKernelSize, convKernelSize], strides=[convStrides, convStrides], padding="SAME", kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.constant_initializer(0.1), kernel_regularizer=tf.contrib.layers.l2_regularizer(WEIGHT_DECAY), name='conv6')   
    bn6 = tf.contrib.layers.batch_norm(conv6, epsilon=0.001, decay = bn_decay, updates_collections=None)
    act6 = tf.nn.leaky_relu(bn6, alpha=0.1, name='act6')
    drop6 = tf.layers.dropout(act6, rate=conv_dropout, training=True)
    
    conv7 = tf.layers.conv2d(drop6, filterSize, kernel_size=[convKernelSize, convKernelSize], strides=[convStrides, convStrides], padding="SAME", kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.constant_initializer(0.1), kernel_regularizer=tf.contrib.layers.l2_regularizer(WEIGHT_DECAY), name='conv7')   
    bn7 = tf.contrib.layers.batch_norm(conv7, epsilon=0.001, decay = bn_decay, updates_collections=None)
    pool7 = tf.nn.max_pool(bn7, ksize = [1, poolKernelSize, poolKernelSize, 1], strides = [1, poolStrides, poolStrides, 1], padding = "SAME")    
    act7 = tf.nn.leaky_relu(pool7, alpha=0.1, name='act7')
    drop7 = tf.layers.dropout(act7, rate=conv_dropout, training=True)
    
    conv8 = tf.layers.conv2d(drop7, filterSize, kernel_size=[convKernelSize, convKernelSize], strides=[convStrides, convStrides], padding="SAME", kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.constant_initializer(0.1), kernel_regularizer=tf.contrib.layers.l2_regularizer(WEIGHT_DECAY), name='conv8')   
    bn8 = tf.contrib.layers.batch_norm(conv8, epsilon=0.001, decay = bn_decay, updates_collections=None)
    pool8 = tf.nn.max_pool(bn8, ksize = [1, poolKernelSize, poolKernelSize, 1], strides = [1, poolStrides, poolStrides, 1], padding = "SAME")    
    act8 = tf.nn.leaky_relu(pool8, alpha=0.1, name='act8')
    drop8 = tf.layers.dropout(act8, rate=conv_dropout, training=True)
    
    conv9 = tf.layers.conv2d(drop8, filterSize, kernel_size=[convKernelSize, convKernelSize], strides=[convStrides, convStrides], padding="SAME", kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.constant_initializer(0.1), kernel_regularizer=tf.contrib.layers.l2_regularizer(WEIGHT_DECAY), name='conv9')   
    bn9 = tf.contrib.layers.batch_norm(conv9, epsilon=0.001, decay = bn_decay, updates_collections=None)
    pool9 = tf.nn.max_pool(bn9, ksize = [1, poolKernelSize, poolKernelSize, 1], strides = [1, poolStrides, poolStrides, 1], padding = "SAME")    
    act9 = tf.nn.leaky_relu(pool9, alpha=0.1, name='act9')
    drop9 = tf.layers.dropout(act9, rate=conv_dropout, training=True)    
    
    conv10 = tf.layers.conv2d(drop9, filterSize, kernel_size=[convKernelSize, convKernelSize], strides=[convStrides, convStrides], padding="SAME", kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.constant_initializer(0.1), kernel_regularizer=tf.contrib.layers.l2_regularizer(WEIGHT_DECAY), name='conv10')   
    bn10 = tf.contrib.layers.batch_norm(conv10, epsilon=0.001, decay = bn_decay, updates_collections=None)
    act10 = tf.nn.leaky_relu(bn10, alpha=0.1, name='act10')
    drop10 = tf.layers.dropout(act10, rate=conv_dropout, training=True)    
    
    conv11 = tf.layers.conv2d(drop10, filterSize, kernel_size=[convKernelSize1, convKernelSize1], strides=[convStrides, convStrides], padding="SAME", kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.constant_initializer(0.1), kernel_regularizer=tf.contrib.layers.l2_regularizer(WEIGHT_DECAY), name='conv11')   
    bn11 = tf.contrib.layers.batch_norm(conv11, epsilon=0.001, decay = bn_decay, updates_collections=None)
    act11 = tf.nn.leaky_relu(bn11, alpha=0.1, name='act11')
    drop11 = tf.layers.dropout(act11, rate=conv_dropout, training=True)
    
    conv12 = tf.layers.conv2d(drop11, filterSize, kernel_size=[convKernelSize1, convKernelSize1], strides=[convStrides, convStrides], padding="SAME", kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.constant_initializer(0.1), kernel_regularizer=tf.contrib.layers.l2_regularizer(WEIGHT_DECAY), name='conv12')   
    bn12 = tf.contrib.layers.batch_norm(conv12, epsilon=0.001, decay = bn_decay, updates_collections=None)
    act12 = tf.nn.leaky_relu(bn12, alpha=0.1, name='act12')
    drop12 = tf.layers.dropout(act12, rate=conv_dropout, training=True)
    
    conv13 = tf.layers.conv2d(drop12, NUM_CLASSES, kernel_size=[convKernelSize, convKernelSize], strides=[convStrides, convStrides], padding="SAME", kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.constant_initializer(0.1), kernel_regularizer=tf.contrib.layers.l2_regularizer(WEIGHT_DECAY), name='conv13')   
    pool13 = tf.nn.max_pool(conv13, ksize = [1, poolKernelSize, poolKernelSize, 1], strides = [1, poolStrides, poolStrides, 1], padding = "SAME")
    drop13 = tf.layers.dropout(pool13, rate=conv_dropout, training=True)
    
    avg13 = tf.keras.layers.GlobalAveragePooling2D(name='avg13')(drop13)        
    
    # Tensorboard Visualization
    visualization.act_histograms(act1, act2, act3, act4, act5, act6, act7, act8, act9, act10, act11, act12)
    
    return avg13

    
def TrainGraph(output):
    
    global_step = tf.Variable(0, name="global_step")
    
    y_true_cls = tf.argmax(y_true, axis=1)   #returns the true class index
    
    # y predicted image class
    y_pred = tf.nn.softmax(output, name="y_pred")
    y_pred_cls = tf.argmax(y_pred, axis=1, name="y_pred_cls")    #returns the predicted index
    
    # Cost function
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=tf.stop_gradient(y_true))
    cost = tf.reduce_mean(cross_entropy)
    
    # Optimizer
    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost, global_step=global_step)
    
    # Accuracy
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)    
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float32"), name="accuracy")
    
    # Creating the session to run the graph
    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    #config.gpu_options.allocator_type = 'BFC'
    #config.gpu_options.per_process_gpu_memory_fraction=0.90
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    
    # Saver
    saver = tf.train.Saver(max_to_keep=1)    
    if(not os.path.exists(SAVE_PATH)): #if directory doesn't exists, then create it
        os.makedirs(SAVE_PATH)
    checkpoint = tf.train.latest_checkpoint(SAVE_PATH) #restoring last checkpoint
    if(checkpoint != None):
        print("Restoring checkpoint %s" %(checkpoint))
        saver.restore(sess, checkpoint)
        print("Model restored")
    else:
        sess.run(tf.global_variables_initializer())
        print("Initialized a new Graph")
    print("")
    
    # Tensorboard
    
    # Visualize conv1 filters and histograms
    with tf.variable_scope('visualization'):
        tf.get_variable_scope().reuse_variables()        
        visualization.conv1_filters()
        visualization.histograms()
    
    # Create a summary to monitor cost tensor
    tf.summary.scalar("loss/step", cost)
    # Create a summary to monitor accuracy tensor
    tf.summary.scalar("accuracy/step", accuracy)
    # Merge all summaries into a single op
    merged_summary_op = tf.summary.merge_all()
    
    # Loading training set
    
    if(not os.path.exists(TRAIN_PATH)):
        print("Error reading training directory")
        return
    
    # Reading the dataset and creating training data class
    with tf.device('/cpu:0'):
        tr_data = ImageDataGenerator(TRAIN_PATH, batch_size=BATCH_SIZE, num_classes=NUM_CLASSES, channel=CHANNEL, shuffle=True, data_aug=AUG, img_size=IMAGE_SIZE, crop=CROP, resize=RESIZE)

        # Create an reinitializable iterator given the dataset structure
        iterator = Iterator.from_structure(tr_data.data.output_types, tr_data.data.output_shapes)        
        next_batch = iterator.get_next()

    # Ops for initializing the training iterator
    training_init_op = iterator.make_initializer(tr_data.data)
    
    # Get the number of training steps per epoch
    train_batches_per_epoch = int(np.floor(tr_data.data_size/BATCH_SIZE))

    # Loading validation set (if used)
    
    if(VALIDATION_EPOCH > 0):        
        # Reading the dataset and creating validation data class
        
        if(not os.path.exists(VALIDATION_PATH)):
            print("Error reading validation directory")
            return
            
        with tf.device('/cpu:0'):
            val_data = ImageDataGenerator(VALIDATION_PATH, batch_size=BATCH_SIZE, num_classes=NUM_CLASSES, channel=CHANNEL, shuffle=False, data_aug=False, img_size=IMAGE_SIZE, crop=CROP, resize=RESIZE)

            # Create an reinitializable iterator given the dataset structure
            val_iterator = Iterator.from_structure(val_data.data.output_types, val_data.data.output_shapes)
            val_next_batch = val_iterator.get_next()

        # Ops for initializing the validation iterator
        validation_init_op = val_iterator.make_initializer(val_data.data)
        
        # Get the number of training steps per epoch
        val_batches_per_epoch = int(np.floor(val_data.data_size/BATCH_SIZE))
    
    
    # Print sizes
    print("")
    print("Dataset Size:")
    print("Training data: " + str(tr_data.data_size))
    
    if(VALIDATION_EPOCH > 0):
        print("Validation data: " + str(val_data.data_size)) 
        
    print("")
    
    # Write logs to Tensorboard
    if(not os.path.exists(LOGS_PATH)): #if directories don't exist, then create them
        os.makedirs(LOGS_PATH)
        
        if(not os.path.exists(LOGS_PATH + "/train")):
            os.makedirs(LOGS_PATH + "/train")
            
        if(VALIDATION_EPOCH > 0):
            if(not os.path.exists(LOGS_PATH + "/val")):
                os.makedirs(LOGS_PATH + "/val")
                
    train_summary_writer = tf.summary.FileWriter(LOGS_PATH + "/train", graph=tf.get_default_graph())
    
    if(VALIDATION_EPOCH > 0):
        val_summary_writer = tf.summary.FileWriter(LOGS_PATH + "/val")

    # Start training
    for i in range(EPOCH):
        print("{} Epoch number: {}".format(datetime.now().strftime('%d-%m-%Y %H:%M:%S'), i+1))
        
        # Initialize iterator with the training dataset
        sess.run(training_init_op)
            
        train_acc = 0.
        train_loss = 0.

        for step in range(train_batches_per_epoch):

            # Get next batch of data
            img_batch, label_batch = sess.run(next_batch)

            # And run the training op
            i_global, _, summary, batch_acc, batch_loss = sess.run([global_step, train_step, merged_summary_op, accuracy, cost], feed_dict={x: img_batch, y_true: label_batch})
        
            # Write Tensorboard logs at every iteration
            train_summary_writer.add_summary(summary, i_global)            
            
            train_acc += batch_acc
            train_loss += batch_loss
            
        train_acc /= train_batches_per_epoch
        train_loss /= train_batches_per_epoch      
        
        # Do validation after VALIDATION_EPOCH epochs
        if (VALIDATION_EPOCH > 0 and (i+1) % VALIDATION_EPOCH == 0):
            # Initialize iterator with the validation dataset
            sess.run(validation_init_op)
            
            val_acc = 0.
            val_loss = 0.
            
            for step in range(val_batches_per_epoch):
                
                # Get next batch of data
                val_img_batch, val_label_batch = sess.run(val_next_batch)

                # And run the training op
                i_global, _, summary, batch_acc, batch_loss = sess.run([global_step, train_step, merged_summary_op, accuracy, cost], feed_dict={x: val_img_batch, y_true: val_label_batch})
            
                # Write Tensorboard logs at every iteration
                val_summary_writer.add_summary(summary, i_global)
                
                val_acc += batch_acc
                val_loss += batch_loss
            
            val_acc /= val_batches_per_epoch
            val_loss /= val_batches_per_epoch
            
        # Print status to screen and tensorboard every PRINT_EPOCH epochs (and the last).
        if ((i+1) % PRINT_EPOCH == 0) or ((i+1) == EPOCH):
            # Print status.
            print("{0} Global Step: {1}, Training Accuracy: {2:>6.1%}".format(datetime.now().strftime('%d-%m-%Y %H:%M:%S'), i_global, train_acc))
            visualization.make_summary('accuracy/average', train_acc, train_summary_writer, i_global)
            visualization.make_summary('loss/average', train_loss, train_summary_writer, i_global)
            
            if (VALIDATION_EPOCH > 0):
                # Print status.
                print("{0} Global Step: {1}, Validation Accuracy: {2:>6.1%}".format(datetime.now().strftime('%d-%m-%Y %H:%M:%S'), i_global, val_acc))
                visualization.make_summary('accuracy/average', val_acc, val_summary_writer, i_global)
                visualization.make_summary('loss/average', val_loss, val_summary_writer, i_global)
            
            
        # Save a checkpoint to disk every SAVE_EPOCH epochs (and the last).
        if ((i+1) % SAVE_EPOCH == 0) or ((i+1) == EPOCH):
            # Save all variables of the TensorFlow graph to a checkpoint.
            saver.save(sess, SAVE_PATH + "/" + MODEL_NAME + ".ckpt", global_step=global_step)
            print("{0} Checkpoint saved.".format(datetime.now().strftime('%d-%m-%Y %H:%M:%S')))

            
def TestGraph():
    # Loading graph
    sess = tf.Session()
    saver = tf.train.Saver(max_to_keep=1) 
    checkpoint = tf.train.latest_checkpoint(SAVE_PATH)
    if(checkpoint != None):
        print("Restoring checkpoint %s" %(checkpoint))
        saver.restore(sess, checkpoint)
        print("Model restored")
    else:
        print("Model not found")
        return
    print("")
    
    
    # Loading test set
    
    if(not os.path.exists(TEST_PATH)):
        print("Error reading testing directory.")
        return
    
    # Reading the dataset and creating testing data class
    with tf.device('/cpu:0'):
        test_data = ImageDataGenerator(TEST_PATH, batch_size=1, num_classes=NUM_CLASSES, channel=CHANNEL, shuffle=False, data_aug=False, img_size=IMAGE_SIZE, resize=TEST)

        # Create an reinitializable iterator given the dataset structure
        iterator = Iterator.from_structure(test_data.data.output_types, test_data.data.output_shapes)
        next_batch = iterator.get_next()

    # Ops for initializing the testing iterator
    testing_init_op = iterator.make_initializer(test_data.data)
    
    print("")
    print("Dataset Size:")
    print("Test data: " + str(test_data.data_size))
    print("")
        
    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    
    # Test the model on the entire testing set
    print("{} Start testing".format(datetime.now().strftime('%d-%m-%Y %H:%M:%S')))
    
    sess.run(testing_init_op)
    
    test_acc = 0.
    test_count = 0
    
    for _ in range(test_data.data_size):    
        img_batch, label_batch = sess.run(next_batch)
        batch_acc = sess.run('accuracy:0', feed_dict={x: img_batch, y_true: label_batch})
        
        test_acc += batch_acc
        test_count += 1
    
    test_acc /= test_count
    
    print("{} Testing Accuracy = {:>6.1%}".format(datetime.now().strftime('%d-%m-%Y %H:%M:%S'), test_acc))    
    

def main():
    output = CreateGraph()
    
    TrainGraph(output)
    
    if(EPOCH == 0):
        TestGraph()
        
if __name__ == "__main__":
    main()
    


