from collections import deque
import tensorflow as tf
import os
import numpy as np
import cv2
from utils import *
import pickle
import os
from model import unet
from loss import *
from generate_data_list import generate_data_list
import time
from datetime import datetime
import imageio

import shutil
#slim = tensorflow.contrib.slim
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('ckpt_restore_dir', '/home/549/jz1585/0909/2020-09-11-11-06-01/model','ckpt_restore_dir')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

batch_size = 50


IMAGE_SIZE = 256


#generate_data_list(TEST_SET_DIR_solar,TEST_SET_DIR_label)

localtime=time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
result_save_dir = 'results'+localtime

if not os.path.exists(result_save_dir):      
    os.makedirs(result_save_dir) 
  


#save_config(ckpt_save_dir,'config.txt',training_config)

try:
    with open('test_set_list.pickle', 'rb') as f:
            test_set_list = pickle.load(f)
except:
    raise EnvironmentError('Data list not existed. Please run generate_data_list.py first.')
#random.shuffle(test_set_list)
test_set_queue = deque(test_set_list)
test_set_size = len(test_set_list)

print ('Testing set built. Size: '+str(test_set_size))

#test_label_list, test_image_list = [],[]
#for d in test_set_list:
#    test_image_list.append(load_image_test(d[0],IMAGE_SIZE))
#    test_label_list.append(convert_label(d[1], threshold=0, IMAGE_SIZE = IMAGE_SIZE))
#test_label_list, test_image_list = np.array(test_label_list), np.array(test_image_list)

del(test_set_list)


with tf.Graph().as_default():
    image = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, 3])


    Unet = unet(image, block_num = 4)


    config = tf.ConfigProto(allow_soft_placement=True)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    config.gpu_options.allow_growth = True


    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(tf.global_variables(),max_to_keep = 100)
        # restore old checkpoint
        checkpoint = tf.train.get_checkpoint_state(FLAGS.ckpt_restore_dir)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
                print("Could not find old network weights")

        
        batch_num = test_set_size//batch_size
        print(batch_num)

        for j in range(0,batch_num):

            minibatch = []
            for count in range(0, batch_size):
                element = test_set_queue.pop()
                minibatch.append(element)
                #print(minibatch)

            image_list = [load_image(d[0],IMAGE_SIZE) for d in minibatch]
            
            image_batch = np.array(image_list)

            output = sess.run(Unet.finalpre,  feed_dict = {image: image_batch})
            output = np.array(output)
            #print(output.shape)            
            for i in range(0,len(output)):
                print(os.path.basename(minibatch[i][0]))
                imageio.imwrite(os.path.join(result_save_dir,os.path.basename(minibatch[i][0])),output[i,:,:,0]*255)
                #misc.imsave(os.path.join(result_save_dir,'ori'+os.path.basename(minibatch[i][0])),image_list[i])