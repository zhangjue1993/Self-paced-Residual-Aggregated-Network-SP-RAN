from collections import deque
import tensorflow as tf
import os
import numpy as np
import cv2
from utils import *
import pickle
import os
from model import *
from loss import *
from generate_data_list import generate_data_list
import time
from datetime import datetime
import imageio
import argparse
import shutil
from vgg import *
#slim = tensorflow.contrib.slim

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", help="filename of train_set",default='/scratch/po21/jz1585/1105/2020-11-09-14-30-49/model/')

parser.add_argument("--data_dir", help="filename of train_set",default='/scratch/po21/jz1585/')
parser.add_argument("--train_set", help="filename of train_set",default='none')
parser.add_argument("--train_set_label", help="filename of train_set_label", default="none")
parser.add_argument("--test_set", help="filename of train_set",default="/Google-Bris/test/")
parser.add_argument("--test_set_label", help="filename of train_set_label", default="/Google-Bris/test-gt/")

parser.add_argument("--vgg_checkpoint", default='/scratch/po21/jz1585/VGG/model-Bris/', help="directory with checkpoints")


parser.add_argument("--model", default='FPN_RFA_3', help="model")
parser.add_argument("--block", type=int,default=4, help="block")
Config=parser.parse_args()



os.environ["CUDA_VISIBLE_DEVICES"] = "0"

batch_size = 27
IMAGE_SIZE = 256

print()

#generate_data_list(TEST_SET_DIR_solar,TEST_SET_DIR_label)

localtime=time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
result_save_dir = 'results'+localtime

if not os.path.exists(result_save_dir):      
    os.makedirs(result_save_dir) 
  
TRAIN_SET_DIR_solar =  Config.data_dir + Config.train_set
TRAIN_SET_DIR_label =  Config.data_dir + Config.train_set_label
TEST_SET_DIR_solar =  Config.data_dir + Config.test_set
TEST_SET_DIR_label =  Config.data_dir + Config.test_set_label
print(TRAIN_SET_DIR_solar,TRAIN_SET_DIR_label,TEST_SET_DIR_solar,TEST_SET_DIR_label)

generate_data_list(TRAIN_SET_DIR_solar,TRAIN_SET_DIR_label,TEST_SET_DIR_solar,TEST_SET_DIR_label)


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
    image = tf.placeholder(tf.float32, [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3])

    vgg = vgg16(image*255)

    if Config.model =='FPN':
        model = FPN(image, block_num = Config.block)
    elif Config.model =='Unet':
        model = unet(image, block_num = Config.block)
    elif Config.model =='FPN_RFA_3':
        model = FPN_RFA_3(image, score = vgg.probs[:,1],block_num = Config.block)
    elif Config.model =='FPN_RFA_BS':
        model = FPN_RFA_BS(image, score = vgg.probs[:,1],block_num = Config.block)
    elif Config.model =='FPN_RFA_BS_2':
        model = FPN_RFA_BS_2(image, score = vgg.probs[:,1],block_num = Config.block)
    elif Config.model =='FPN_RFA_BS_3':
        model = FPN_RFA_BS_3(image, score = vgg.probs[:,1],block_num = Config.block)
    elif Config.model =='FPN_RFA_BS_4':
        model = FPN_RFA_BS_4(image, score = vgg.probs[:,1],block_num = Config.block)


    config = tf.ConfigProto(allow_soft_placement=True)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    config.gpu_options.allow_growth = True


    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

    #saver = tf.train.Saver(max_to_keep=100)
        variables = tf.trainable_variables()
        #print(variables)
        variables_model = [v for v in variables if v.name.split('/')[0]=='FPN_RFA_BS']
        saver = tf.train.Saver(variables_model)
        # restore old checkpoint
        checkpoint = tf.train.get_checkpoint_state(Config.checkpoint)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")
        
        #VGG
        variables_vgg = [v for v in variables if v.name.split('/')[0]!='FPN_RFA_BS']
        saver2 = tf.train.Saver(variables_vgg)
        # restore old checkpoint
        checkpoint = tf.train.get_checkpoint_state(Config.vgg_checkpoint)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver2.restore(sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded VGG:", checkpoint.model_checkpoint_path)
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

            image_list = [load_image_test(d[0],IMAGE_SIZE) for d in minibatch]
            
            image_batch = np.array(image_list)

            output,prob = sess.run([model.finalpre,vgg.probs[:,1]], feed_dict = {image: image_batch})
            output = np.array(output)
            for i in range(0,len(output)):
                print(os.path.basename(minibatch[i][0]),prob[i])
                imageio.imwrite(os.path.join(result_save_dir,os.path.basename(minibatch[i][0])),output[i,:,:,0]*255)
                #misc.imsave(os.path.join(result_save_dir,'ori'+os.path.basename(minibatch[i][0])),image_list[i])