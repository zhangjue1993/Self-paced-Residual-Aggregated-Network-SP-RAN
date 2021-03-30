
# -*- coding: UTF-8 -*-
import tensorflow as tf
from vgg import vgg16
import os
import numpy as np
import cv2
from utils import *
import os
import imageio
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:                   
        os.makedirs(path)  


image_dir = '/scratch/po21/jz1585/Google-ACT-256/train-400/'

layername = 'conv4_3'


image = tf.placeholder(tf.float32, [None, 256, 256, 3])
sess = tf.Session()
vgg = vgg16(image, "vgg16.npy", sess)

model_path = './model/'
kk = 0

with tf.Session() as sess:
    checkpoint = tf.train.get_checkpoint_state('./model')

    if checkpoint and checkpoint.model_checkpoint_path:
        print('Restoring model...')

        sess.run(tf.global_variables_initializer())
        saver.restore(sess, checkpoint.model_checkpoint_path)

        print('Restoration complete.')
        
        image1 = imageio.imread(os.path.join(image_dir))
        image = np.expand_dims(image1, axis=0)
        prob = sess.run(vgg.probs, feed_dict={vgg.imgs: image})[0]
        preds = (np.argsort(prob)[::-1])[0:2]
        print('\n *******prob:', prob)