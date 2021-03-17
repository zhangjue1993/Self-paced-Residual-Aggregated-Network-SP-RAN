import tensorflow as tf
import numpy as np
from keras.layers import Conv2D,Activation,concatenate
from collections import OrderedDict
from keras import backend as K

from keras.models import Sequential, Model
from keras.models import model_from_json
from keras.layers import Input
from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Add, Lambda
from keras.layers import Dropout, BatchNormalization, concatenate
from keras.layers import Softmax

from keras.initializers import RandomNormal
from keras.models import Model
# from imagenet_classes import class_names


class unet:
    def __init__(self, imgs,block_num=4):
        self.imgs = imgs
        self.block_num = block_num
        self.prediction = []
        self.finalpre = []
        self.layers={}
        self.convlayers()
        

    def convlayers(self):
        self.layers = {}
        down_features = [64, 128,256,512]
        up_features = [64,128,256,512,1024]

        std_dev = 5e-3
        # down_path
        with tf.name_scope('path_down') as scope:
            pattern = 'same'
            down_conv = OrderedDict()
            innode =  self.imgs
            self.layers['input'] = innode
            for layer in range(0,self.block_num):
                with tf.variable_scope('down_block_'+str(layer+1)):
                    conv = Conv2D(filters=down_features[layer], kernel_size=3, strides=(1, 1), padding=pattern,
                        activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(innode)
                    print("down block {layer} : CONV-1 shape is {shape}".format(layer=layer, shape=conv.shape))
                    conv = Conv2D(filters=down_features[layer], kernel_size=3, strides=(1, 1), padding=pattern,
                        activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(conv)
                    print("down block {layer} : CONV-2 shape is {shape}".format(layer=layer, shape=conv.shape))
                    down_conv[str(layer)] = conv
                    pool = MaxPooling2D(pool_size=(2,2))(conv)
                    print("down block {layer} : POOL shape is {shape}".format(layer=layer, shape=pool.shape))
                    innode = pool
                    self.layers['down_block_'+str(layer+1)] = innode
            
        with tf.name_scope('bottom') as scope:
            with tf.variable_scope('bottom_conv'):
                conv = Conv2D(filters=up_features[self.block_num], kernel_size=3, strides=(1, 1), padding=pattern,
                        activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(innode)
                print("bottom {layer} : CONV-1 shape is {shape}".format(layer=layer, shape=conv.shape))
                conv = Conv2D(filters=up_features[self.block_num], kernel_size=3, strides=(1, 1), padding=pattern,
                        activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(conv)
                print("bottom {layer} : CONV-2 shape is {shape}".format(layer=layer, shape=conv.shape))
                innode = conv
                self.layers['bottom'] = innode

        with tf.name_scope('path_up') as scope:
            for layer in range(self.block_num-1,-1,-1):
                with tf.variable_scope('up_block_'+str(layer)):
                    conv = Conv2DTranspose(filters=up_features[layer], kernel_size=3, strides=(2, 2), padding=pattern,
                        activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(innode)
                    conv = concatenate([down_conv[str(layer)],conv],axis=3)
                    conv = Conv2D(filters=up_features[layer], kernel_size=3, strides=(1, 1), padding=pattern,
                            activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(conv)
                    print("up block {layer} : CONV-1 shape is {shape}".format(layer=layer, shape=conv.shape))
                    conv = Conv2D(filters=up_features[layer], kernel_size=3, strides=(1, 1), padding=pattern,
                            activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(conv)
                    print("up block {layer} : CONV-2 shape is {shape}".format(layer=layer, shape=conv.shape))
                    innode = conv
                    self.layers['up_block_'+str(layer)] = innode

            with tf.variable_scope('up_block_'+str(layer-1)):
                output = Conv2D(filters=2, kernel_size=1, strides=(1, 1), padding=pattern,
                            activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(innode)
                print("up block {layer} :  shape is {shape}".format(layer=layer, shape=output.shape))

        self.layers['output'+str(layer)] = output
        self.prediction = output
        print("prediction :  shape is {shape}".format(shape=self.prediction.shape))
        self.finalpre = tf.nn.softmax(output)
        print("finalpre :  shape is {shape}".format(shape=self.finalpre.shape))

