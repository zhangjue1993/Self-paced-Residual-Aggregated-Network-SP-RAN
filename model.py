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
std_dev = 5e-3
pattern = 'same'


def residual_block(innode, namescope):
    print(innode)
    _,_,_,C = innode.get_shape().as_list()
    with tf.name_scope(namescope) as scope:

        conv = Conv2D(filters=C, kernel_size=3, strides=(1, 1), padding=pattern,
                                activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(innode)

        conv = Conv2D(filters=C, kernel_size=3, strides=(1, 1), padding=pattern,
                                activation=None, kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(conv)

        output = conv+innode
        return output,conv

def RFA_block(innode, channel, namescope,num=2):

    with tf.name_scope(namescope) as scope:
        conv = innode
        newre = []
        for i in range(0,num):
            output,temp = residual_block(conv, namescope+'_RB_'+str(i+1))
            conv = output
            if i==0:
                newre = temp
            else:
                newre = concatenate([newre,temp],axis=3)
        newconv = Conv2D(filters=channel, kernel_size=3, strides=(1, 1), padding=pattern,
                                activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(conv)
        newre = concatenate([newre,newconv],axis=3)
        convre = Conv2D(filters=channel, kernel_size=3, strides=(1, 1), padding=pattern,
                                activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(newre)

    return convre

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

        pattern = 'same'
        std_dev = 5e-3
        # down_path
        with tf.name_scope('FPN_RFA_BS'):
            with tf.name_scope('path_down') as scope:
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

class FPN:
    def __init__(self, imgs,block_num=4):
        self.imgs = imgs
        self.block_num = block_num
        self.prediction = []
        self.finalpre = []
        self.layers={}
        self.convlayers()    

    def convlayers(self):

        down_features = [64, 128,256,512,1024]
        up_features = 256
        std_dev = 5e-3
        with tf.name_scope('FPN_RFA_BS'):
            with tf.name_scope('path_down') as scope:
                pattern = 'same'
                down_conv = OrderedDict()
                innode =  self.imgs
                self.layers['input'] = innode

                block = 0
                innode = self.imgs

                # Two conv : input B*H*W*3 output B*H*W*64
                conv = Conv2D(filters=down_features[block], kernel_size=3, strides=(1, 1), padding=pattern,
                                activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(innode)
                #print("down block {layer} : CONV-1 shape is {shape}".format(layer=layer, shape=conv.shape))
                conv = Conv2D(filters=down_features[block], kernel_size=3, strides=(1, 1), padding=pattern,
                                activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(conv)
                innode = conv
                self.layers['down_block_'+str(block)] = innode

                #Down Blocks 
                for block in range(1,self.block_num+1):
                    with tf.variable_scope('down_block_'+str(block)):
                        pool = MaxPooling2D(pool_size=(2,2))(innode)
                        #print("down block {layer} : POOL shape is {shape}".format(layer=block, shape=pool.shape))
                        conv = Conv2D(filters=down_features[block], kernel_size=3, strides=(1, 1), padding=pattern,
                                activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(pool)
                        #print("down block {layer} : CONV-1 shape is {shape}".format(layer=block, shape=conv.shape))
                        conv = Conv2D(filters=down_features[block], kernel_size=3, strides=(1, 1), padding=pattern,
                                activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(conv)
                        #print("down block {layer} : CONV-2 shape is {shape}".format(layer=block, shape=conv.shape))
                        down_conv[str(block)] = conv
                        innode = conv
                        self.layers['down_block_'+str(block)] = innode

            with tf.name_scope('path_up') as scope:
                for block in range(0,self.block_num):
                    with tf.variable_scope('up_block_'+str(block)):
                        conv = Conv2D(filters=up_features, kernel_size=3, strides=(1, 1), padding=pattern,
                                activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(innode)
                        #print("up block {layer} : CONV-1 shape is {shape}".format(layer=block, shape=conv.shape))
                        conv = Conv2DTranspose(filters=up_features, kernel_size=3, strides=(2, 2), padding=pattern,
                            activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(conv)
                        #print("up block {layer} : CONV-1 shape is {shape}".format(layer=block, shape=conv.shape))

                        down_f = self.layers['down_block_'+str(self.block_num-block-1)]
                        down_f = Conv2D(filters=up_features, kernel_size=1, strides=(1, 1), padding=pattern,
                                activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(down_f)
                        #print(conv.shape,conv.dtype)
                        #print(down_f.shape,down_f.dtype)
                        conv = tf.add(conv,down_f)

                        output = Conv2D(filters=32, kernel_size=3, strides=(1, 1), padding=pattern,
                                activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(conv)
                        #print(output.shape,output.dtype)
                        output = Conv2D(filters=2, kernel_size=1, strides=(1, 1), padding=pattern,
                                activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(output)

                        self.layers['output'+str(block)] = output

                        innode = conv

            

            self.prediction = output
            print("prediction :  shape is {shape}".format(shape=self.prediction.shape))
            self.finalpre = tf.nn.softmax(output)
            print("finalpre :  shape is {shape}".format(shape=self.finalpre.shape))
  
class FPN_RFA:

    def __init__(self, imgs,block_num=4):
        self.imgs = imgs
        self.block_num = block_num
        self.prediction = []
        self.finalpre = []
        self.layers={}
        self.convlayers()
        

    def convlayers(self):
        self.layers = {}
        down_features = [64, 128,256,512,1024]
        up_features = 256

        pattern = 'same'
        std_dev = 5e-3

        innode = self.imgs
        conv = Conv2D(filters=down_features[0], kernel_size=3, strides=(1, 1), padding=pattern,
                        activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(innode)
        conv = Conv2D(filters=down_features[0], kernel_size=3, strides=(1, 1), padding=pattern,
                        activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(conv)
        innode = conv
        self.layers['down_block_0'] = innode
        # down_path
        with tf.name_scope('path_down') as scope:
            pattern = 'same'
            down_conv = OrderedDict()

            for block in range(1,self.block_num+1):
                with tf.variable_scope('down_block_'+str(block)):
                    
                    pool = MaxPooling2D(pool_size=(2,2))(innode)
                    conv = RFA_block(pool, down_features[block], 'down_block_'+str(block),num=2)
                    innode = conv
                    print('down_block_'+str(block),conv.shape)
                    self.layers['down_block_'+str(block)] = innode
            

        with tf.name_scope('path_up') as scope:
            for block in range(0,self.block_num):
                with tf.variable_scope('up_block_'+str(block)):
                    conv = Conv2D(filters=up_features, kernel_size=3, strides=(1, 1), padding=pattern,
                            activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(innode)
                    #print("up block {layer} : CONV-1 shape is {shape}".format(layer=block, shape=conv.shape))
                    conv = Conv2DTranspose(filters=up_features, kernel_size=3, strides=(2, 2), padding=pattern,
                        activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(conv)
                    #print("up block {layer} : CONV-1 shape is {shape}".format(layer=block, shape=conv.shape))

                    down_f = self.layers['down_block_'+str(self.block_num-block-1)]
                    down_f = Conv2D(filters=up_features, kernel_size=1, strides=(1, 1), padding=pattern,
                            activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(down_f)
                    #print(conv.shape,conv.dtype)
                    #print(down_f.shape,down_f.dtype)
                    conv = tf.add(conv,down_f)

                    output = Conv2D(filters=32, kernel_size=3, strides=(1, 1), padding=pattern,
                            activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(conv)
                    #print(output.shape,output.dtype)
                    output = Conv2D(filters=2, kernel_size=1, strides=(1, 1), padding=pattern,
                            activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(output)

                    self.layers['output'+str(block)] = output

                    innode = conv

        
        self.prediction = output
        print("prediction :  shape is {shape}".format(shape=self.prediction.shape))
        self.finalpre = tf.nn.softmax(output)
        print("finalpre :  shape is {shape}".format(shape=self.finalpre.shape))

class FPN_RFA_H:
    def __init__(self, imgs,block_num=4):
        self.imgs = imgs
        self.block_num = block_num
        self.prediction = []
        self.finalpre = {}
        self.layers={}
        self.convlayers()
        

    def convlayers(self):
        self.layers = {}
        down_features = [64, 128,256,512,1024]
        up_feature = 256

        pattern = 'same'
        std_dev = 5e-3

        innode = self.imgs
        conv = Conv2D(filters=down_features[0], kernel_size=3, strides=(1, 1), padding=pattern,
                        activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(innode)
        conv = Conv2D(filters=down_features[0], kernel_size=3, strides=(1, 1), padding=pattern,
                        activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(conv)
        innode = conv
        self.layers['down_block_0'] = innode
        # down_path
        with tf.name_scope('path_down') as scope:
            pattern = 'same'
            down_conv = OrderedDict()

            for block in range(1,self.block_num+1):
                with tf.variable_scope('down_block_'+str(block)):
                    
                    pool = MaxPooling2D(pool_size=(2,2))(innode)
                    conv = RFA_block(pool, down_features[block], 'down_block_'+str(block),num=2)
                    innode = conv
                    print('down_block_'+str(block),conv.shape)
                    self.layers['down_block_'+str(block)] = innode
            

        with tf.name_scope('path_up') as scope:
            for block in range(0,self.block_num):
                with tf.variable_scope('up_block_'+str(block)):
                    conv = Conv2D(filters=up_features, kernel_size=3, strides=(1, 1), padding=pattern,
                            activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(innode)
                    #print("up block {layer} : CONV-1 shape is {shape}".format(layer=block, shape=conv.shape))
                    conv = Conv2DTranspose(filters=up_features, kernel_size=3, strides=(2, 2), padding=pattern,
                        activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(conv)
                    #print("up block {layer} : CONV-1 shape is {shape}".format(layer=block, shape=conv.shape))

                    down_f = self.layers['down_block_'+str(self.block_num-block-1)]
                    down_f = Conv2D(filters=up_features, kernel_size=1, strides=(1, 1), padding=pattern,
                            activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(down_f)
                    #print(conv.shape,conv.dtype)
                    #print(down_f.shape,down_f.dtype)
                    conv = tf.add(conv,down_f)

                    output = Conv2D(filters=32, kernel_size=3, strides=(1, 1), padding=pattern,
                            activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(conv)
                    #print(output.shape,output.dtype)
                    output = Conv2D(filters=2, kernel_size=1, strides=(1, 1), padding=pattern,
                            activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(output)

                    self.finalpre[str(block)]= tf.nn.softmax(output)
                    self.layers['output'+str(block)] = output

                    innode = conv
        
        self.prediction = output
        print("prediction :  shape is {shape}".format(shape=self.prediction.shape))
        self.finalpre = tf.nn.softmax(output)
        print("finalpre :  shape is {shape}".format(shape=self.finalpre.shape))

class FPN_H:
    def __init__(self, imgs,block_num=4):
        self.imgs = imgs
        self.block_num = block_num
        self.prediction = []
        self.finalpre = {}
        self.layers={}
        self.convlayers()    

    def convlayers(self):

        down_features = [64, 128,256,512,1024]
        up_features = 256
        std_dev = 5e-3

        with tf.name_scope('path_down') as scope:
            pattern = 'same'
            down_conv = OrderedDict()
            innode =  self.imgs
            self.layers['input'] = innode

            block = 0
            innode = self.imgs

            # Two conv : input B*H*W*3 output B*H*W*64
            conv = Conv2D(filters=down_features[block], kernel_size=3, strides=(1, 1), padding=pattern,
                            activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(innode)
            #print("down block {layer} : CONV-1 shape is {shape}".format(layer=layer, shape=conv.shape))
            conv = Conv2D(filters=down_features[block], kernel_size=3, strides=(1, 1), padding=pattern,
                            activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(conv)
            innode = conv
            self.layers['down_block_'+str(block)] = innode

            #Down Blocks 
            for block in range(1,self.block_num+1):
                with tf.variable_scope('down_block_'+str(block)):
                    pool = MaxPooling2D(pool_size=(2,2))(innode)
                    #print("down block {layer} : POOL shape is {shape}".format(layer=block, shape=pool.shape))
                    conv = Conv2D(filters=down_features[block], kernel_size=3, strides=(1, 1), padding=pattern,
                            activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(pool)
                    #print("down block {layer} : CONV-1 shape is {shape}".format(layer=block, shape=conv.shape))
                    conv = Conv2D(filters=down_features[block], kernel_size=3, strides=(1, 1), padding=pattern,
                            activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(conv)
                    #print("down block {layer} : CONV-2 shape is {shape}".format(layer=block, shape=conv.shape))
                    down_conv[str(block)] = conv
                    innode = conv
                    self.layers['down_block_'+str(block)] = innode

        with tf.name_scope('path_up') as scope:
            for block in range(0,self.block_num):
                with tf.variable_scope('up_block_'+str(block)):
                    conv = Conv2D(filters=up_features, kernel_size=3, strides=(1, 1), padding=pattern,
                            activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(innode)
                    #print("up block {layer} : CONV-1 shape is {shape}".format(layer=block, shape=conv.shape))
                    conv = Conv2DTranspose(filters=up_features, kernel_size=3, strides=(2, 2), padding=pattern,
                        activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(conv)
                    #print("up block {layer} : CONV-1 shape is {shape}".format(layer=block, shape=conv.shape))

                    down_f = self.layers['down_block_'+str(self.block_num-block-1)]
                    down_f = Conv2D(filters=up_features, kernel_size=1, strides=(1, 1), padding=pattern,
                            activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(down_f)
                    #print(conv.shape,conv.dtype)
                    #print(down_f.shape,down_f.dtype)
                    conv = tf.add(conv,down_f)

                    output = Conv2D(filters=32, kernel_size=3, strides=(1, 1), padding=pattern,
                            activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(conv)
                    #print(output.shape,output.dtype)
                    output = Conv2D(filters=2, kernel_size=1, strides=(1, 1), padding=pattern,
                            activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(output)

                    self.layers[str(block)] = output
                    self.finalpre[str(block)] = tf.nn.softmax(output)

                    innode = conv

class FPN_RFA_BS:

    def __init__(self, imgs,score,block_num=4):
        self.imgs = imgs
        self.block_num = block_num
        self.prediction = []
        self.finalpre = []
        self.layers={}
        self.score = score
        self.convlayers()
        

    def convlayers(self):
        self.layers = {}
        down_features = [64, 128,256,512,1024]
        up_features = 256

        pattern = 'same'
        std_dev = 5e-3

        innode = self.imgs
        with tf.name_scope('FPN_RFA_BS'):
            conv = Conv2D(filters=down_features[0], kernel_size=3, strides=(1, 1), padding=pattern,
                            activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(innode)
            conv = Conv2D(filters=down_features[0], kernel_size=3, strides=(1, 1), padding=pattern,
                            activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(conv)
            conv = Conv2D(filters=down_features[0], kernel_size=3, strides=(1, 1), padding=pattern,
                            activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(conv)
            innode = conv
            self.layers['down_block_0'] = innode
            # down_path
            with tf.name_scope('path_down') as scope:
                pattern = 'same'
                down_conv = OrderedDict()

                for block in range(1,self.block_num+1):
                    with tf.variable_scope('down_block_'+str(block)):
                        
                        pool = MaxPooling2D(pool_size=(2,2))(innode)
                        conv = RFA_block(pool, down_features[block], 'down_block_'+str(block),num=2)
                        innode = conv
                        print('down_block_'+str(block),conv.shape)
                        self.layers['down_block_'+str(block)] = innode
                

            with tf.name_scope('path_up') as scope:
                for block in range(0,self.block_num):
                    with tf.variable_scope('up_block_'+str(block)):
                        conv = Conv2D(filters=up_features, kernel_size=3, strides=(1, 1), padding=pattern,
                                activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(innode)
                        #print("up block {layer} : CONV-1 shape is {shape}".format(layer=block, shape=conv.shape))
                        conv = Conv2DTranspose(filters=up_features, kernel_size=3, strides=(2, 2), padding=pattern,
                            activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(conv)
                        #print("up block {layer} : CONV-1 shape is {shape}".format(layer=block, shape=conv.shape))

                        down_f = self.layers['down_block_'+str(self.block_num-block-1)]
                        down_f = Conv2D(filters=up_features, kernel_size=1, strides=(1, 1), padding=pattern,
                                activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(down_f)
                        #print(conv.shape,conv.dtype)
                        #print(down_f.shape,down_f.dtype)
                        conv = tf.add(conv,down_f)

                        output = Conv2D(filters=32, kernel_size=3, strides=(1, 1), padding=pattern,
                                activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(conv)
                        #print(output.shape,output.dtype)
                        output = Conv2D(filters=2, kernel_size=1, strides=(1, 1), padding=pattern,
                                activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(output)

                        self.layers['output'+str(block)] = output

                        innode = conv

            
            self.prediction = output
            print("prediction :  shape is {shape}".format(shape=self.prediction.shape))
            B,H,W,C = output.shape
            print(self.score.shape)
            temp = tf.expand_dims(tf.expand_dims(tf.expand_dims(self.score, axis=1),axis=1),axis=1)
            temp2 = tf.ones(temp.shape, dtype=tf.dtypes.float32)
            CS = tf.reshape(tf.concat([temp,temp2],axis= 3),[B,1,2])

            self.finalpre= tf.nn.softmax(tf.reshape(tf.reshape(output,[B,H*W,C])*CS,[B,H,W,C]))
            print("finalpre :  shape is {shape}".format(shape=self.finalpre.shape))

class FPN_RFA_BS_1:

    def __init__(self, imgs,score,block_num=4):
        self.imgs = imgs
        self.block_num = block_num
        self.prediction = []
        self.finalpre = []
        self.layers={}
        self.score = score
        self.convlayers()
        

    def convlayers(self):
        self.layers = {}
        down_features = [64, 128,256,512,1024]
        up_features = 256

        pattern = 'same'
        std_dev = 5e-3

        innode = self.imgs
        with tf.name_scope('FPN_RFA_BS'):
            conv = Conv2D(filters=down_features[0], kernel_size=3, strides=(1, 1), padding=pattern,
                            activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(innode)
            conv = Conv2D(filters=down_features[0], kernel_size=3, strides=(1, 1), padding=pattern,
                            activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(conv)
            innode = conv
            self.layers['down_block_0'] = innode
            # down_path
            with tf.name_scope('path_down') as scope:
                pattern = 'same'
                down_conv = OrderedDict()

                for block in range(1,self.block_num+1):
                    with tf.variable_scope('down_block_'+str(block)):
                        
                        pool = MaxPooling2D(pool_size=(2,2))(innode)
                        conv = RFA_block(pool, down_features[block], 'down_block_'+str(block),num=2)
                        innode = conv
                        print('down_block_'+str(block),conv.shape)
                        self.layers['down_block_'+str(block)] = innode
                

            with tf.name_scope('path_up') as scope:
                for block in range(0,self.block_num):
                    with tf.variable_scope('up_block_'+str(block)):
                        conv = Conv2D(filters=up_features, kernel_size=3, strides=(1, 1), padding=pattern,
                                activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(innode)
                        #print("up block {layer} : CONV-1 shape is {shape}".format(layer=block, shape=conv.shape))
                        conv = Conv2DTranspose(filters=up_features, kernel_size=3, strides=(2, 2), padding=pattern,
                            activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(conv)
                        #print("up block {layer} : CONV-1 shape is {shape}".format(layer=block, shape=conv.shape))

                        down_f = self.layers['down_block_'+str(self.block_num-block-1)]
                        down_f = Conv2D(filters=up_features, kernel_size=1, strides=(1, 1), padding=pattern,
                                activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(down_f)
                        #print(conv.shape,conv.dtype)
                        #print(down_f.shape,down_f.dtype)
                        if block <self.block_num-1:
                            conv = tf.add(conv,down_f)
                        else: 
                            conv = tf.concat([conv,down_f],axis =3)

                        output = Conv2D(filters=32, kernel_size=3, strides=(1, 1), padding=pattern,
                                activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(conv)
                        #print(output.shape,output.dtype)
                        output = Conv2D(filters=2, kernel_size=1, strides=(1, 1), padding=pattern,
                                activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(output)

                        self.layers['output'+str(block)] = output

                        innode = conv

            
            self.prediction = output
            print("prediction :  shape is {shape}".format(shape=self.prediction.shape))
            B,H,W,C = output.shape
            print(self.score.shape)
            temp = tf.expand_dims(tf.expand_dims(tf.expand_dims(self.score, axis=1),axis=1),axis=1)
            temp2 = tf.ones(temp.shape, dtype=tf.dtypes.float32)
            CS = tf.reshape(tf.concat([temp,temp2],axis= 3),[B,1,2])

            self.finalpre= tf.nn.softmax(tf.reshape(tf.reshape(output,[B,H*W,C])*CS,[B,H,W,C]))
            print("finalpre :  shape is {shape}".format(shape=self.finalpre.shape))

class FPN_RFA_BS_2:

    def __init__(self, imgs,score,block_num=4):
        self.imgs = imgs
        self.block_num = block_num
        self.prediction = []
        self.finalpre = []
        self.layers={}
        self.score = score
        self.convlayers()
        

    def convlayers(self):
        self.layers = {}
        down_features = [64, 128,256,512,1024]
        up_features = 256

        pattern = 'same'
        std_dev = 5e-3

        innode = self.imgs
        with tf.name_scope('FPN_RFA_BS'):
            conv = Conv2D(filters=down_features[0], kernel_size=3, strides=(1, 1), padding=pattern,
                            activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(innode)
            conv = Conv2D(filters=down_features[0], kernel_size=3, strides=(1, 1), padding=pattern,
                            activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(conv)
            innode = conv
            self.layers['down_block_0'] = innode
            # down_path
            with tf.name_scope('path_down') as scope:
                pattern = 'same'
                down_conv = OrderedDict()

                for block in range(1,self.block_num+1):
                    with tf.variable_scope('down_block_'+str(block)):
                        
                        pool = MaxPooling2D(pool_size=(2,2))(innode)
                        conv = RFA_block(pool, down_features[block], 'down_block_'+str(block),num=2)
                        innode = conv
                        print('down_block_'+str(block),conv.shape)
                        self.layers['down_block_'+str(block)] = innode
                

            with tf.name_scope('path_up') as scope:
                for block in range(0,self.block_num):
                    with tf.variable_scope('up_block_'+str(block)):
                        conv = Conv2D(filters=up_features, kernel_size=3, strides=(1, 1), padding=pattern,
                                activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(innode)
                        #print("up block {layer} : CONV-1 shape is {shape}".format(layer=block, shape=conv.shape))
                        
                        conv = Conv2DTranspose(filters=up_features, kernel_size=3, strides=(2, 2), padding=pattern,
                            activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(conv)
                        #print("up block {layer} : CONV-1 shape is {shape}".format(layer=block, shape=conv.shape))

                        down_f = self.layers['down_block_'+str(self.block_num-block-1)]
                        down_f = Conv2D(filters=up_features, kernel_size=1, strides=(1, 1), padding=pattern,
                                activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(down_f)
                        #print(conv.shape,conv.dtype)
                        #print(down_f.shape,down_f.dtype)
                        conv = tf.add(conv,down_f)
                        innode = conv

                output = Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding=pattern,
                                activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(innode)
                output = Conv2D(filters=32, kernel_size=3, strides=(1, 1), padding=pattern,
                                activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(output)
                output = Conv2D(filters=32, kernel_size=3, strides=(1, 1), padding=pattern,
                                activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(output)
                        #print(output.shape,output.dtype)
                output = Conv2D(filters=2, kernel_size=1, strides=(1, 1), padding=pattern,
                                activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(output)


            
            self.prediction = output
            print("prediction :  shape is {shape}".format(shape=self.prediction.shape))
            B,H,W,C = output.shape
            print(self.score.shape)
            temp = tf.expand_dims(tf.expand_dims(tf.expand_dims(self.score, axis=1),axis=1),axis=1)
            temp2 = tf.ones(temp.shape, dtype=tf.dtypes.float32)
            CS = tf.reshape(tf.concat([temp,temp2],axis= 3),[B,1,2])

            self.finalpre= tf.nn.softmax(tf.reshape(tf.reshape(output,[B,H*W,C])*CS,[B,H,W,C]))
            print("finalpre :  shape is {shape}".format(shape=self.finalpre.shape))


class FPN_RFA_BS_3:

    def __init__(self, imgs,score,block_num=4):
        self.imgs = imgs
        self.block_num = block_num
        self.prediction = []
        self.finalpre = []
        self.layers={}
        self.score = score
        self.convlayers()
        

    def convlayers(self):
        self.layers = {}
        down_features = [64, 128,256,512,1024]
        up_features = 256

        pattern = 'same'
        std_dev = 5e-3

        innode = self.imgs
        with tf.name_scope('FPN_RFA_BS'):
            conv = Conv2D(filters=down_features[0], kernel_size=3, strides=(1, 1), padding=pattern,
                            activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(innode)
            conv = Conv2D(filters=down_features[0], kernel_size=3, strides=(1, 1), padding=pattern,
                            activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(conv)
            innode = conv
            self.layers['down_block_0'] = innode
            # down_path
            with tf.name_scope('path_down') as scope:
                pattern = 'same'
                down_conv = OrderedDict()

                for block in range(1,self.block_num+1):
                    with tf.variable_scope('down_block_'+str(block)):
                        
                        pool = MaxPooling2D(pool_size=(2,2))(innode)
                        conv = RFA_block(pool, down_features[block], 'down_block_'+str(block),num=2)
                        innode = conv
                        print('down_block_'+str(block),conv.shape)
                        self.layers['down_block_'+str(block)] = innode
                

            with tf.name_scope('path_up') as scope:
                for block in range(0,self.block_num):
                    with tf.variable_scope('up_block_'+str(block)):
                        conv = Conv2D(filters=up_features, kernel_size=3, strides=(1, 1), padding=pattern,
                                activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(innode)
                        #print("up block {layer} : CONV-1 shape is {shape}".format(layer=block, shape=conv.shape))
                        
                        conv = Conv2DTranspose(filters=up_features, kernel_size=3, strides=(2, 2), padding=pattern,
                            activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(conv)
                        #print("up block {layer} : CONV-1 shape is {shape}".format(layer=block, shape=conv.shape))

                        down_f = self.layers['down_block_'+str(self.block_num-block-1)]
                        down_f = Conv2D(filters=up_features, kernel_size=1, strides=(1, 1), padding=pattern,
                                activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(down_f)
                        #print(conv.shape,conv.dtype)
                        #print(down_f.shape,down_f.dtype)
                        conv = tf.add(conv,down_f)
                        innode = conv

                output = Conv2D(filters=128, kernel_size=3, strides=(1, 1), padding=pattern,
                                activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(innode)
                output = Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding=pattern,
                                activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(output)
                output = Conv2D(filters=32, kernel_size=3, strides=(1, 1), padding=pattern,
                                activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(output)
                output = Conv2D(filters=32, kernel_size=3, strides=(1, 1), padding=pattern,
                                activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(output)#print(output.shape,output.dtype)
                output = Conv2D(filters=2, kernel_size=1, strides=(1, 1), padding=pattern,
                                activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(output)


            
            self.prediction = output
            print("prediction :  shape is {shape}".format(shape=self.prediction.shape))
            B,H,W,C = output.shape
            print(self.score.shape)
            temp = tf.expand_dims(tf.expand_dims(tf.expand_dims(self.score, axis=1),axis=1),axis=1)
            temp2 = tf.ones(temp.shape, dtype=tf.dtypes.float32)
            CS = tf.reshape(tf.concat([temp,temp2],axis= 3),[B,1,2])

            self.finalpre= tf.nn.softmax(tf.reshape(tf.reshape(output,[B,H*W,C])*CS,[B,H,W,C]))
            print("finalpre :  shape is {shape}".format(shape=self.finalpre.shape))

class FPN_RFA_3:

    def __init__(self, imgs,score,block_num=4):
        self.imgs = imgs
        self.block_num = block_num
        self.prediction = []
        self.finalpre = []
        self.layers={}
        self.score = score
        self.convlayers()
        

    def convlayers(self):
        self.layers = {}
        down_features = [64, 128,256,512,1024]
        up_features = 256

        pattern = 'same'
        std_dev = 5e-3

        innode = self.imgs
        with tf.name_scope('FPN_RFA_BS'):
            conv = Conv2D(filters=down_features[0], kernel_size=3, strides=(1, 1), padding=pattern,
                            activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(innode)
            conv = Conv2D(filters=down_features[0], kernel_size=3, strides=(1, 1), padding=pattern,
                            activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(conv)
            innode = conv
            self.layers['down_block_0'] = innode
            # down_path
            with tf.name_scope('path_down') as scope:
                pattern = 'same'
                down_conv = OrderedDict()

                for block in range(1,self.block_num+1):
                    with tf.variable_scope('down_block_'+str(block)):
                        
                        pool = MaxPooling2D(pool_size=(2,2))(innode)
                        conv = RFA_block(pool, down_features[block], 'down_block_'+str(block),num=2)
                        innode = conv
                        print('down_block_'+str(block),conv.shape)
                        self.layers['down_block_'+str(block)] = innode
                

            with tf.name_scope('path_up') as scope:
                for block in range(0,self.block_num):
                    with tf.variable_scope('up_block_'+str(block)):
                        conv = Conv2D(filters=up_features, kernel_size=3, strides=(1, 1), padding=pattern,
                                activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(innode)
                        #print("up block {layer} : CONV-1 shape is {shape}".format(layer=block, shape=conv.shape))
                        
                        conv = Conv2DTranspose(filters=up_features, kernel_size=3, strides=(2, 2), padding=pattern,
                            activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(conv)
                        #print("up block {layer} : CONV-1 shape is {shape}".format(layer=block, shape=conv.shape))

                        down_f = self.layers['down_block_'+str(self.block_num-block-1)]
                        down_f = Conv2D(filters=up_features, kernel_size=1, strides=(1, 1), padding=pattern,
                                activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(down_f)
                        #print(conv.shape,conv.dtype)
                        #print(down_f.shape,down_f.dtype)
                        conv = tf.add(conv,down_f)
                        innode = conv

                output = Conv2D(filters=128, kernel_size=3, strides=(1, 1), padding=pattern,
                                activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(innode)
                output = Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding=pattern,
                                activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(output)
                output = Conv2D(filters=32, kernel_size=3, strides=(1, 1), padding=pattern,
                                activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(output)
                output = Conv2D(filters=32, kernel_size=3, strides=(1, 1), padding=pattern,
                                activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(output)#print(output.shape,output.dtype)
                output = Conv2D(filters=2, kernel_size=1, strides=(1, 1), padding=pattern,
                                activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(output)


            
            self.prediction = output
            self.finalpre= tf.nn.softmax(output)

class FPN_RFA_4:

    def __init__(self, imgs,score,block_num=4):
        self.imgs = imgs
        self.block_num = block_num
        self.prediction = []
        self.finalpre = []
        self.layers={}
        self.score = score
        self.convlayers()
        

    def convlayers(self):
        self.layers = {}
        down_features = [64, 128,256,512,1024]
        up_features = 256

        pattern = 'same'
        std_dev = 5e-3

        innode = self.imgs
        with tf.name_scope('FPN_RFA_BS'):
            conv = Conv2D(filters=down_features[0], kernel_size=3, strides=(1, 1), padding=pattern,
                            activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(innode)
            conv = Conv2D(filters=down_features[0], kernel_size=3, strides=(1, 1), padding=pattern,
                            activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(conv)
            innode = conv
            self.layers['down_block_0'] = innode
            # down_path
            with tf.name_scope('path_down') as scope:
                pattern = 'same'
                down_conv = OrderedDict()

                for block in range(1,self.block_num+1):
                    with tf.variable_scope('down_block_'+str(block)):
                        
                        pool = MaxPooling2D(pool_size=(2,2))(innode)
                        conv = RFA_block(pool, down_features[block], 'down_block_'+str(block),num=2)
                        innode = conv
                        print('down_block_'+str(block),conv.shape)
                        self.layers['down_block_'+str(block)] = innode
                

            with tf.name_scope('path_up') as scope:
                for block in range(0,self.block_num):
                    with tf.variable_scope('up_block_'+str(block)):
                        conv = Conv2D(filters=128, kernel_size=3, strides=(1, 1), padding=pattern,
                                activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(innode)
                        conv = Conv2D(filters=up_features, kernel_size=3, strides=(1, 1), padding=pattern,
                                activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(conv)
                        #print("up block {layer} : CONV-1 shape is {shape}".format(layer=block, shape=conv.shape))
                        
                        conv = Conv2DTranspose(filters=up_features, kernel_size=3, strides=(2, 2), padding=pattern,
                            activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(conv)
                        #print("up block {layer} : CONV-1 shape is {shape}".format(layer=block, shape=conv.shape))

                        down_f = self.layers['down_block_'+str(self.block_num-block-1)]
                        down_f = Conv2D(filters=up_features, kernel_size=1, strides=(1, 1), padding=pattern,
                                activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(down_f)
                        #print(conv.shape,conv.dtype)
                        #print(down_f.shape,down_f.dtype)
                        conv = tf.add(conv,down_f)
                        innode = conv

                output = Conv2D(filters=32, kernel_size=3, strides=(1, 1), padding=pattern,
                                activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(innode)
                output = Conv2D(filters=32, kernel_size=3, strides=(1, 1), padding=pattern,
                                activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(output)
                output = Conv2D(filters=32, kernel_size=3, strides=(1, 1), padding=pattern,
                                activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(output)
                output = Conv2D(filters=32, kernel_size=3, strides=(1, 1), padding=pattern,
                                activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(output)#print(output.shape,output.dtype)
                output = Conv2D(filters=2, kernel_size=1, strides=(1, 1), padding=pattern,
                                activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(output)


            
            self.prediction = output
            self.finalpre= tf.nn.softmax(output)


class FPN_RFA_BS_4:

    def __init__(self, imgs,score,block_num=4,image_size=256):
        self.imgs = imgs
        self.image_size = image_size
        self.block_num = block_num
        self.prediction = []
        self.finalpre = []
        self.layers={}
        self.score = score
        self.weights = []
        self.output ={}
        self.convlayers()
        

    def convlayers(self):
        self.layers = {}
        
        down_features = [64, 128,256,512,1024]
        up_features = 256

        pattern = 'same'
        std_dev = 5e-3

        innode = self.imgs
        with tf.name_scope('FPN_RFA_BS'):
            conv = Conv2D(filters=down_features[0], kernel_size=3, strides=(1, 1), padding=pattern,
                            activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(innode)
            conv = Conv2D(filters=down_features[0], kernel_size=3, strides=(1, 1), padding=pattern,
                            activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(conv)
            innode = conv

            self.layers['down_block_0'] = innode
            # down_path
            with tf.name_scope('path_down') as scope:
                pattern = 'same'
                down_conv = OrderedDict()

                for block in range(1,self.block_num+1):
                    with tf.variable_scope('down_block_'+str(block)):
                        
                        pool = MaxPooling2D(pool_size=(2,2))(innode)
                        conv = RFA_block(pool, down_features[block], 'down_block_'+str(block),num=2)
                        innode = conv
                        print('down_block_'+str(block),conv.shape)
                        self.layers['down_block_'+str(block)] = innode
                

            with tf.name_scope('path_up') as scope:
                for block in range(0,self.block_num):
                    with tf.variable_scope('up_block_'+str(block)):
                        conv = Conv2D(filters=128, kernel_size=3, strides=(1, 1), padding=pattern,
                                activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(innode)
                        
                        conv = Conv2DTranspose(filters=up_features, kernel_size=3, strides=(2, 2), padding=pattern,
                            activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(conv)
                        #print("up block {layer} : CONV-1 shape is {shape}".format(layer=block, shape=conv.shape))

                        down_f = self.layers['down_block_'+str(self.block_num-block-1)]
                        down_f = Conv2D(filters=up_features, kernel_size=1, strides=(1, 1), padding=pattern,
                                activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(down_f)
                        #print(conv.shape,conv.dtype)
                        #print(down_f.shape,down_f.dtype)
                        conv = tf.add(conv,down_f)
                        innode = conv

                        #conv = Conv2DTranspose(filters=64, kernel_size=3, strides=(2**(self.block_num-block-1), 2**(self.block_num-block-1)), padding=pattern,
                                #activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(innode)
                        conv = Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding=pattern,
                                activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(innode)
                        conv = Conv2D(filters=32, kernel_size=3, strides=(1, 1), padding=pattern,
                                activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(conv)
                        conv = Conv2D(filters=32, kernel_size=3, strides=(1, 1), padding=pattern,
                                activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(conv)
                        conv = Conv2D(filters=2, kernel_size=3, strides=(1, 1), padding=pattern,
                                activation="relu", kernel_initializer=RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer="zeros")(conv)
                        
                        conv = UpSampling2D(size=(self.image_size,self.image_size))(conv)
                        
                        self.output[str(block)]= conv
            
            self.weights = tf.Variable(tf.zeros([self.block_num-1]), trainable = True)
            
            OUT = 0
            for block in range(self.block_num):
                if block == 0:
                    continue
                else:
                    OUT += self.weights[block-1]*self.output[str(block)]
                    tf.summary.scalar('weights'+str(block),self.weights[block-1])

            
            self.prediction = OUT
            print("prediction :  shape is {shape}".format(shape=self.prediction.shape))
            B,H,W,C = OUT.shape
            print(self.score.shape)
            temp = tf.expand_dims(tf.expand_dims(tf.expand_dims(self.score, axis=1),axis=1),axis=1)
            temp2 = tf.ones(temp.shape, dtype=tf.dtypes.float32)
            CS = tf.reshape(tf.concat([temp,temp2],axis= 3),[B,1,2])

            self.finalpre= tf.nn.softmax(tf.reshape(tf.reshape(OUT,[B,H*W,C])*CS,[B,H,W,C]))
            print("finalpre :  shape is {shape}".format(shape=self.finalpre.shape))
