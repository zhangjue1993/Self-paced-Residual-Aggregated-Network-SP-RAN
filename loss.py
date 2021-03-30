#coding=utf-8
import tensorflow as tf
# from vgg import vgg16
import os
import numpy as np
import cv2
import random
import utils

def dice_loss(y_true, y_pred):
  numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=-1)
  denominator = tf.reduce_sum(y_true + y_pred, axis=-1)

  return tf.reduce_mean(1 - (numerator + 1) / (denominator + 1))

def cross_entropy(y_true, y_pred):
    loss = -tf.reduce_mean(labels*tf.log(y_hat)) + -tf.reduce_mean((1-labels)*tf.log(1-y_hat))
    return loss


def loss_deep_supervision(labels, y_hat, batch_size, T, stddev_tensor):
    # loss = tf.constant(0)
    loss = []
    loss_sum = 0
    k = 0
    # stddev_tensor = tf.placeholder(tf.float32, name='stddev')
    for y in y_hat:
        k = k+1
        y_shape = y.shape[1]
        label_scaled = tf.image.resize_images(labels, [y_shape, y_shape], method=tf.image.ResizeMethod.BICUBIC)
        T_matrix_rand = tf.random_normal([batchsize, 256, 256, 1], mean=T, stddev=(stddev_tensor/k))
        one = tf.ones_like(T_matrix)
        zero = tf.zeros_like(T_matrix)
        T_scaled = tf.where(T_matrix_rand < 0.5, zero, one)
        tf.summary.image('Transformation Matrix_'+str(k), T_scaled)
        temp = cross_entropy_normal(label_scaled, y+1e-5, batch_size, y_shape, T_scaled)
        loss.append(temp)
        loss_sum += temp
    return loss, loss_sum

def cross_entropy_normal(labels, y_hat, batch_size, image_size, T):
    beta = tf.reduce_mean(labels[:,:,:,0]) # ratio of foreground
    # loss = -tf.reduce_mean(labels[:,:,:,0]*tf.log(y_hat[:,:,:,0])) + -tf.reduce_mean((labels[:,:,:,1])*tf.log(y_hat[:,:,:,1]))
    # loss = -tf.reduce_mean(labels[:,:,:,0]*tf.log(y_hat[:,:,:,0])+labels[:,:,:,1]*tf.log(y_hat[:,:,:,1]))
    # loss = (-tf.reduce_sum(labels[:,:,:,0]*tf.log(y_hat[:,:,:,0])*beta)+-tf.reduce_sum(labels[:,:,:,1]*tf.log(y_hat[:,:,:,1])*(1-beta))/(batch_size * image_size * image_size * 2)
    # loss = -tf.reduce_mean(labels*tf.log(y_hat))
    # loss = -tf.reduce_sum(labels*tf.log(y_hat))/(batch_size * image_size * image_size * 2)
    # loss = -tf.reduce_sum(labels[:,:,:,0] * tf.log(y_hat[:,:,:,0]) + labels[:,:,:,1] * tf.log(y_hat[:,:,:,1]))/(batch_size * image_size * image_size * 2) #real cross_entropy
    loss = -tf.reduce_mean(tf.squeeze(T)* labels[:,:,:,0] * tf.log(y_hat[:,:,:,0]) +  tf.squeeze(T)* labels[:,:,:,1] * tf.log(y_hat[:,:,:,1]))
    # loss = (-tf.reduce_sum(labels[:,:,:,0]*tf.log(y_hat[:,:,:,0]))+-tf.reduce_sum(labels[:,:,:,1]*tf.log(y_hat[:,:,:,1])))/(batch_size * image_size * image_size * 2)
    return loss

def weight_balanced_cross_entropy(labels, y_hat):
    beta = tf.reduce_sum(tf.squeeze(T)*labels[:,:,:,0])/tf.reduce_sum(T) # ratio of foreground
    loss = -tf.reduce_mean((1-beta)*tf.squeeze(T)* labels[:,:,:,0] * tf.log(y_hat[:,:,:,0]) +  beta*tf.squeeze(T)* labels[:,:,:,1] * tf.log(y_hat[:,:,:,1]))
    # loss = (-tf.reduce_sum(labels[:,:,:,0]*tf.log(y_hat[:,:,:,0]))+-tf.reduce_sum(labels[:,:,:,1]*tf.log(y_hat[:,:,:,1])))/(batch_size * image_size * image_size * 2)
    return loss

def weighted_cross_entropy(label, y_hat):
    _,h,w,_ = y_hat.shape
    beta = tf.reduce_sum(label[:,:,:,0])/tf.cast(h*w,dtype=tf.float32) # ratio of foreground
    a = (1-beta)* label[:,:,:,0] * tf.log(y_hat[:,:,:,0]+1e-5)
    b =  beta* label[:,:,:,1] * tf.log((y_hat[:,:,:,1]+1e-5))
    loss = -(tf.reduce_mean(a)+tf.reduce_mean(b))
    return loss

def weighted_cross_entropy_s(label, y_hat):
    _,h,w,_ = y_hat.shape
    beta = tf.reduce_sum(label[:,:,:,0])/tf.cast(h*w,dtype=tf.float32) # ratio of foreground
    a = (1-beta)* label[:,:,:,0] * tf.log(y_hat[:,:,:,0]+1e-5)
    b =  beta* label[:,:,:,1] * tf.log((y_hat[:,:,:,1]+1e-5))
    loss = -tf.reduce_mean(tf.reduce_mean(a+b,axis = 2),axis =1)
     #loss = -(tf.reduce_mean(a)+tf.reduce_mean(b))
    return loss

def cross_entropy_H(label, finalpre):
    loss=0
    for block in finalpre:
        y_hat = finalpre[block]
        _,h,w,_=y_hat.shape
        newlabel = tf.image.resize(label, [h,w],method=tf.image.ResizeMethod.BICUBIC)
        a = newlabel[:,:,:,0] * tf.log(y_hat[:,:,:,0]+1e-5)
        b = newlabel[:,:,:,1] * tf.log((y_hat[:,:,:,1]+1e-5))
        loss -=(tf.reduce_mean(a)+tf.reduce_mean(b))
    return loss

def weighted_cross_entropy_H(label, finalpre):
    loss=0
    for block in finalpre:
        y_hat = finalpre[block]
        _,h,w,_=y_hat.shape
        newlabel = tf.image.resize(label, [h,w],method=tf.image.ResizeMethod.BICUBIC)
        beta = tf.reduce_sum(newlabel[:,:,:,0])/tf.cast(h*w,dtype=tf.float32) # ratio of foreground
        a = (1-beta)* newlabel[:,:,:,0] * tf.log(y_hat[:,:,:,0]+1e-5)
        b =  beta* newlabel[:,:,:,1] * tf.log((y_hat[:,:,:,1]+1e-5))
        loss -=(tf.reduce_mean(a)+tf.reduce_mean(b))
    return loss

def _cross_entropy(label, y_hat,image):
    _,h,w,_ = y_hat.shape
    beta = tf.reduce_sum(label[:,:,:,0])/tf.cast(h*w,dtype=tf.float32) # ratio of foreground
    a = (1-beta)* label[:,:,:,0] * tf.log(y_hat[:,:,:,0]+1e-5)
    b =  beta* label[:,:,:,1] * tf.log((y_hat[:,:,:,1]+1e-5))
    loss = -(tf.reduce_mean(a)+tf.reduce_mean(b))
    return loss


def _get_mesh(N, H, W):
    M1 = tf.range(0, W, 1.0)
    M1 = tf.repeat(tf.reshape(tf.cast(M1,dtype = tf.float32),[1, 1, W, 1]),repeats = N, axis = 0)
    M1 = tf.repeat(M1,repeats = H, axis = 1)
    M2 = tf.range(0, H, 1.0)
    M2 = tf.repeat(tf.reshape(tf.cast(M2,dtype = tf.float32),[1, H, 1, 1]),repeats = N, axis = 0)
    M2 = tf.repeat(M2,repeats = W, axis = 2)
    
    return tf.concat([M1,M2],axis= 3)

def _unfold(img, radius):
    N, H, W ,C= img.shape
    diameter = 2 * radius + 1
    pathes = tf.image.extract_patches(img,  sizes=[1, diameter, diameter, 1], strides=[1, 1, 1, 1],rates=[1, 1, 1, 1], padding='SAME', name=None)
    return tf.reshape(pathes,[N,diameter,diameter,H,W,C]) 

def _create_kernels_from_features(features, radius, kenerl_mask):
    N, H, W ,C= features.shape
    print(features.shape)
    print(kenerl_mask.shape)

    kernels = _unfold(features, radius)
    shape = kernels.shape
    kernels = kernels - tf.reshape(kernels[:, radius, radius, :,:, :], [N, 1, 1, H, W,C])
    kernels = tf.exp(tf.math.reduce_sum(-0.5 * kernels ** 2,axis = 5,keepdims=True))
    mask = tf.reshape(kenerl_mask,[1, shape[1], shape[2], 1, 1,1])
    kernels = kernels*mask
    return kernels


def _create_kernels( kernels_config, kernels_radius, input_feature,kenerl_mask, N, H, W):
    kernels = None
    for i, desc in enumerate(kernels_config):
        weight = desc['weight']
        features = []
        for modality, sigma in desc.items():
            if modality == 'weight':
                continue
            if modality == 'xy':
                feature = _get_mesh(N, H, W)
            else:
                feature = input_feature
            
            feature /= sigma
            features.append(feature)
        features = tf.concat(features, axis=3)

        kernel = weight * _create_kernels_from_features(features, kernels_radius,kenerl_mask)
        kernels = kernel if kernels is None else kernel + kernels
        return kernels

def CRF_loss(y_hat_softmax,input_feature,kernels_radius,CRF_loss_config,kenerl_mask,mask_dst=None,mask_src=None,compatibility=None):
    B,H,W,C = input_feature.shape
    _,_,_,C2 = y_hat_softmax.shape
    loss = 0
    kernels = _create_kernels(CRF_loss_config, kernels_radius, input_feature,kenerl_mask, B, H, W)
    denom = tf.cast(B * H * W,dtype=tf.float32)

    if mask_src is not None:
        denom = tf.math.reduce_sum(mask_src)
        mask_src = _unfold(mask_src, kernels_radius)
        kernels = kernels * mask_src

    if mask_dst is not None:
        denom = tf.math.reduce_sum(mask_dst)
        mask_dst = tf.reshpe(mask_dst,[B, 1, 1, H, W,1])
        kernels = kernels * mask_dst
    
    y_hat_unfolded = _unfold(y_hat_softmax, kernels_radius)

    product_kernel_x_y_hat = tf.math.reduce_sum(tf.reshape(kernels * y_hat_unfolded,[B, (kernels_radius * 2 + 1) ** 2, H, W,C2]), axis=1)
    
    if compatibility is None:
            # Using shortcut for Pott's class compatibility model
            # comment out to save computation, total loss may go below 0
            loss = -tf.math.reduce_sum(product_kernel_x_y_hat * y_hat_softmax)
            loss = tf.math.reduce_sum(kernels) + loss  

    return loss/denom

def CRF_loss_s(y_hat_softmax,input_feature,kernels_radius,CRF_loss_config,kenerl_mask,mask_dst=None,mask_src=None,compatibility=None):
    B,H,W,C = input_feature.shape
    _,_,_,C2 = y_hat_softmax.shape
    loss = 0
    kernels = _create_kernels(CRF_loss_config, kernels_radius, input_feature,kenerl_mask, B, H, W)
    denom = tf.cast(B * H * W,dtype=tf.float32)

    if mask_src is not None:
        denom = tf.math.reduce_sum(mask_src)
        mask_src = _unfold(mask_src, kernels_radius)
        kernels = kernels * mask_src

    if mask_dst is not None:
        denom = tf.math.reduce_sum(mask_dst)
        mask_dst = tf.reshpe(mask_dst,[B, 1, 1, H, W,1])
        kernels = kernels * mask_dst
    
    y_hat_unfolded = _unfold(y_hat_softmax, kernels_radius)

    product_kernel_x_y_hat = tf.math.reduce_sum(tf.reshape(kernels * y_hat_unfolded,[B, (kernels_radius * 2 + 1) ** 2, H, W,C2]), axis=1)
    
    if compatibility is None:
            # Using shortcut for Pott's class compatibility model
            # comment out to save computation, total loss may go below 0
            loss = -tf.math.reduce_sum(tf.reshape(product_kernel_x_y_hat * y_hat_softmax,[B,-1]),axis=1)
            newloss = tf.math.reduce_sum(tf.reshape(kernels,[B,-1]),axis=1) + loss  

    return newloss/denom




def W_CRF_loss(y_hat_softmax,input_feature,kernels_radius,score,CRF_loss_config,kenerl_mask,mask_dst=None,mask_src=None,compatibility=None):
    '''
    Classification score weighted GCRF loss
    '''
    B,H,W,C = input_feature.shape
    _,_,_,C2 = y_hat_softmax.shape
    loss = 0
    kernels = _create_kernels(CRF_loss_config, kernels_radius, input_feature,kenerl_mask, B, H, W)
    denom = tf.cast(B * H * W,dtype=tf.float32)

    if mask_src is not None:
        denom = tf.math.reduce_sum(mask_src)
        mask_src = _unfold(mask_src, kernels_radius)
        kernels = kernels * mask_src

    if mask_dst is not None:
        denom = tf.math.reduce_sum(mask_dst)
        mask_dst = tf.reshpe(mask_dst,[B, 1, 1, H, W,1])
        kernels = kernels * mask_dst
    
    y_hat_unfolded = _unfold(y_hat_softmax, kernels_radius)

    product_kernel_x_y_hat = tf.math.reduce_sum(tf.reshape(kernels * y_hat_unfolded,[B, (kernels_radius * 2 + 1) ** 2, H, W,C2]), axis=1)
    
    if compatibility is None:
            # Using shortcut for Pott's class compatibility model
            # comment out to save computation, total loss may go below 0
            loss = -tf.math.reduce_sum(tf.reshape(product_kernel_x_y_hat * y_hat_softmax,[H*W*C2,B])*tf.reshape(score,[1,B]))
            loss = tf.math.reduce_sum(tf.reshape(kernels,[H*W,B])*tf.reshape(score,[1,B])) + loss  

    return loss/denom




def self_paced_learning_loss(loss,lamda):
    shape = loss.shape
    temp = tf.zeros(shape, dtype=tf.dtypes.float32)
    temp2 = tf.ones(shape, dtype=tf.dtypes.float32)
    #print(temp)
    mask = tf.where(tf.less(loss,lamda),x=loss, y=temp)
    newloss = tf.math.reduce_sum(mask)/tf.math.reduce_sum(tf.dtypes.cast(tf.where(tf.less(loss,lamda),x=temp2, y=temp),tf.float32))
    return newloss,tf.where(tf.less(loss,lamda),x=temp2, y=temp)#tf.where(tf.less(loss,lamda*0.8),x=temp2, y=temp)