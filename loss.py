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

def _cross_entropy(label, y_hat):
    _,h,w,_ = y_hat.shape
    beta = tf.reduce_sum(label[:,:,:,0])/tf.cast(h*w,dtype=tf.float32) # ratio of foreground
    a = (1-beta)* label[:,:,:,0] * tf.log(y_hat[:,:,:,0]+1e-5)
    b =  beta* label[:,:,:,1] * tf.log((y_hat[:,:,:,1]+1e-5))
    loss = -(tf.reduce_mean(a)+tf.reduce_mean(b))
    return loss