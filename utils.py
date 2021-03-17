#coding=utf-8
import tensorflow as tf
import os
import numpy as np
import cv2
import random
import imageio
slim = tf.contrib.slim
from PIL import Image



def convert_to_one_hot(y, C):
    return np.eye(C)[y.reshape(-1)].T



def load_image(path,IMAGE_SIZE):
    # load image and prepocess.
    img = imageio.imread(path)
    resized_img = Image.fromarray(img).resize(size=(IMAGE_SIZE, IMAGE_SIZE))
    resized_img = np.asarray(resized_img)

    #resized_img = misc.imresize(img, (IMAGE_SIZE, IMAGE_SIZE))
    #rotate_angle = random.choice(rotate_angle_list)
    #image = skimage.transform.rotate(resized_img, rotate_angle)
    images = np.asarray(resized_img, np.float32)
    return images/255.0

def load_image_test(path,IMAGE_SIZE):
    # load image and prepocess.
    img = imageio.imread(path)
    resized_img = Image.fromarray(img).resize(size=(IMAGE_SIZE, IMAGE_SIZE))
    resized_img = np.asarray(resized_img)

    #print(resized_img)
    #resized_img = misc.imresize(img, (IMAGE_SIZE, IMAGE_SIZE))
    if resized_img.shape[2] != 3:
        resized_img = resized_img[:, :, 0:3]
    #rotate_angle = random.choice(rotate_angle_list)
    #image = skimage.transform.rotate(resized_img, rotate_angle)
    images = np.asarray(resized_img, np.float32)
    return images/255

# convert label batch_size*imgsize*imgsize*1 to batch_size*imgsize*imgsize*2
def convert_label(labels_path,IMAGE_SIZE = 256):
    label_1 = imageio.imread(labels_path)
    label_1 = Image.fromarray(label_1).resize(size=(IMAGE_SIZE, IMAGE_SIZE))
    label_1 = np.asarray(label_1)


    #misc.imsave('1.png',label_1)
    label = generate_label(label_1)
    shape = label.shape
    #misc.imsave('3.png',label)
    temp = np.asarray(label,np.float32)/255.0
    #misc.imsave('4.png',temp*255)
    temp2 = 1-temp 
    #misc.imsave('4.png',temp2*255)
    temp = np.reshape(temp,[shape[0],shape[1],1])
    temp2 = np.reshape(temp2,[shape[0],shape[1],1])
    newlabel = np.concatenate((temp, temp2), axis=2)
    #print(newlabel)

    return newlabel


# convert label batch_size*imgsize*imgsize*1 to batch_size*imgsize*imgsize*2
def generate_label(label):
    shape = label.shape
    if len(shape) != 2:
        newlabel = label[:,:,0]
    else: newlabel = label
    #newlabel = newlabel/np.max(newlabel)*255
    
    #th = threshold*255
    _, newlabel = cv2.threshold(newlabel,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #_, newlabel = cv2.threshold(newlabel,170,255,cv2.THRESH_BINARY)
    #ret2,th2 = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    #misc.imsave('2.png',newlabel)
    
    return newlabel

def save_config(pre_path,path,name,config):
    info = 'information'+'\r\n'
    try:
        # adding exception handling
        with open(pre_path+'/'+'config.txt','r') as fq1:
            data = fq1.read()
        with open(os.path.join(path,name),"w") as f:
            f.write(config+"\r\n"+data) 
    except: 
        print('no previous config')
        with open(os.path.join(path,name),"w") as f:
            f.write(config+"\r\n") 



