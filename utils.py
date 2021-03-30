#coding=utf-8
import tensorflow as tf
import os
import numpy as np
import cv2
import random
import imageio
from collections import deque
import shutil
slim = tf.contrib.slim
from PIL import Image



def convert_to_one_hot(y, C):
    return np.eye(C)[y.reshape(-1)].T



def load_image(path,IMAGE_SIZE=256):
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
    #print(path,img.shape)
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
def convert_label(labels_path, IMAGE_SIZE = 256):
    label_1 = imageio.imread(labels_path)
    label_1 = Image.fromarray(label_1).resize(size=(IMAGE_SIZE, IMAGE_SIZE))
    label = np.asarray(label_1)


    #misc.imsave('1.png',label_1)
    
    label = generate_label(label)


    #misc.imsave('3.png',label)
    temp = np.asarray(label,np.float32)/255.0
    shape = label.shape
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

def generate_initial_label(labelpath,initial_label_dir):

    labelsname = os.listdir(labelpath)
    for labelname in labelsname:
        if labelname[0]=='1':
            label = imageio.imread(os.path.join(labelpath,labelname))
            shape = label.shape
            if len(shape) != 2:
                newlabel = label[:,:,0]
            else: newlabel = label
            _, gt = cv2.threshold(newlabel,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            imageio.imwrite(os.path.join(initial_label_dir,labelname),gt)
    
def generate_new_label(label,labelname, label_update_dir):

    imageio.imwrite('1.png',label*255)
    label = imageio.imread('1.png')
    _, gt = cv2.threshold(label,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    newlabel = cv2.morphologyEx(gt, cv2.MORPH_OPEN, kernel)
    imageio.imwrite(os.path.join(label_update_dir,labelname),newlabel)

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



def generate_kenerl_mask(keneral_radius):
    mask = np.ones([(2*keneral_radius+1),(2*keneral_radius+1)],dtype=np.float32)
    mask[keneral_radius,keneral_radius]=0

    return mask

def choose_sample(sess, train_list,batch_size=32,image_size=256):
    new_train_list=[]
    batch_num = len(train_list)//batch_size
    step=0
    train_set_queue = deque(train_list)
    for j in range(0,batch_num):
        minibatch = []
        for count in range(0, batch_size):
            element = train_set_queue.pop()
            minibatch.append(element)
            train_set_queue.appendleft(element)

        image_list = [load_image(d[0],image_size) for d in minibatch]
        label_list = [convert_label(d[1]) for d in minibatch]

        image_batch = np.array(image_list)
        label_batch = np.array(label_list)

        image_batch = np.reshape(image_batch, [batch_size, image_size, image_size, 3])
        label_batch = np.reshape(label_batch, [batch_size, image_size, image_size, 2])
    
        mask = sess.run(vi, feed_dict = {image: image_batch})

        for index in range(0,batch_size):
            if mask[index]==1:
                new_train_list.append(image_list[index])

    return new_train_list

# def generate_new_label(label_save_dir, image_path):
    
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
#     temp = imageio.imread(os.path.join(label_save_dir,image_path))
#     _, newlabel = cv2.threshold(temp,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#     erosion = cv2.morphologyEx(newlabel, cv2.MORPH_OPEN, kernel)
#     dilation = cv2.dilate(erosion,cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
#     imageio.imsave(os.path.join(label_save_dir,image_path),dilation)

def copy_label(old_label_dir, new_label_dir):
    target = new_label_dir
    for filename in os.listdir(old_label_dir):
        source = os.path.join(old_label_dir,filename)
        shutil.copy(source, target)




def label_update(new_train_list, new_label_dir,batch_size,image_size,sess,image, label, kenerl_mask, global_step, global_ep,model,ep,step,mask,train_set,new_train_set_size):
    new_train_set_size = len(new_train_list)
    print(new_train_set_size)
    train_set_queue = deque(new_train_list)
    batch_num = new_train_set_size//batch_size

    for iter in range(0,batch_num):
        minibatch = []
        for count in range(0, batch_size):
            element = train_set_queue.pop()
            minibatch.append(element)
            train_set_queue.appendleft(element)

        image_list = [load_image(d[0]) for d in minibatch]
        label_list = [convert_label(d[1]) for d in minibatch]

        image_batch = np.array(image_list)
        label_batch = np.array(label_list)

        image_batch = np.reshape(image_batch, [batch_size, image_size, image_size, 3])
        label_batch = np.reshape(label_batch, [batch_size, image_size, image_size, 2])
    
        results = sess.run(model.finalpre[:,:,:,0], feed_dict = {image: image_batch, label: label_batch,kenerl_mask: mask,global_step:step,global_ep:ep,train_set:new_train_set_size})

        for index in range(0,batch_size):
            imageio.imsave(os.path.join(new_label_dir,os.path.basename(minibatch[index][0])),results[index,:,:])
            generate_new_label(new_label_dir, os.path.basename(minibatch[index][0]))


def sample_select(train_set_list,old_label_dir,batch_size,image_size,sess,image, label, kenerl_mask, global_step, global_ep,model, vi, ep,step,mask,train_set,new_train_set_size):
    new_train_list=[]
    batch_num = len(train_set_list)//batch_size
    train_set_queue = deque(train_set_list)
    for iter in range(0,batch_num):
        minibatch = []
        for count in range(0, batch_size):
            element = train_set_queue.pop()
            minibatch.append(element)
            train_set_queue.appendleft(element)

        image_list = [load_image(d[0],image_size) for d in minibatch]

        label_dir = old_label_dir
        label_list = [convert_label(os.path.join(label_dir,os.path.basename(d[1])),label_update=True) for d in minibatch]

        image_batch = np.array(image_list)
        label_batch = np.array(label_list)

        image_batch = np.reshape(image_batch, [batch_size, image_size, image_size, 3])
        label_batch = np.reshape(label_batch, [batch_size, image_size, image_size, 2])
        #print(image_batch.shape,label_batch.shape)
    
        results,samples = sess.run([model.finalpre[:,:,:,0],vi], feed_dict = {image: image_batch, label: label_batch,kenerl_mask: mask,\
            global_step:step,global_ep:ep,train_set:new_train_set_size})

        for index in range(0,batch_size):
            if samples[index]==1:
                new_train_list.append(minibatch[index])

    return new_train_list