#coding=utf-8
import tensorflow as tf
from vgg import vgg16
import os
import numpy as np
import cv2
import random
import imageio
from PIL import Image
slim = tf.contrib.slim
def get_data(classes, datadir):
    images = []
    labels = []
    with tf.Session() as sess:
        for i in range(len(classes)):
            image_dir = datadir + classes[i] + '/'
            for file_name in os.listdir(image_dir):
                # print file_name
                image = misc.imread(image_dir + file_name)
                image = misc.imresize(image,[256, 256])
                distorted_image = tf.image.random_brightness(image,max_delta=20)
                distorted_image = tf.image.random_contrast(distorted_image,
                                            lower=0.6, upper=1.2)
                label = i
                images.append(image)
                images.append(sess.run(distorted_image))
                labels.append(label)
                labels.append(label)
        images = np.asarray(images, np.float32)
        labels = np.asarray(labels, np.int32)
    return images, labels

def get_data_all(classes, datadir):
    images = []
    labels = []
    with tf.Session() as sess:
        for i in range(len(classes)):
            image_dir = datadir + classes[i] + '/'
            for file_name in os.listdir(image_dir):
                image = misc.imread(image_dir + file_name)
                image = misc.imresize(image,[256, 256])
                distorted_image = tf.image.random_brightness(image,max_delta=5)
                distorted_image = tf.image.random_contrast(distorted_image,
                                            lower=0.8, upper=1.1)
                label = i
                images.append(image)
                images.append(sess.run(distorted_image))
                labels.append(label)
                labels.append(label)
        images = np.asarray(images, np.float32)
        labels = np.asarray(labels, np.int32)
    return images, labels

def get_data_all_without_distor(classes, datadir):
    images = []
    labels = []
    for i in range(len(classes)):
        image_dir = datadir + classes[i] + '/'
        for file_name in os.listdir(image_dir):
            # print file_name
            image = misc.imread(image_dir + file_name)
            image = misc.imresize(image,[256, 256])
            label = i
            images.append(image)
            labels.append(label)
    images = np.asarray(images, np.float32)
    labels = np.asarray(labels, np.int32)
    return images, labels

def convert_to_one_hot(y, C):
    return np.eye(C)[y.reshape(-1)].T

def grad_cam(x, vgg, sess, predicted_class, layer_name, nb_classes):
    print("Setting gradients to 1 for target class and rest to 0")
    conv_layer = vgg.layers[layer_name]
    one_hot = tf.sparse_to_dense(predicted_class, [nb_classes], 1.0)
    signal = tf.multiply(vgg.layers['fc3'], one_hot)
    loss = tf.reduce_mean(signal)
    grads = tf.gradients(loss, conv_layer)[0]
    norm_grads = tf.div(grads, tf.sqrt(tf.reduce_mean(tf.square(grads))) + tf.constant(1e-5))
    output, grads_val = sess.run([conv_layer, norm_grads], feed_dict={vgg.imgs: x})
    output = output[0]           # [7,7,512]
    grads_val = grads_val[0]	 # [7,7,512]
    weights = np.mean(grads_val, axis = (0, 1)) 			# [512]
    cam = np.ones(output.shape[0 : 2], dtype = np.float32)	# [7,7]
    for i, w in enumerate(weights):
        cam += w * output[:, :, i]
    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam)
    cam = cv2.resize(cam, (256,256))

    #cam = Image.fromarray(cam).resize(size=(256, 256))
    #cam = np.asarray(cam)
    #cam[cam<0.2]=0
    cam3 = np.expand_dims(cam, axis=2)
    cam3 = np.tile(cam3,[1,1,3])
    return cam3

def grad_cam_plus_plus(x, vgg, sess, predicted_class, layer_name, nb_classes):
    label_index = tf.placeholder("int64", ())
    conv_layer = vgg.layers[layer_name]
    one_hot = tf.sparse_to_dense(predicted_class, [nb_classes], 1.0)
    signal = tf.multiply(vgg.fc3l, one_hot)
    cost = signal
    target_conv_layer_grad = tf.gradients(cost, conv_layer)[0]
    first_derivative = tf.exp(cost)[0][label_index]*target_conv_layer_grad
    second_derivative = tf.exp(cost)[0][label_index]*target_conv_layer_grad*target_conv_layer_grad 
    triple_derivative = tf.exp(cost)[0][label_index]*target_conv_layer_grad*target_conv_layer_grad*target_conv_layer_grad 
    # sess.run(init)
	# output = [0.0]*vgg.prob.get_shape().as_list()[1] #one-hot embedding for desired class activations
		#creating the output vector for the respective class
    output = [0.0]*vgg.probs.get_shape().as_list()[1]
    print output
    conv_output, conv_first_grad, conv_second_grad, conv_third_grad = sess.run([conv_layer, first_derivative, second_derivative, triple_derivative], feed_dict={vgg.imgs: x, label_index:0})
    global_sum = np.sum(conv_output[0].reshape((-1,conv_first_grad[0].shape[2])), axis=0)
    alpha_num = conv_second_grad[0]
    alpha_denom = conv_second_grad[0]*2.0 + conv_third_grad[0]*global_sum.reshape((1,1,conv_first_grad[0].shape[2]))
    alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, np.ones(alpha_denom.shape))
    alphas = alpha_num/alpha_denom
    weights = np.maximum(conv_first_grad[0], 0.0)
    alphas_thresholding = np.where(weights, alphas, 0.0)
    alpha_normalization_constant = np.sum(np.sum(alphas_thresholding, axis=0),axis=0)
    alpha_normalization_constant_processed = np.where(alpha_normalization_constant != 0.0, alpha_normalization_constant, np.ones(alpha_normalization_constant.shape))
    alphas /= alpha_normalization_constant_processed.reshape((1,1,conv_first_grad[0].shape[2]))
    deep_linearization_weights = np.sum((weights*alphas).reshape((-1,conv_first_grad[0].shape[2])),axis=0)
    grad_CAM_map = np.sum(deep_linearization_weights*conv_output[0], axis=2)
    cam = np.maximum(grad_CAM_map, 0)
    cam = cam / np.max(cam)
    cam = misc.imresize(cam, (256, 256))
    return cam

def sobel(img):
    sobelx=cv2.Sobel(img,cv2.CV_64F,dx=1,dy=0)
    sobelx=cv2.convertScaleAbs(sobelx)
    sobely=cv2.Sobel(img,cv2.CV_64F,dx=0,dy=1)
    sobely=cv2.convertScaleAbs(sobely)
    result=np.array(cv2.addWeighted(sobelx,0.5,sobely,0.5,0),np.float)/255.
    return result


def sigmoid(x):
    # TODO: Implement sigmoid function
    return 1/(1 + np.exp(-x))

def get_batch(batchsize, train_dir, gt_dir):
    ids = []
    Inputs = np.zeros([batchsize, 256, 256, 3],dtype=np.float)
    Targets = np.zeros([batchsize, 256, 256, 2] ,dtype=np.float)
    Smaps = np.zeros([batchsize, 256, 256, 1] ,dtype=np.float)
    Laps = np.zeros([batchsize, 256, 256, 1] ,dtype=np.float)
    T = np.zeros([batchsize, 256, 256, 2, 2] ,dtype=np.float)
    for file in os.listdir(gt_dir):
        # print file
        if file.endswith(".tif"):
            # print(file)
            ids.append(str(file))  # list of str
    ids = random.sample(ids, batchsize)
    # print(ids)
    for i,img_id in enumerate(ids):
        smap = cv2.imread(gt_dir+img_id, cv2.IMREAD_GRAYSCALE)
        lap = sobel(smap)
        _, gt = cv2.threshold(smap, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        gt = gt/255.0
        image = cv2.imread(train_dir+img_id)
        gt = np.array(gt, dtype=np.float32)
        # gt = gt[:, :, np.newaxis]
        image = np.array(image, dtype=np.float32) / 255.
        Inputs[i, :, :, :] = image
        smap = np.array(smap, dtype=np.float32)/255.
        Smaps[i, :, :, :] = np.expand_dims(smap, 2)
        # Targets[i, :, :, 0] = smap
        # Targets[i, :, :, 1] = 1.-smap
        Targets[i, :, :, 0] = gt
        Targets[i, :, :, 1] = 1.-gt
        Laps[i, :, :, :] = np.expand_dims(lap, 2)
        T00 = (1-lap) * smap
        T11 = (1-lap) * (1-smap)
        # T[i, :, :, 0, 0] = T00
        # T[i, :, :, 1, 0] = 1 - T00
        # T[i, :, :, 0, 1] = 1 - T11
        # T[i, :, :, 1, 1] = T11

        # T[i, :, :, 0, 0] = T00
        # T[i, :, :, 1, 0] = 1 - T11
        # T[i, :, :, 0, 1] = 1 - T00
        # T[i, :, :, 1, 1] = T11

        T[i, :, :, 0, 0] = 2.*np.abs(smap-0.5)
        T[i, :, :, 1, 0] = 1 - 2.*np.abs(smap-0.5)
        T[i, :, :, 0, 1] = 1 - 2.*np.abs(smap-0.5)
        T[i, :, :, 1, 1] = 2.*np.abs(smap-0.5)

        # T[i, :, :, 0, 0] = 1.
        # T[i, :, :, 1, 0] = 0.
        # T[i, :, :, 0, 1] = 0.
        # T[i, :, :, 1, 1] = 1.
    return Inputs, Targets, Smaps, Laps, T

def predict(model, image_dir, gt_dir):
    print('predicting in files: '+gt_dir)
    ids = []
    with tf.Session() as sess:
        for file in os.listdir(image_dir):
            if file.endswith(".tif"):
                ids.append(str(file))  # list of str
        for i,img_id in enumerate(ids):
            image = cv2.imread(image_dir+img_id)
            image = np.array(image, dtype=np.float32) / 255.
            image = np.expand_dims(image, axis=0)
            image = sess.run([model.sig], feed_dict={model.input: image})

def top_k_loss(model, labels, k, batchsize):
    threshold = np.trunc(k*256*256*batchsize)
    loss_map = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=model.logist)
    loss_map_vector = tf.reshape( loss_map, [1,256 * 256 * batchsize])
    loss_map_top_k = tf.nn.top_k(-1*loss_map_vector, threshold)
    predict_loss_map_top_k = -1*tf.reduce_mean(loss_map_top_k[0])
    # predict_loss_map_top_k = tf.reduce_mean(loss_map)
    return predict_loss_map_top_k,loss_map

def content_loss(img, labels, model):
    a = tf.multiply(tf.tile(labels, [1, 1, 1, 3]), img)
    b = tf.multiply(tf.tile(model.sig, [1, 1, 1, 3]), img)
    return tf.reduce_mean(tf.square(a - b))

def dilated_conv2D_layer(inputs,num_outputs,kernel_size,activation_fn,rate,padding,weights_regularizer,scope):
    '''
    使用Tensorflow封装的空洞卷积层ＡPI:包含空洞卷积和激活函数，但不包含池化层
    :param inputs:
    :param num_outputs:
    :param kernel_size:　卷积核大小，一般是[1,1]，[３,３]，[５,５]
    :param activation_fn:激活函数
    :param rate:
    :param padding: SAME or VALID
    :param scope:　scope name
    :param weights_regularizer:正则化,例如：weights_regularizer = slim.l2_regularizer(scale=0.01)
    :return:
    '''
    with  tf.variable_scope(name_or_scope=scope):
        in_channels = inputs.get_shape().as_list()[3]
        kernel=[kernel_size[0],kernel_size[1],in_channels,num_outputs]
        # filter_weight=tf.Variable(initial_value=tf.truncated_normal(shape,stddev=0.1))
        filter_weight = slim.variable(name='weights',
                                      shape=kernel,
                                      initializer=tf.truncated_normal_initializer(stddev=0.1),
                                      regularizer=weights_regularizer)
        bias = tf.Variable(tf.constant(0.01, shape=[num_outputs]))
        # inputs = tf.nn.conv2d(inputs,filter_weight, strides, padding=padding) + bias
        inputs = tf.nn.atrous_conv2d(inputs, filter_weight, rate=rate, padding=padding) + bias
        if not activation_fn is None:
            inputs = activation_fn(inputs)
        return inputs

def load_image(path,IMAGE_SIZE):
    # load image and prepocess.
    img = imageio.imread(path)
    #sess = tf.Session() 
        
    #image = resized_img

    #distorted_image = tf.image.random_brightness(image,max_delta=5)
    #distorted_image = tf.image.random_contrast(distorted_image, lower=0.8, upper=1.1)

    #img = sess.run(distorted_image)
    #rotate_angle = random.choice(rotate_angle_list)
    #image = skimage.transform.rotate(resized_img, rotate_angle)
    #images = np.asarray(img, np.float32)
    #sess.close()
    #print(resized_img)
    return img

def load_image_test(path,IMAGE_SIZE):
    # load image and prepocess.
    img = imageio.imread(path)
    #print(path,img.shape)

    #rotate_angle = random.choice(rotate_angle_list)
    #image = skimage.transform.rotate(resized_img, rotate_angle)
    return img 
