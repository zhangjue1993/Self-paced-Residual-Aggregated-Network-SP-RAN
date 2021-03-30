from collections import deque
import tensorflow as tf
import os
import imageio
import numpy as np
import cv2
from utils import *
import pickle
import os
import random
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


epoch = 100
batch_size = 16
lr = 1e-4
istrain = True
result_dir = './result'
IMAGE_SIZE = 256

try:
    with open('train_set_list.pickle', 'rb') as f:
            train_set_list = pickle.load(f)
except:
    raise EnvironmentError('Data list not existed. Please run generate_data_list.py first.')
random.shuffle(train_set_list)
train_set_queue = deque(train_set_list)
train_set_size = len(train_set_list)
del train_set_list

print ('Training set built. Size: '+str(train_set_size))


try:
    with open('test_set_list.pickle', 'rb') as f:
            test_set_list = pickle.load(f)
except:
    raise EnvironmentError('Data list not existed. Please run generate_data_list.py first.')
random.shuffle(test_set_list)
test_set_queue = deque(test_set_list)
test_set_size = len(test_set_list)


print ('Testing set built. Size: '+str(test_set_size))

test_image_list = [load_image_test(d[0],IMAGE_SIZE) for d in test_set_list]
test_label_list = [d[1] for d in test_set_list]
#print(len(test_image_list[0][0][0]),len(test_label_list))
test_label_list, test_image_list = np.array(test_label_list), np.array(test_image_list)
#test_image_list = np.asarray(test_image_list)
test_label = convert_to_one_hot(test_label_list, 2).T
del test_set_list
#print(test_label)

image = tf.placeholder(tf.float32, [None, 256, 256, 3])
label = tf.placeholder(tf.int16, [None, 2])
test_loss = tf.placeholder(tf.float32)

tf.summary.scalar('test_loss',test_loss)
tf.summary.image('train_image', image)


#distorted_image = tf.image.random_brightness(image,max_delta=5)
#distorted_image = tf.image.random_contrast(distorted_image, lower=0.8, upper=1.1)
#tf.summary.image('distorted_image', distorted_image)

sess = tf.Session()
vgg = vgg16(image)



loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=vgg.fc3l, labels=label))
# train_step = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)
train_step = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(loss)
optimizer = tf.train.RMSPropOptimizer(learning_rate=lr)
grads, variables = zip(*optimizer.compute_gradients(loss))
# grads, global_norm = tf.clip_by_global_norm(grads, 5)
# train_step = optimizer.apply_gradients(zip(grads, variables))

accuracy = tf.reduce_mean(tf.cast(tf.equal(vgg.prediction, tf.argmax(label, 1)), tf.float32))

tf.summary.scalar('acc', accuracy)
tf.summary.scalar('loss', loss)
for var in tf.trainable_variables():
    tf.summary.histogram(var.name, var)
# Summarize all gradients
# grads = tf.gradients(loss, tf.trainable_variables())
# grads = list(zip(grads, tf.trainable_variables()))
for i in range(len(grads)):
    tf.summary.histogram(variables[i].name + '/gradient', grads[i])
summary_step = tf.summary.merge_all()

saver = tf.train.Saver(max_to_keep=100)
model_path = './model/'
kk = 0

#images, labels = get_data_all(classification_class, data_dir)
#test_image, test_label = get_data_all_without_distor(classification_class, val_dir)
#labels = convert_to_one_hot(labels, 2).T
#test_label = convert_to_one_hot(test_label, 2).T
#config = tf.ConfigProto(allow_soft_placement=True)
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
#config.gpu_options.allow_growth = True


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    checkpoint = tf.train.get_checkpoint_state('./model-ACT')

    if checkpoint and checkpoint.model_checkpoint_path:
        print('Restoring model...')

        sess.run(tf.global_variables_initializer())
        saver.restore(sess, checkpoint.model_checkpoint_path)

        print('Restoration complete.')


    summary_writer = tf.summary.FileWriter('./log',sess.graph)
    for ep in range(epoch):
        j = 0
        random.shuffle(train_set_queue)
        batch_num = train_set_size//batch_size
        for j in range(0,batch_num):
            minibatch = []
            for count in range(0, batch_size):
                element = train_set_queue.pop()
                
                minibatch.append(element)
                train_set_queue.appendleft(element)
            
            image_list = [load_image(d[0],IMAGE_SIZE) for d in minibatch]
            label_list = [d[1] for d in minibatch]
            
            image_batch = image_list
            image_batch = np.array(image_list)
            #print(image_batch.shape)
            label_batch = np.array(label_list)
            #print(label_batch.shape)
            image_batch = np.reshape(image_batch, [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3])
            label_batch = np.reshape(label_batch, [batch_size])
            #print(image_batch.shape)
            #print(label_batch.shape)
            
            label_batch_new = convert_to_one_hot(label_batch, 2).T
            #print(label_batch_new.shape)
            _,train_acc, train_loss = sess.run([train_step,accuracy,loss], feed_dict = {image: image_batch, label: label_batch_new})
            kk = kk+1
            if j % 10 ==0:
                print('epoch%d, step:%d acc:%f, loss:%f' %(ep, kk, train_acc, train_loss))
            
            if j % 50 ==0:
                acc, loss_value = sess.run([accuracy, loss],  feed_dict = {image: test_image_list, label: test_label})
                summary = sess.run(summary_step,  feed_dict = {image: image_batch, label: label_batch_new, test_loss:loss_value})
                print('testing information : epoch%d, step:%d, test_acc:%f, test_loss:%f' %(ep, kk, acc, loss_value))
                summary_writer.add_summary(summary, kk)
        
        if ep % 10 ==0:
            saver.save(sess, model_path+'model_'+str(ep)+'.ckpt')
    saver.save(sess, model_path+'model_'+str(ep)+'.ckpt')