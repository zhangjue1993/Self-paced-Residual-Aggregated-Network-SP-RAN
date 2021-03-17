from collections import deque
import tensorflow as tf
import os
import numpy as np
import cv2
from utils import *
import pickle
import os
from model import unet
from loss import *
from generate_data_list import generate_data_list
import time
from datetime import datetime
import shutil
#slim = tensorflow.contrib.slim
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_float('initial_learning_rate', 1e-3,'initial learning rate')
tf.app.flags.DEFINE_integer('batch_size', 16,'batch size')
tf.app.flags.DEFINE_integer('epoch', 400,'epoch')
tf.app.flags.DEFINE_integer('decay_ep', 20,'decay epoches')
tf.app.flags.DEFINE_float('decay_rate', 0.8,'decay rate')
tf.app.flags.DEFINE_string('train_label_dir', '/home/549/jz1585/PsudoLabel/conv4_3/','pesudo label for training')
tf.app.flags.DEFINE_string('ckpt_restore_dir', 'none','ckpt_restore_dir')


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

TRAIN_SET_DIR_solar = '/home/549/jz1585/Google-ACT-256/train/'
TRAIN_SET_DIR_label = FLAGS.train_label_dir
TEST_SET_DIR_label = '/home/549/jz1585/Google-ACT-256/test-gt'
TEST_SET_DIR_solar = '/home/549/jz1585/Google-ACT-256/test/'



IMAGE_SIZE = 256


generate_data_list(TRAIN_SET_DIR_solar,TRAIN_SET_DIR_label,TEST_SET_DIR_solar,TEST_SET_DIR_label)

localtime=time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
ckpt_save_dir = localtime


if not os.path.exists(ckpt_save_dir):      
    os.makedirs(ckpt_save_dir) 
    os.makedirs(ckpt_save_dir+'/logs')
    os.makedirs(ckpt_save_dir+'/model')



training_config = 'initial_learning_rate '+str(FLAGS.initial_learning_rate)+'\r\n'+'batch_size'+str(FLAGS.batch_size)+'\r\n'+'epoch '+str(FLAGS.epoch)+ '\r\n'+\
 'decay_ep '+str(FLAGS.decay_ep)+ '\r\n'+'decay_rate '+str(FLAGS.decay_rate)+'\r\n' \
 'train_label_dir '+str(FLAGS.train_label_dir) +'\r\n'+ 'test_image_dir'+TEST_SET_DIR_solar+ '\r\n'+ 'test_label_dir'+TEST_SET_DIR_label+'\r\n''ckpt_restore_dir '+\
  str(FLAGS.ckpt_restore_dir) 

save_config(FLAGS.ckpt_restore_dir, ckpt_save_dir,'config.txt',training_config)

try:
    with open('train_set_list.pickle', 'rb') as f:
            train_set_list = pickle.load(f)
except:
    raise EnvironmentError('Data list not existed. Please run generate_data_list.py first.')
random.shuffle(train_set_list)
train_set_queue = deque(train_set_list)
train_set_size = len(train_set_list)
for d in train_set_list:
    img = load_image(d[0],IMAGE_SIZE)
    #print(d[0],img.shape)
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
test_label_list = [convert_label(d[1]) for d in test_set_list]
test_label_list, test_image_list = np.array(test_label_list), np.array(test_image_list)
test_label_list, test_image_list = test_label_list[:FLAGS.batch_size,:,:,:],test_image_list[:FLAGS.batch_size,:,:,:]


del test_set_list
#print(test_label)
with tf.Graph().as_default():
    image = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, 3],name='input')
    label = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, 2],name = 'train_label')
    test_loss = tf.placeholder(tf.float32,name='loss_test')
    test_image = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, 3],name = 'test_image')
    test_label = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, 1],name = 'test_label')
    test_output = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, 1],name = 'test_output')

    Unet = unet(image, block_num = 4)
    #loss = weighted_cross_entropy(label, Unet.prediction)

    #y_hat = Unet.finalpre
    #_,h,w,_ = y_hat.shape
    #beta = tf.reduce_sum(label[:,:,:,0])/tf.cast(h*w,dtype=tf.float32) # ratio of foreground
    #a = (1-beta)* label[:,:,:,0] * tf.log(y_hat[:,:,:,0]+1e-5)
    #b =  beta* label[:,:,:,1] * tf.log((y_hat[:,:,:,1]+1e-5))
    #loss = -(tf.reduce_mean(a)+tf.reduce_mean(b))
    loss = weighted_cross_entropy(label, Unet.finalpre)
    global_step = tf.placeholder(tf.int32,name='global_step')
    decay_steps = FLAGS.decay_ep*train_set_size//FLAGS.batch_size
    lr=tf.train.exponential_decay(FLAGS.initial_learning_rate,global_step = global_step, decay_steps = decay_steps, decay_rate = FLAGS.decay_rate,staircase = 'True')
    # train_step = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)
    train_step = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(loss)
    optimizer = tf.train.RMSPropOptimizer(learning_rate=lr)
    grads, variables = zip(*optimizer.compute_gradients(loss))
    # grads, global_norm = tf.clip_by_global_norm(grads, 5)
    # train_step = optimizer.apply_gradients(zip(grads, variables))

    tf.summary.scalar('train_loss', loss)
    tf.summary.scalar('test_loss', test_loss)
    tf.summary.scalar('learning_rate', lr)
    tf.summary.image('train_images', image)
    tf.summary.image('test_images', test_image)
    tf.summary.image('labels', label[:,:,:,:1])
    tf.summary.image('test_label', test_label*255)
    tf.summary.image('output', test_output)
    tf.summary.image('finalpre', Unet.finalpre[:,:,:,:1])

    for var in tf.trainable_variables():
        tf.summary.histogram(var.name, var)
    # Summarize all gradients
    # grads = tf.gradients(loss, tf.trainable_variables())
    # grads = list(zip(grads, tf.trainable_variables()))

    for i in range(len(grads)):
        tf.summary.histogram(variables[i].name + '/gradient', grads[i])
    summary_step = tf.summary.merge_all()


    config = tf.ConfigProto(allow_soft_placement=True)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    config.gpu_options.allow_growth = True


    step = 0
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(ckpt_save_dir+'/logs', sess.graph)

    #saver = tf.train.Saver(max_to_keep=100)

        saver = tf.train.Saver(tf.global_variables(),max_to_keep = 100)
        # restore old checkpoint
        checkpoint = tf.train.get_checkpoint_state(FLAGS.ckpt_restore_dir+'model')
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
                print("Could not find old network weights")


        for ep in range(FLAGS.epoch):
            j = 0
            batch_num = train_set_size//FLAGS.batch_size
            random.shuffle(train_set_queue)
            for j in range(0,batch_num):
                minibatch = []
                for count in range(0, FLAGS.batch_size):
                    element = train_set_queue.pop()
                    minibatch.append(element)
                    train_set_queue.appendleft(element)
                #print(minibatch)
                image_list = [load_image(d[0],IMAGE_SIZE) for d in minibatch]
                label_list = [convert_label(d[1]) for d in minibatch]

                image_batch = np.array(image_list)
                label_batch = np.array(label_list)
                #print(image_list)
                #print(label_list)
                #print(image_batch.shape)

                image_batch = np.reshape(image_batch, [FLAGS.batch_size, IMAGE_SIZE, IMAGE_SIZE, 3])
                label_batch = np.reshape(label_batch, [FLAGS.batch_size, IMAGE_SIZE, IMAGE_SIZE, 2])
                

                _, train_loss= sess.run([train_step,loss], feed_dict = {image: image_batch, label: label_batch,global_step:step})
                
                #for key in layers.keys():
                    #print(layers[key]) 


                step = step +1            

                if step%10 ==0 and step>0:

                    loss_test,output = sess.run([loss,Unet.finalpre],  feed_dict = {image: test_image_list, label: test_label_list})
                    #print(output[1,:,:,1].shape)
                    #print(test_label_list[1,:,:,0].shape)
                    printstr = ('%s: step %d, train_loss = %.5f, test_loss = %5f')
                    print(printstr % (datetime.now(), step, train_loss, loss_test))
                    #print(step,train_loss, loss_test)
                    output = np.reshape(output[:,:,:,0],[-1,IMAGE_SIZE, IMAGE_SIZE,1])
                    test_labels = np.reshape(test_label_list[:,:,:,:1],[-1,IMAGE_SIZE, IMAGE_SIZE,1])
                    #print(test_label.shape)
                    summary = sess.run([summary_step],  feed_dict = {image: image_batch, label: label_batch, test_image: test_image_list, test_label:test_labels, test_loss: loss_test,global_step:step, test_output:output})
                    summary_writer.add_summary(summary[0], step)
            
            if ep % 20 == 0 and ep>0:

                saver.save(sess, ckpt_save_dir+'/model/'+str(ep)+'.ckpt')

                
        saver.save(sess, ckpt_save_dir+'/model/'+str(ep)+'.ckpt')