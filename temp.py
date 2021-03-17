import numpy as np
from scipy import misc

IMAGE_SIZE= 256
labels_path = '/root/VGG-ACT-add/result_train_gradcam/conv4_3/1-190.png'
label_1 = misc.imread(labels_path)
  
label = label_1/float(np.max(label_1))
print(np.max(label_1))
mask1 = np.where(label < 0.4, 1, 0)
mask2 = np.where(label > 0.8, 1, 0)
mask = mask1+mask2
misc.imsave('1.png',mask*255)
misc.imsave('2.png',label*255)
