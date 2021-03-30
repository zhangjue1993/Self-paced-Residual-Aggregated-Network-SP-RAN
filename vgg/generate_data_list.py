""" Generate lists containing filepaths and labels for training, validation and evaluation. """
import pickle
import os.path
import random

##### generate training set #####
TRAIN_SET_DIR = '/scratch/po21/jz1585/Google-ACT-256/train-new1202/train-400/'

train_set_list = []
pos_num = 0
neg_num = 0
# negative samples
for i in range(1, 5000):
    img_path = os.path.join(TRAIN_SET_DIR, '0-'+ str(i)+'.png')
    if  os.path.exists(img_path):
        train_set_list.append((img_path, [0]))
        neg_num += 1

# positive samples
for i in range(1, 5000):
    img_path = os.path.join(TRAIN_SET_DIR, '1-'+ str(i)+'.png')
    if  os.path.exists(img_path):
        train_set_list.append((img_path, [1]))
        pos_num += 1

# positive samples
for i in range(1, 5000):
    img_path = os.path.join(TRAIN_SET_DIR, 'aux-'+ str(i)+'.png')
    if  os.path.exists(img_path):
        train_set_list.append((img_path, [1]))
        pos_num += 1

random.shuffle(train_set_list)
print(train_set_list)

with open('train_set_list.pickle', 'wb') as f:
    pickle.dump(train_set_list, f)

print ('Train set list done. # positive samples: '+str(pos_num)+' # negative samples: '+str(neg_num))

##### generate test set #####
TEST_SET_DIR = '/scratch/po21/jz1585/Google-ACT-256/test/'
test_set_list = []
pos_num = 0
neg_num = 0
# negative samples
for i in range(1, 66000):
    img_path = os.path.join(TEST_SET_DIR, '0-'+ str(i)+'.png')
    if  os.path.exists(img_path):
        test_set_list.append((img_path, [0]))
        neg_num += 1

# positive samples
for i in range(1, 66000):
    img_path = os.path.join(TEST_SET_DIR, '1-'+ str(i)+'.png')
    if  os.path.exists(img_path):
        test_set_list.append((img_path, [1]))
        pos_num += 1


random.shuffle(test_set_list)
print(test_set_list)
with open('test_set_list.pickle', 'wb') as f:
    pickle.dump(test_set_list, f)

print ('Test set list done. # positive samples: '+str(pos_num)+' # negative samples: '+str(neg_num))

