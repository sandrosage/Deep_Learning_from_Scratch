import os.path
import json
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import zipfile
from skimage.transform import resize,rotate
from sklearn import utils
import random

# Also, note that in addition to the above slicing suggestion, NumPy provides the functions np.fliplr (left-right) and np.flipud (up-down).


# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):

        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.

        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle
        self.batch_index = 0
        self.rotations = [0,90,180,270]
        self.mirror_flags = [0,1]
        self.epoch_index = 0
        self.epoch_identifier = 0

        with open(self.label_path, 'r') as file:
            data = json.load(file)
            self.labels = {int(k):int(v) for k,v in data.items()}
            self.labels = list(dict(sorted(self.labels.items())).values())
            self.labels = np.array(self.labels)

        
        self.data_length = self.labels.shape[0]

        self.data = []
        for i in range(0, len(os.listdir(self.file_path))):
            np_array = np.load(os.path.join(self.file_path, (str(i) + ".npy")))
            self.data.append(np_array)
        self.data = np.array(self.data)
    
        if self.shuffle:
            self.data, self.labels = utils.shuffle(self.data, self.labels)


    def return_data(self):
        return self.data
    
    
    def return_labels(self):
        return self.labels
        

    def next(self):

        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases

        # print((self.batch_index),(self.batch_index+self.batch_size))
        if self.epoch_identifier:
            self.epoch_index = self.epoch_index + 1
            self.epoch_identifier = 0
        self.label_batch = self.labels[self.batch_index:(self.batch_index+self.batch_size)]
        self.data_batch = self.data[self.batch_index:(self.batch_index+self.batch_size)]
        if (self.batch_index + self.batch_size) >= self.data_length:
            self.label_batch = np.concatenate((self.label_batch, self.labels[0:(self.batch_index + self.batch_size - self.data_length)]), axis=0)
            self.data_batch = np.concatenate((self.data_batch, self.data[0:(self.batch_index + self.batch_size - self.data_length)]), axis=0)
            self.batch_index = 0
            self.epoch_identifier = 1
            if self.shuffle:
                self.data, self.labels = utils.shuffle(self.data, self.labels)

        else:
            self.batch_index = (self.batch_index+self.batch_size)
        images = []
        for img in self.data_batch:
            img = resize(img, self.image_size)
            img = self.augment(img)
            images.append(img)
        self.data_batch = np.array(images)


        return (self.data_batch,self.label_batch)
    

    def augment(self,img):

        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image

        if self.rotation:
            rotation_angle = random.choice(self.rotations)
            img = rotate(img, rotation_angle)
        if self.mirroring:
            mirror_flag = random.choice(self.mirror_flags)
            if mirror_flag:
                img = np.fliplr(img)
            else:
                img = np.flipud(img)

        return img
        
    def current_epoch(self):
        # return the current epoch number
        return self.epoch_index

    def class_name(self, x):

        # This function returns the class name for a specific input

        return self.class_dict[x]
    
    def show(self,columns):

        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.

        columns = columns
        rows = int(np.ceil(self.batch_size/columns))
        fig = plt.figure()
        (images,labels) = self.next()
        for (i),(img,label) in enumerate(zip(images,labels)):
            index = i + 1
            fig.add_subplot(rows, columns, index).set_title(self.class_name(label))
            plt.imshow(img)
        plt.subplots_adjust(hspace=1)
        plt.show()

