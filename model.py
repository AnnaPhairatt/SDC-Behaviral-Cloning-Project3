'''
    steering angle prediction using neural network
    Input: Image left/center/right
    Output: Steering angle (feed to the control)
    
'''

import csv
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import pickle
import random
import os

from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout,  ELU 
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import Callback, EarlyStopping
from keras.optimizers import Adam
from keras.preprocessing.image import *    #threading
from keras.utils.visualize_util import plot

#-------------------------------- Image data preprocessing functions --------------------------- #

# Crop and resize
def ImageResize(image, SIZE=(64,64)):
    '''
    Input  image: 160x320x3
    Output image: 64x64x3
    '''
    img = image[30:135, :, :]
    img = cv2.resize(img, SIZE, interpolation=cv2.INTER_AREA)
    return img

# Shift left/right and top bottom
def ImageShift(image, steering, shift_rc = [40,100]):
    # shift max range up/down (row), and left/right (column)
    tx = shift_rc[1] * (np.random.uniform() - 0.5)   # shift column (+,- range)
    ty = shift_rc[0] * (np.random.uniform() - 0.5)   # shift row (+,- range)
    M = np.float32([[1, 0, tx], [0, 1,ty]])
    image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    # we need to adjust steering angle after left/right shift :: 0.004/pixel :: 0.4*(1 pixel/100)
    steering = steering + 0.4*(tx/shift_rc[1])
    return image, steering
    
# Brightness adjustment
def AdjustBrightness(image):
    # Adjust random brightness
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    randBright = min(0.25+np.random.uniform(), 1.0)     # below 0.25 is too dark
    hsv[:,:,2] = hsv[:,:,2] * randBright
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return image

# Flip image horizontally
def ImageFlip(image,steering):
    return cv2.flip(image,1), -1.0*steering

# Add artificial shadow- optional/work without add artificial shadow provided training with track 2 data
# Codes from https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.kwwm9nr31
def AddShadow(image):
    top_y = 320*np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320*np.random.uniform()
    image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    shadow_mask = 0*image_hls[:,:,1]
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]

    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
    #random_bright = .25+.7*np.random.uniform()
    if np.random.randint(2)==1:
        random_bright = .5
        cond1 = shadow_mask==1
        cond0 = shadow_mask==0
        if np.random.randint(2)==1:
            image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
        else:
            image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright    
    image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)
    return image 

# Apply all transformation above to the image
def ProcessImage(image, steering, image_resize, image_shift):
    # flip image
    if np.random.random() < 0.5:
        image,steering = ImageFlip(image, steering)
    # add shadow    
    #if np.random.random() < 0.5:
        #image = AddShadow(image)
        
    # shift left/right, up/down
    image, steering = ImageShift(image, steering, shift_rc = image_shift)
    # adjust brightness
    image = AdjustBrightness(image)
    # crop to the target size
    image = ImageResize(image, SIZE=image_resize)
    return image, steering


# --------------------- Threads and call backs -------------------------------------------#
# code from srikanthpagadala https://github.com/srikanthpagadala/udacity/blob/master/
#                            Self-Driving Car Engineer Nanodegree/BehavioralCloning-P3
# and http://anandology.com/blog/using-iterators-and-generators/
class threadsafe_iter:
    """
        Takes an iterator/generator and makes it thread-safe by
        serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self
    
    def __next__(self):
        with self.lock:
            return self.it.__next__()
        
def threadsafe_generator(f):
    """
        A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g


class TrainingCallback(Callback):

    def on_epoch_begin(self, epoch, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
        # Lower Bias threshold for every epoch run
        global BiasThreshold
        BiasThreshold = 1 / (epoch + 1)

    def on_batch_begin(self, batch, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

    def on_train_begin(self, logs={}):
        self.losses = []

    def on_train_end(self, logs={}):
        pass
        

# --------------------------- Data generator  -------------------------------#
class DataTrainGenerator():
    def __init__(self, BatchSize, TestSplit, ImageResize, ImageShift, CamOffset):
        self.filepath       = None
        self.lines_train    = None
        self.lines_test     = None
        self.BatchSize      = BatchSize
        self.TestSplit      = TestSplit
        self.imageResize    = ImageResize
        self.imageShift     = ImageShift
        self.camOffset      = CamOffset
        self.SteerTrain     = []
               
        
    def ParseLogFile(self, filepath):
        # Reading log data from one or more csv file-  TODO using pandas 
        self.filepath = filepath
        lines = []
        with open(filepath + 'driving_log.csv') as csvfile:
            reader = csv.reader(csvfile)
            # each line in the csv
            for line in reader:
                lines.append(line)
                
        # save augment log data
        all_lines = np.array(lines)
        # shuffle the original data for data spliting
        np.random.shuffle(all_lines)
        
        # split train test data
        split_n = int(len(all_lines) * (1.0-self.TestSplit))
        self.lines_train = all_lines[0:split_n]
        self.lines_test  = all_lines[split_n:len(all_lines)]
        print("Training length = ", len(self.lines_train))
        print("Testing length  = ", len(self.lines_test))
            
    
    def GetImageData(self, line_data):
        # line_data order ['center', 'left', 'right', 'steering_angle', 'throttle', 'brake', 'speed']
        # randomly pick centre/left/right image
        i = np.random.randint(3)
        if (i == 0):
            path_file = self.filepath + 'IMG/' + line_data[0].split('/')[-1]
            cam_offset = self.camOffset
        elif (i == 1):
            path_file = self.filepath + 'IMG/' + line_data[1].split('/')[-1]
            cam_offset = 0.0
        elif (i == 2):
            path_file = self.filepath + 'IMG/'+ line_data[2].split('/')[-1]
            cam_offset = -self.camOffset
            
        steering_angle = float(line_data[3]) + cam_offset
        img = cv2.imread(path_file)
        #opencv read BGR format 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img, steering_angle = ProcessImage(img, steering_angle, self.imageResize, self.imageShift)
        return img, steering_angle
     
    @threadsafe_generator    
    def DataGenerator(self, DataSelection):
        # Train or Test
        if DataSelection == "Train":
            data_lines = self.lines_train
        else:
            data_lines = self.lines_test
            
        # Aloc batch
        w = self.imageResize[1]
        h = self.imageResize[0] 
        X_gen = np.zeros((self.BatchSize, h, w, 3))
        y_gen = np.zeros(self.BatchSize)    
        
        while 1:
            for i in range(self.BatchSize):
                # Control bias here - low steering won't be selected as much
                # In this loop, low angles will gradually pass through
                # Every the end of epoch - self.BiasThreshold will be reduced
                # Idea from https://github.com/srikanthpagadala/udacity/blob/master/Self-Driving Car Engineer Nanodegree/BehavioralCloning-P3
                keep = 0
                while keep == 0:
                    # ramdomly pick a data
                    index = np.random.randint(len(data_lines))
                    ImageData = data_lines[index]
                    steering = float(ImageData[3])
                    if abs(steering) < 0.1:
                        # This will determine if we want to let the low angle data through or not
                        val = np.random.uniform()
                        if val > BiasThreshold:
                            keep = 1
                    else:
                        keep = 1
                
                # Image and steering data 
                X_gen[i],y_gen[i] = self.GetImageData(ImageData)
                # Record what steering angles has been used to train the data
                self.SteerTrain.append(y_gen[i])          
            yield X_gen, y_gen
            
                     

#
#  Model
#
def Nvidia_Mod():
    model = Sequential()
    model.add( Lambda(lambda x: x/255.0 - 0.5, input_shape = (64,64,3)) )
    model.add(Convolution2D(24,3,3,subsample=(2,2),activation="relu"))      
    model.add(Convolution2D(36,3,3,subsample=(2,2),activation="relu"))
    model.add(Convolution2D(48,3,3,subsample=(2,2),activation="relu"))
    model.add(Convolution2D(64,3,3,activation="relu"))
    model.add(Convolution2D(64,3,3,activation="relu")) 
    
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Dropout(0.2))   
    model.add(Dense(64))     
    model.add(Dropout(0.5))   
    model.add(Dense(16))      
    model.add(Dense(1))
    opt = Adam(lr=0.001)     
    model.compile(optimizer=opt, loss='mse', metrics=[])
    return model

#
# Main
#
if __name__ == "__main__":
    # Exploring data set
    '''
    columns = ['center', 'left', 'right', 'steering_angle', 'throttle', 'brake', 'speed']
    data = pd.read_csv('./data_mix2/driving_log.csv', names=columns)
    print("Dataset Columns:", columns, "\n")
    print("Shape of the dataset:", data.shape, "\n")
    print(data.describe(), "\n")
    print("Data loaded...")
    
    # histogram before image augmentation
    binwidth = 0.025
    plt.hist(data.steering_angle,bins=np.arange(min(data.steering_angle), max(data.steering_angle) + binwidth, binwidth))
    plt.title('Number of images per steering angle')
    plt.xlabel('Steering Angle')
    plt.ylabel('# Frames')
    plt.show()
    '''
    
    n_batch  = 256          # default 256
    epochs   = 8            # default 8
    n_split  = 0.2          # split test data size
    img_resize = (64,64)    # image resize
    img_shift = [40,100]    # maximum shift by row and column
    cam_offset = 0.25       # Left/Right camera steering offset from the centre image
    BiasThreshold = 1.0     # init value for bias control every epoch (global variable)
    DataGen  = DataTrainGenerator(n_batch, n_split, img_resize, img_shift, cam_offset)
    
    # Parse the log csv data from this folder
    print("Parse log file ....")
    #DataGen.ParseLogFile('./data_mix2/')
    DataGen.ParseLogFile('./data_track1/')
    
    #
    # Build model
    #
    print("Build model ....")
    model   = Nvidia_Mod()
    training_callback = TrainingCallback()
    early_stop = EarlyStopping(monitor='val_loss', patience=2, verbose=0)
    callbacks = [early_stop, training_callback,]
    
    #
    # Train the model
    #
    print("Train model ....")
    # round to the size of n-batch so keras won't complain- Note 1 line has 3 images
    length_train = 3*DataGen.BatchSize * int(len(DataGen.lines_train)/DataGen.BatchSize)
    length_test  = 3*DataGen.BatchSize * int(len(DataGen.lines_test)/DataGen.BatchSize)
    DataTrain = DataGen.DataGenerator("Train")
    DataTest  = DataGen.DataGenerator("Test")
    
    model.fit_generator(DataTrain, samples_per_epoch = length_train, nb_epoch = epochs, verbose=1,\
                        validation_data=DataTest, nb_val_samples=length_test, callbacks=callbacks)
      
    # Save model configuration and weights        
    model_json = model.to_json()
    with open("./model.json", "w") as json_file:
        json.dump(model_json, json_file)
    model.save_weights("./model.h5")
    print("Saving model weights and configuration file.")   
        
        
    # trained steer data record
    binwidth = 0.025
    plt.hist(DataGen.SteerTrain,bins=np.arange(min(DataGen.SteerTrain), max(DataGen.SteerTrain) + binwidth, binwidth))
    plt.title('Number of images per steering angle')
    plt.xlabel('Steering Angle')
    plt.ylabel('# Frames')
    plt.show()
   
    # save model graph  
    #from keras.utils.visualize_util import plot
    #plot(model, to_file='model.png', show_shapes=True)
    
    exit()
    
    
    