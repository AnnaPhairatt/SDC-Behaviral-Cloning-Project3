# **Project 3: Use Deep Learning to Clone Driving Behavior** 

#### Project submission
------------------------    
**Mode**: Track1(Simple)- Constant speed  
**Input**: Images  
**Output**: Steering angle
(see [codes](https://github.com/LukePhairatt/SDC-Behaviral-Cloning-Project3/blob/master/model.py))

#### Exploration work  
------------------------
**Mode**: Track2(Twisted and turn corners with up/down hills)-Constant speed  
**Input**: Images  
**Output**: Steering angle
(see [codes](https://github.com/LukePhairatt/SDC-Behaviral-Cloning-Project3/tree/master/track2))

**Mode**: Track1- Variable speed drive  
**Input**:Images  
**Output**: Steering angle and Throttle
(see [codes](https://github.com/LukePhairatt/SDC-Behaviral-Cloning-Project3/tree/master/track1_racing))


The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one & two without leaving the road
* Summarize the results with a written report
* My exploration work includes both Steering and Throttle learning.  
  This is my first prototype to test out the idea of using deep neural netorks for multiple outputs. I only tried on Track 1.  
  The result was not too shabby. At least the car stayed on the track with the variable speed (upto 30 mph) which I trained it to drive!  
  However, some what improvement is needed. The car might struggle to go uphills on start-up or incredible speed down hills.  
  One could use throttle information to adjust the speed accordingly to the tracks.


[//]: # (Image References)

[image1]: ./examples/model.png "Model Visualization"
[image2]: ./examples/recovery1.jpg "Recovery Image1"
[image3]: ./examples/recovery2.jpg "Recovery Image2"
[image4]: ./examples/recovery3.jpg "Recovery Image3"
[image5]: ./examples/bias.png "Zero steering angle bias"
[image6]: ./examples/center.png "Normal Image"
[image7]: ./examples/flip.png "Flipped Image"
[image8]: ./examples/brightness.png "Adjust brightness"
[image9]: ./examples/shift.png "Shifted image"
[image10]: ./examples/resize.png "Resized image"

---
#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* model.json containing training weights 
* writeup_report.pdf summarizing the results

#### 2. Running code

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.json
```

#### 3. Model code

The model.py file contains the code for training and saving the convolution neural network and training weights (model.h5, model.json).  
The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.  
The batch generater is programmed with a thread safe.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of 5 convolution neural networks and 3 fully connected layers (slightly modified from Nvidia model).  
The model includes RELU layers, and the data is normalized in the model using a Keras lambda layer. 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting and using Early stopping callback.   
The model was trained and validated on different data sets to ensure that the model was not overfitting.  
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

![Model plot][image1] 
_Graph model_

#### 3. Model parameter tuning

The model used an adam optimizer with the initial learning rate of 0.001. 8 Epoch, 256 Batch size

#### 4. Training data

Two set of driving behaviors have been acquired and combined for the training and validation the model. 
* First is to teach the model with good driving style by keeping it on the center. 
* Another is to teach the car to recover to the middle when it wheel off course to left or right

![Recovery Image1][image2]
![Recovery Image2][image3]
![Recovery Image2][image4]
_Recovery learning images_

The car is steering from the right to the centre for learning recovery.

In addition, the data was visualised to see any sign of bias. It was seen that the driving data is bias around the zero steering angles  
and rather negative. This will have an impact in driving straight and turninng left. So as to avoid this,  
I used the sampling technique to control amount of low angles data to go through the each training loop (model.py lines 239-251).

![Bias data][image5]
_Bias data near zero angles_


#### 5. Solution Design Approach

My first step was to use a convolution neural network model similar to the Nvidia. I thought this model might be appropriate because it has a good number of layers but not too complex to learning the driving task.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.  
To combat the overfitting, I modified the model with two drop out layers.

The final step was to run the simulator to see how well the car was driving around track one. However, there are a few corners that the car nearly fell off the track. So to obtain more training data and generalise the training even more, I augmented training data with brightness adjustment, flip, adding left-right images and also getting additional data on recovery driving. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 6. Training Set & Training Process
Track 2 is more challenging than track 1 with rather steep up and down hills, sharp bends, zig zag  
road and shadow overcast. To help generalise the driving in this condition without using a lot of  
data, I augmented the training data with image up/down, left/right translation with a corrected  
steering shift of 0.004 radian/pixel, and also randomly adjusting the brightness.

Image preprocessing is shown below.
![Normal image][image6] _Original image_
![Flipped][image7] _Flipped image_
![Brightness][image8] _Change brightness_
![Shifted][image9] _Shifted image_
![Resized][image10] _Resized image_

Having collected all data, I randomly shuffled and splited the data 80% training and 20% testing. 
During the training process, the left/centre/right images is randomly and uniformly selected for image augmentation and training.  

For the training process, I had 7680,31488 number of images for track 1 and 2 respectively. 
The model was successfully trained with the initial learning rate of 0.001. 8 Epoch, 256 Batch size on both tracks.


#### 7. Variable speed drive- see [here](https://github.com/LukePhairatt/SDC-Behaviral-Cloning-Project3/tree/master/track1_racing)
* I used the same data on track 1 to teach the car to drive with the speed as I did on the simulator. 
* The work is primitive and needed to be refined further. 
* I added throttle data to the network in order to get the prediction on the throttle as well as steering. 
* The aim was to let the car learn when to go fast or slow in the straight line, conering, up hills or down hills. 
  








 

