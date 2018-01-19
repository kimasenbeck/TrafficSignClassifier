# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognizer Project**

The goals / steps of this project were the following:
* Load the data set of German Traffic Sign images
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./writeup/visualization.png "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./new_images/30.jpg "Traffic Sign 1"
[image5]: ./new_images/left_turn.jpg "Traffic Sign 2"
[image6]: ./new_images/right.jpg "Traffic Sign 3"
[image7]: ./new_images/traffic_signal.jpg "Traffic Sign 4"
[image8]: ./new_images/60.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

This is it! Here is a link to my [project code.](https://github.com/kimasenbeck/TrafficSignClassifier/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

At the start of the project, I calculated summary statistics of the traffic
signs data set:

* The size of training set is 34799 photos
* The size of the validation set is 4410 photos
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a histogram showing how the occurrence of each of the labels in the test set. 

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

As a first step, I decided to convert the images to grayscale. Converting to grayscale allows the classifier to become more agnostic to lighting conditions. In other words, a red stop sign in bright light and a red stop sign in darkness could be classified as two different signs if we trained our model incorrectly. However, it's important that both stop signs, no matter the lighting condition, are classified correctly as such. 

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data in order to improve the accuracy of the gradient descent process. Since my model's learning rate needs to work for all of the given weights, it's important that the values fall in a given range. You can find a more thorough explanation of this on a conceptual level at [this link.](https://stats.stackexchange.com/questions/185853/why-do-we-need-to-normalize-the-images-before-we-put-them-into-cnn) 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.)

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					| 								|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6			 				|
| Convolution 3x3	    | 1x1stride,  valid padding, outputs 10x10x16      									|
| RELU					| 								|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16			 				|
|Dropout layer | 60% keep rate in training |
|Flattening layer | Input = 5x5x16. Output = 400 | 
| Fully connected		| Input = 400. Output = 12       									|
| RELU				|         									|
|	Fully connected			|Input = 120. Output = 84												|
| RELU					|												|
| Fully connected | Input = 84. Output = n_classes (43) | 
 
#### 3. Describe how you trained your model.

To train the model, I used TensorFlow's provided Adam optimizer, minimizing loss based on cross entropy. 
I trained on batches of size 128, and repeated the training over a series of 50 epochs with a learning rate of 0.01. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.9%
* validation set accuracy of 95.7%
* test set accuracy of 93.7%

I based my architecture off of the LeNet classifier. My classifier differs only in that I included a dropout layer, which the LeNet classifier did not utilize. 
I believed the Lenet classifier would be lend itself well to the traffic sign application, since it is originally an image classifier. It is considered to be the most powerful image processing model that has been invented, so I had confidence that it would work reasonably well applied to the German Traffic Sign dataset. 
The final model's accuracy on the training, validation and test set provides evidence that the model is working well, since it is consistently accurate, and classified the test data with only a small margin of error. 
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![Speed limit (30kmh)][image4] ![Turn left ahead][image5] ![Keep right][image6] 
![Traffic signals][image7] ![Speed limit (60kmh)][image8]

This set of images shouldn't be particuarly difficult to classify, but the images do differ noticeably from the provided images in the training, validation, and test sets. To be specific, my new images have much higher color contrast. The provided images all came from "the wild," so to speak, which is to say that they were extracted from actual images of signs on real roads. My images, in contrast, do not come from "the wild." This variation does not necessarily make them easier or harder to classify, but it does pose an interesting new challenge for the model. 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set. 

The model correctly classified all five traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set, where accuracy was only 93.7%. Granted, this was a significantly smaller sample size. This new set of images contained only five signs, whereas the test set contained 12630 images. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

Each of the images were successfully classified with fairly high probability. The table below shows the model's confidence in each of the signs' classifications. Notably, two of the images are classified with 100% probability, which may be overconfident. On the other hand, the 60kmh speed limit sign was classified with only 71% probability. 

| Confidence         	|     Sign	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Speed limit (30km/h)   									| 
| .94     				| Turn left ahead 										|
| 1.0					| Keep right											|
| .99	      			| Traffic signals					 				|
| .71			    | Speed limit (60km/h)      							|

Since the 60kmh speed limit sign had the lowest probability of all the results, it's worth taking a closer look at the softmax probabilities for this sign. Again, the classification was made with 71% confidence. However, you can see that the model also predicted the other speed limit signs with fairly high confidence. It's easy to imagine how the numbers 60, 30, and 50 are hard to distinguish from one another. Beyond the 3 speed limit sign predictions, the confidence drops significantly. In other words, the model was fairly certain that this sign was a speed limit sign, but it wasn't 100% sure which one it was. What's interesting to note that the 60kmh sign was more difficult to classify than the 30kmh sign, which was classified with 100% confidence. 

| Confidence         	|     Sign	        					| 
|:---------------------:|:---------------------------------------------:| 
| .71        			| Speed limit (60km/h)   									| 
| .17     				| Speed limit (30km/h) 										|
| .12					| Speed limit (50km/h)											|
| < .01	      			| Children crossing				 				|
| < .01			    | Go straight or right      							|

