# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup/histogram.png "Visualization"
[image2]: ./writeup/examples.png "Examples"
[image3]: ./writeup/hist_eq.png "Histogram Equalization"
[image4]: ./images/1.png "Traffic Sign 1"
[image5]: ./images/2.png "Traffic Sign 2"
[image6]: ./images/3.png "Traffic Sign 3"
[image7]: ./images/4.png "Traffic Sign 4"
[image8]: ./images/5.png "Traffic Sign 5"
[image9]: ./writeup/model.png "Model"
[image10]: ./writeup/top5.png "Top 5 Probabilities"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/tdb-alcorn/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34,799 images
* The size of the validation set is 4,410 images
* The size of test set is 4,410 images
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

![alt text][image2]

Below is a bar chart showing how the data is distributed across the labels in
each of the training, validation and test sets. It is clear that the labels
are distributed similarly in each data set, so we don't need to worry about
dealing with bias in the training set.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

My preprocessing pipeline was:

1. Convert to YUV color space, 
2. Histogram equalize the Y channel (luminance) to boost contrast.
3. Normalize the data using the mean and stddev of the training set.

I decided to convert to YUV color space since it is closer to human eye
perception and should carry more semantic information, especially on images
of artifacts that are intended to be easy for humans to see, like traffic
signs. I used histogram equalization because it increases contrast which
should help the algorithm distinguish signs across a wider variety of
lighting conditions. I used data normalization because it is well known that
neural networks perform better on normalized data, since it reduces the
magnitude of the transformations the network needs to perform to compute
output probabilities from input data.

Below is an example of a training image after histogram equalization,
exhibiting increased contrast.

![alt text][image3]


I decided to generate additional data because I found that the network was
reaching a plateau of validation accuracy as it saturated what it could 
learn from the original training data set, which consisted of only ~30,000
images. To add more data to the the data set, I added an augmentation
step to the input data pipeline, which randomly performed rotations, rescalings,
crops and flip operations. This augmentation was performed at model training
time using the Tensorflow Dataset API.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model was roughly based on LeNet, and consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 32x32x8 	|
| RELU					|												|
| Dropout				| Keep 50%										|
| Max pooling	      	| 2x2 stride,  outputs 16x16x8   				|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 16x16x16 	|
| RELU					|												|
| Dropout				| Keep 50%										|
| Max pooling	      	| 2x2 stride,  outputs 8x8x16   				|
| Flatten               | Outputs 1024                                  |
| Fully connected		| 128 hidden units        						|
| RELU					|												|
| Dropout				| Keep 50%										|
| Fully connected		| 64 hidden units        						|
| RELU					|												|
| Dropout				| Keep 50%										|
| Fully connected		| 43 hidden units        						|
| Softmax				| This represents the output probabilities		|
 
Tensorboard visualization of model architecture:
![alt text][image9]


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used Tensorflow's Adam optimizer with default parameters. I used 300 epochs with a batch size varying from 100 up to 5000 for the later epochs. The model was trained on an AWS p2.xlarge instance (Nvidia Tesla K80 gpu)

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:

* training set accuracy of 92.7%
* validation set accuracy of 93.8%
* test set accuracy of 93.1%

Since the training, validation and test set accuracies are all roughly equal, I conclude that the model is not overfitting
and will generalize well to new data.

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

I came up with the basic architecture based on LeNet and added dropout layers to prevent overfitting. I experimented with
adding a third convolutional layer but found that it did not help much and was much slower to converge, so I stuck
with the simple architecture in the end. I chose LeNet because it has performed well on this exact task before. 
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The images were screenshotted from views of Cologne, DE in Google Street View.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Ahead Only      		| Ahead Only   									| 
| No entry     			| No entry 										|
| Yield					| Yield											|
| Roundabout mandatory	| Speed limit (120km/h)			 				|
| Bicycles crossing 	| Keep right         							|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This compares poorly to
the accuracy on the test set of 93.1%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell
of the Ipython notebook.

The top 5 probabilities are as follows:

![alt text][image10]

For example, in the first image, the model is very confident that this is
'Ahead Only' (probability of 0.91), and the image does indeed contain 'Ahead Only'.
The top five soft max probabilities, as shown in the image above, are

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .91         			| Ahead Only   									| 
| .07     				| Turn right ahead 								|
| .01					| Turn left ahead								|
| .00	      			| Go straight or right			 				|
| .00				    | Go straight or left  							|


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


