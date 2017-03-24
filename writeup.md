## Traffic Sign Recognition

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:

* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/explore_1.png "Histogram of Traffic Signs"
[image2]: ./writeup_images/explore_2.png "Example training images for each traffic sign class"
[image3]: ./writeup_images/preprocess.png "Example pre-processed images"
[image4]: ./web-german-traffic-signs/3%20Speed_Limit_60.jpg "Speed limit sign"
[image5]: ./web-german-traffic-signs/9%20No_Passing.jpg "No Passing sign"
[image6]: ./web-german-traffic-signs/11%20Right_Of_Way_Next_Intersection.jpg "Right of way"
[image7]: ./web-german-traffic-signs/18%20General_Caution.jpg "No Passing sign"
[image8]: ./web-german-traffic-signs/25%20Road_work.jpg "Road Work"
[image9]: ./web-german-traffic-signs/28%20Children_Crossing.jpg "Children Crossing"
[image10]: ./writeup_images/web_predictions.png "Predictions on web images"
[image11]: ./training_rate.png "Training rate"


## Rubric Points
| Rubric Point            	| Work Done
|:------|:------------| 
|Submission Files| All required files included in github repo|
|Data Set Summary| Provided brief summary of data set |
|Exploratory Visualization| Histogram of traffic sign class. Sample images for each traffic sign class |
|Preprocessing| Normalized using cv2|
|Model Architecture| LeNet with dropouts |
|Model Training| Done |
|Solution Approach| Done |
|Acquiring New Images| used google image to get traffic signs |
|Performance on New Images| Done |
|Model Certainty, Softmax Probabilities| Done|


## Data Set Summary & Exploration
#### 1. Basic Summary of the data set
Using pandas and numpy, a basic summary of the data set is provided below:
>Exploring Data set:  
>Number of training examples = 34799  
>Number of validation examples = 4410  
>Number of testing examples = 12630  
>Image data shape = (32, 32, 3)  
>Number of Traffic Signs (class) = 43  
>y_train.shape =  (34799,)  

Also summarized the Most and Least common road signs (prefixed with #sign-id):
> Most common road signs:  
> 2 Speed limit (50km/h)  
> 1 Speed limit (30km/h)  
> 13 Yield  
> 12 Priority road  
> 38 Keep right  
> Least common road signs:  
> 0 Speed limit (20km/h)  
> 37 Go straight or left  
> 19 Dangerous curve to the left  
> 32 End of all speed and passing limits  
> 27 Pedestrians  


#### 2. Visualising the data set
- Histogram of the count for each of the traffic sign class
![alt text][image1]
- Displaying 3 images per traffic signs for all traffic signs
![alt text][image2]


## Design and Test the Model

### 1. Pre-processing the image data
Preprocessing is being done in the IPython code cells 4,5 and 6

Preprocessing techinques explored:

- conversion to grayscale 
- minmax scaling to the range [0.1, 0.9]
- minmax scaling to the range [10, 240]
- cv2 normalize the image

Observations:

- Using grayscale and minmax normalization, the training-accuracy was not increasing beyond 0.91
- Using just minmax normalization the accuracy of the model was not increasing beyond 0.92
- When i switched to cv2 nomalize to pull up dark images, the accuracy improved.
- Tried skimage's rescale_intensity as well

**Example Preprocessed images:**

![alt text][image3]


### 2. Data for training, validation and testing
No additional data split was done since the input data was already split into 3 files one each for training, valid (for cross-validation) and testing.

> STANDOUT - TODO  
> augmentation - TODO  
>   rotation  
>   skew  (seeing at an angle)  
>   noise (simulate rain/fog)  
>   degauss (simulate fog)  
>   brightness clipping (simulate high contrast scenario)  


### 3. Model Architecure 
Code block #7 contains the Model Architecture.
The model is the same as LeNet with dropout added to the fully connected layers.

| Layer | Stage | Info |
|:----:|:---|:---------|
|0|Input|32x32x3 RGB image| 
|1|Convolution|wgts 5x5x3x6, stride 1x1, valid padding, outputs 28x28x6|
|1|RELU| |
|1|Max pooling| k 2x2, string 2x2, outputs 14x14x6 |
|2|Convolution|wgts 5x5x6x16, stride 1x1, valid padding, outputs 10x10x16|
|2|RELU| |
|2|Max pooling| k 2x2, string 2x2, outputs 5x5x16|
|2|Flatten| Flatten image. Output 400|
|3|Fully Connected|Input 400, Output 120|
|3|RELU||
|3|Dropout||
|4|Fully Connected|Input 120, Output 84|
|4|RELU||
|4|Dropout||
|5|Fully Connected|Input 84, Output 43 (num traffic sign classes)|


### 4. Model Training
Code blocks 8, 9, 10 and 11 contain the model training and validation code.
For training Adam Optimizer was used with loss operation being a reduced_mean of cross_entropy.
Hyper-parameters:

- Batch Size 128
- EPOCHS 50
- Learning Rate 0.001
- Dropout 0.75

### 5. Approach for finding solution

The LeNet model was used as the starting point for building the solution. With the LeNet model as is, the accuracy peaked at 0.94. Augumented the LeNet model by adding dropout to reduce over-fitting to input data.

Reason for starting with LeNet model is that the traffic signs were mostly mono chromatic and had stick symbols that conveyed meaning (similar to numbers). The finaly accuracy is well below LeNet's accuracy with numbers. The model can be improved.

Currently iam not aware of how to heuristically build an architecture. Will try out once i get the initial solution done. 

The Code for calculating accuracy is in code block 9.

My final model results were:

* validation set accuracy of  0.96
* test set accuracy of 0.94

> TODO  
> explore - different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function
> 

Over multiple iterations these parameters were tuned to understand the behaviour:

- Batch Size: (increasing batch size tended to reduce the accuracy)
- EPOCH: For model with dropout increasing epochs resulted in convergence to larger accuracy
- Training rate: lesser value needed more EPOCHs and hence more time to train on my laptop without a graphics card
- Tried different pre-processing and found that it had a significant impact on accuracy of the training

**Validation accuracy across epochs**  
![alt text][image11]

## Test a Model on New Images

### 1. Web Images

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6]  ![alt text][image7] ![alt text][image8] ![alt text][image9]


### 2. Prediction on web images
The code for making predictions with the new images is located in code block 17, 18. The model was able to guess 5 out of 6 images correctly once i cropped them to include just the image in the bounding box. 

Pediction accuracy: 83.3%

### 3. Sotmax probabilities for web images

![alt text][image10]

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.
