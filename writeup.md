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

[explore1]: ./writeup_images/explore1.png "Histogram of Traffic Signs"
[explore2]: ./writeup_images/explore2.png "Example training images for each traffic sign class"

[pp1]: ./writeup_images/preprocess1.png "Example pre-processed images"
[pp2]: ./writeup_images/preprocess2.png "Example pre-processed images"
[pp3]: ./writeup_images/preprocess3.png "Example pre-processed images"

[valid_epoch]: ./writeup_images/validation_epochs.png "Training rate"
[tensor_accuracy]: ./writeup_images/tensorboard_accuracy.png  "Tensorboard Accuracy graph"
[tensor_loss]: ./writeup_images/tensorboard_loss.png  ""Tensorboard Loss graph"


[lenet_graph]: ./writeup_images/lenet_graph.png "tfboard view of LeNet graph"

[web_softmax]: ./writeup_images/web_predictions.png "Predictions on web images"

[image4]: ./web-german-traffic-signs/3%20Speed_Limit_60.jpg "Speed limit sign"
[image5]: ./web-german-traffic-signs/9%20No_Passing.jpg "No Passing sign"
[image6]: ./web-german-traffic-signs/11%20Right_Of_Way_Next_Intersection.jpg "Right of way"
[image7]: ./web-german-traffic-signs/18%20General_Caution.jpg "No Passing sign"
[image8]: ./web-german-traffic-signs/25%20Road_work.jpg "Road Work"
[image9]: ./web-german-traffic-signs/28%20Children_Crossing.jpg "Children Crossing"



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
- Histogram of the count for each of the traffic sign class for train,valid and test data-sets. All the data-sets are proportional. 
![alt text][explore1]
- Displaying 3 images per traffic signs for all traffic signs
![alt text][explore2]


## Design and Test the Model

### 1. Pre-processing the image data
Preprocessing is being done in the IPython code cells 4 and 5

**Preprocessing techinques explored**

- conversion to grayscale 
- minmax scaling to the range [0.1, 0.9]
- minmax scaling to the range [10, 240]
- cv2 normalize the image
- Histogram Equialization (CLAHE) 

**Observations**

1. Using grayscale and minmax normalization, the validation accuracy was not increasing beyond 0.91
2. Using just minmax normalization the validation accuracy of the model was not increasing beyond 0.92
3. When i switched to cv2 nomalize to pull up dark images, the validation accuracy improved to 0.95
4. Tried skimage's rescale_intensity as well
5. Tried Histogram Equalization on grayscale image
6. Then tried CLAHE on L-Channel of LAB Colorspace image before converting back to BGR colorspace. With this the validation accuracy improved to 0.96

**Example Preprocessed images:**

![alt text][pp1]
![alt text][pp2]

### 2. Data for training, validation and testing
No additional data split was done since the input data was already split into 3 files one each for training, valid (for cross-validation) and testing.

**Augmenting Image Data**  
Image data was augmented by transforming a subset(50%) of the training images and adding it to the original training set.   
This is being done in code block 6. The transforms applied were a combination of the following:

- rotate: randomly between -5 to 5 degrees
- scale: randomly between 0.9 to 1.3
- translate (shift): randomly between -2 and 2 independently over both axes
- shear: randomly between -5 to 15 degrees

With augmentation the validation accuracy improved.

- When augmenting 20% of the training images, the validation accuracy climbed to 0.97
- When augmenting 50% of the training images, the validation accuracy climbed further to 0.985


![alt text][pp3]

> augmentation - TODO  
>   noise (simulate rain/fog)  
>   degauss (simulate fog)  
>   brightness clipping (simulate high contrast scenario)  

*ref:  
http://florianmuellerklein.github.io/cnn_streetview/  
http://benanne.github.io/2014/04/05/galaxy-zoo.html*


#### 2.1 Impact of Data pre-processing and augmentation


| Pre Processing | Validation Accuracy |
|:----|:---|
|grayscale + minmax normalization|0.91|
|minmax normalization|0.92|
|cv2 normalization|0.93|
|CLAHE on L-Channel|0.96|

| Data Augmentation (with CLAHE) | Validation Accuracy |
|:----|:---|
|No Augmentation|0.96|
|Image augmentation 20%|0.97|
|Image augmentation 50%|0.985|
 


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

**Visualizing the Model using tensorboard**
![alt text][lenet_graph]

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

* validation set accuracy of  0.985
* test set accuracy of 0.959

> TODO  
> explore - different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function
> 

Over multiple iterations these parameters were tuned to understand the behaviour:

- Batch Size: (increasing batch size tended to reduce the accuracy)
- EPOCH: For model with dropout increasing epochs resulted in convergence to larger accuracy
- Training rate: lesser value needed more EPOCHs and hence more time to train on my laptop without a graphics card
- Tried different pre-processing and found that it had a significant impact on accuracy of the training

**Validation accuracy across epochs**  
![alt text][valid_epoch]

**tensorboard graphs for accuracy and loss for 2 training runs (20% augmentation, 50% augmentation)**

![alt text][tensor_accuracy]

![alt text][tensor_loss]

## Test a Model on New Images

### 1. Web Images

Here are five German traffic signs that I found on the web. The images were curated to get the best prediction results from the model. All the images are well illuminated with a central histogram. Some of the images have watermarks on the corner which could throw off the prediction. In once of the test images it was consistently misclassified and i removed it from the test set.

![alt text][image4] ![alt text][image5] ![alt text][image6]  ![alt text][image7] ![alt text][image8] ![alt text][image9]


### 2. Prediction on web images
The code for making predictions with the new images is located in code block 17, 18. 

Web Image prediction accuracy:  100 %
Test Image prediction accuracy: 95.9 %

The model seems general enough to work with scaled down images taken with regular camera. The model was able to guess all of the images correctly once i cropped them to include just the image in the bounding box. 
I had also noticed that presence of watermark on an image (removed from set) resulted in mis-classification indicating over-fitting.

### 3. Sotmax probabilities for web images

- The model has a very high confidence in prediction in 3 out of the 6 images
- For the rest 3 signs, the confidence of classification was 80% which is a decently high.


![alt text][web_softmax]

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.
