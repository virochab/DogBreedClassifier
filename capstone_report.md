Machine Learning Engineer Nanodegree

Capstone Project

Virochan Badiger

April 09, 2021

**
Metrics**

**I. Definition**
Capstone project for Udacity Machine Learning Engineer nanodegree. 
**II. Analysis**
**Project Overview**
This project is on the Image classification using CNN. If the image of a dog is provided it should identify the dog breed. If image of a human is provided it 
should provide the most resembling canine breed. This is a multi-class classification problem using 
supervised machine learning under deep learning CNN.

**Metrics**
Accuracy will be the main metric used to measure this model. 

**II. Analysis**
To detect human faces, we use Open CV’s Haar feature based cascade classifiers model. 
OpenCV already contains many pre-trained classifiers for face, eyes, smile etc. Those XML files are 
stored in opencv/data/haarcascades/ folder. Using this facedetector we can detect human faces. 
To detect dogs, we use a pretrained VGG-16 model. VGG16 is a 16 layers deep convolutional neural 
network model proposed by K. Simonyan and A. Zisserman from the University of Oxford in the paper 
“Very Deep Convolutional Networks for Large-Scale Image Recognition”. The model achieves 92.7% top 5 test accuracy in ImageNet. 
We then create a CNN from scratch to classify Dog breeds. We should get an accuracy of atleast 10%. 
Now create a CNN using transfer learning using resnet18. We should get an accuracy of atleast 60% with 
this approach. 

For transfer learning we use ResNet-18. ResNet-18 is a convolutional neural network that is 18 layers 
deep. You can load a pretrained version of the network trained on more than a million images from the 
ImageNet database. The pretrained network can classify images into 1000 object categories, such as keyboard, mouse, pencil, and many animals. As a result, the network has learned rich feature 
representations for a wide range of images. The network has an image input size of 224-by-224 
We use this model for transfer learning and adapt it to our requirements. 

Now we create an algorithm to detect if image is of a dog then return the predicted breed. 
If image is of a human then return the resembling dog breed. If image is of neither dog nor human then 
provide output that indicates an error.

**Data Exploration**
Datasets and input are provided by the Udacity from links 
Dog dataset: https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip
Human dataset: https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip
There are total 13233 human images and 8351 dog images. Images are of size 250*250. 

**Benchmark**
The CNN model created from scratch should have atleast 10% accuracy. 
The CNN model created using transfer learning should have accuracy of 60%. 

**III. Methodology**
First: Detect human faces using Open CV’s Haar feature based cascade classifiers model. 

Second: Detect dogs using a pretrained VGG-16 model 

Third: Create a CNN to classify Dog breeds from scratch. We should get an accuracy of atleast 10%. 
I will use convolution layers, each followed by maxpooling layer. Maxpool layers helps to reduce the 
dimensions of the feature maps. 
The output is then passed to linear layers with dropout layers in between them. Drop out layers help in 
avoiding overfitting by zeroing few nodes in network. 

Fourth: Create a CNN using transfer learning using resnet18. We should get an accuracy of atleast 60%. 
The last layer is modified and trained to output only 133 feature outputs. To classify the dog breeds. 

Fifth: - Algorithm to detect if image is a dog then return the predicted breed. 
- if image is of a human then return the resembling dog breed. 
- if image is of neither dog nor human then provide output that indicates an error.

**Data Preprocessing**
Transformations are applied on train dataset images to centre crop of size 224*224. 
Also data augmentation is done using random horizontal flip of train dataset images. 

**Implementation**

For human face detection:
OpenCV's implementation of Haar feature-based cascade classifiers is used to detect human faces in images.

For the model from scratch:
The input dimensions of the images are (224,224,3). 3 is the number of channels.

model print out for reference: (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)) (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)) (conv3): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False) (fc1): Linear(in_features=6272, out_features=500, bias=True) (fc2): Linear(in_features=500, out_features=133, bias=True) (dropout): Dropout(p=0.3)

First Conv layer has 3 input channels and 32 output features maps. kernel size is 3 and stride is 2, padding is 1. After the conv2d layer there is a maxpool layer of size (2,2). This is used to reduce the dimensions of the image. Dimension output is reduced to 56*56

Second Conv layer has 32 input channels and 64 output features maps. kernel size is 3 and stride is 2, padding is 1. After the conv2d layer there is a maxpool layer of size (2,2). This is used to reduce the dimensions of the image. Dimension output is reduced to 14*14

Third Conv layer has 64 input channels and 128 output features maps. kernel size is 3 and stride is 1, padding is 1. After the conv2d layer there is a maxpool layer of size (2,2). This is used to reduce the dimensions of the image. Dimension output is reduced to 7*7

This output is flattened and passed through dropout layer which is used to reduce the overfitting by zeroing some proportion of neurons in the network Then it is passed to linear layer. The input size of the layer is 12877. Output feature size is 500

The output is again passed through a dropout layer which reduces overfitting. Second linear layer input features are 500 and output is 133.
Relu activation function is used for all layers.

For transfer learning model:
For transfer learning I have used ResNet-18. ResNet-18 is a convolutional neural network that is 18 layers deep. You can load a pretrained version of the network trained on more than a million images from the ImageNet database. Resnet-18 is used for the transfer learning task here. We use the pretrained model and choose the last layer alone for training. It is modified to output 133 outputs and trained again on our dataset.
Because of the depth of the network present in this model and also the vast image dataset it is pretrained on. This is suitable for out task.


**IV. Results**

**Model Evaluation and Validation**
The model from scratch reported a accuracy of 17% compared to the benchmark minimum expectation of 10%
The Transfer learning model reported an accuracy of 82% compared to the benchmark minimum expectation of 60%


**Justification**
The model from scratch is not deep enough to get higher accuracy on a dataset of so vast. Hence a basic from scratch model giving 17% accuracy is enough for the project purpose.
The transfer learning model uses resnet 18 and trains last layer alone to get 82% accuracy. If more time is provided to train on the accuracy would increase. 


**V. Conclusion**
Pytorch CNN model architecture used for the project are sufficient for the required benchmarks. If need higher accuracy we can explore more depth layers and higher training time. 


**Reflection**
I had to learn CNN Pytorch for this project from scratch. Having no experience in CNN models. This project was challenging and big learning experience for me. 
The CNN models require understanding of CNN layers covering convolution, maxpooling, dropout layers, activation functions, dimension reduction of images, transformation of the images as part of preprocessing. All this is covered in this project. 

**Improvement**
The CNN model from scratch needs to be more deep and and trained for higher epochs to get more accuracy on image set of vast dog breeds of 133. 
Since working with the restrictions of GPUs and time for project it is challenging to cover in this project. 
The transfer learning could be improved with more exploration of different pretrained models and optimizer selections. Due to time constraint on the use of GPUs on VM and overall project deadline time, a more detailed exploration of different pretrained models was not possible. 




