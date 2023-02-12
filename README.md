# MNIST-dataset-classification-using-CUDA
## 1 - Introduction
This project aims to train a simple fully connected neural network with one hidden layer for MNIST dataset classification using CUDA programming in C++. In section 2, information about the dataset will be provided, followed by section 3, where the implemented neural network is explained. In Section 4, the CUDA implementation of the network is discussed, and Section 5 illustrates the results and a comparison to the PyTorch implementation of the same network with the same dataset. Finally, section 6 draws a conclusion.

## 2 - MNIST dataset
MNIST contains 70,000 images of handwritten digits from 0 to 9: 60,000 for training and 10,000 for testing. The images are grayscale, 28x28 pixels. Figure 1 shows some examples of the images with their corresponding labels.

![image](https://user-images.githubusercontent.com/57262710/218329153-1b2a8398-38e3-4226-8403-9e3f315770ca.png)
 
Figure 1: Dataset example
Each pixel has a value between 0 to 255. To speed up the training process, all pixel values were normalized between 0 to 1 in Matlab. The resulting data was saved into separate .txt files as training and test data. In this project, the images are fed to the network as a 1-D array with 28*28 elements, and the output should be the label for this image.



## 3 - Fully connected neural network
A fully connected (FC) neural network consists of a series of fully connected layers that connect every neuron in one layer to every neuron in the next layer. Two important hyperparameters of an FC network are the number of hidden layers and the number of neurons in each layer. The input and output layers’ number of neurons is specified by the problem specification. In this project, an FC network is developed with three layers with 784, 400, and 10 neurons, respectively. The network takes each pixel of an input image and out an array of 10 numbers. ReLU is used as the hidden layer activation function, and softmax is utilized at the final layer to out the probability of the target handwritten number. In other words, the 10 number in the output of the network is the probability of inputting a handwritten picture. For example, if the input image illustrates digit 5, the sixth element of the output array should have the maximum probability, perfectly, it should be around 1, and other elements should be around zero. This classification technique is called one-hot labeling. Clearly, the target data for the supervised learning algorithm is the one-hot labeled 10 digits corresponding to each input image.
For training the model, categorical cross-entropy, which is one of the best cost functions for multiclass classification problems, was chosen, as shown in equation (1) [1]:

	J(W,b)=-1/m ∑_(i=1)^m▒∑_(j=1)^c▒〖y_ji  log⁡〖y ̂_ji 〗 〗	(1)
  
  
