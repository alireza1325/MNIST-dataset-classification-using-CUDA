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

![image](https://user-images.githubusercontent.com/57262710/218329222-f40feb6b-7bbf-4a57-89f9-273ac1cbb664.png)

where W and b are the network’s weights and biases, m is the total number of training samples, c is the number of classes (for the MNIST dataset is 10), y_ji is the ground truth label and y ̂_ji is the predicted label. The learning rate was also selected 0.001, and batch size was set to 100 to the speedup training process. The softmax activation at the output of the network will turn the output into probabilities. Assume a_i is the output of model for ith input image. The predicted probability of this input can be calculated as shown in equation (2) [1].

![image](https://user-images.githubusercontent.com/57262710/218329240-28b410d0-7a97-4fbd-a333-7d76a2abfe6d.png)


### 3.1 - Forward path
In the forward path, we just have some matrix-vector multiplication. The whole network in the forward path can be summarized in equations 3 to 6 [2].

![image](https://user-images.githubusercontent.com/57262710/218329269-4f2457cf-cc5e-43f7-82b4-d2b84e5be771.png)

W_1 and b_1 are weight matrix and bias vector of the first layer and W_2 and b_2  are weights and biases of the second layer.

### 3.2	- Backward pass
For implementing the backward pass, we need to calculate the the derivative of cost with respect to each element of weights and biases. This derivation somehow estimates the sensitivity of the cost with respect to that weight. By having all derivation, we can update weights and biases using the gradient decent algorithm, as depicted in equations (7) and (8).

![image](https://user-images.githubusercontent.com/57262710/218329294-bd19b17b-0b28-4c03-90b4-3b20efc0b878.png)

Clearly, the derivative of cost with respect to first layer weight and biases can be calculated through the chain rule and multiplying partial derivatives of subsequent layers. It has worth mentioning that the derivation of categorical cross entropy cost with respect to the output of the network can be computed using equation (9).

![image](https://user-images.githubusercontent.com/57262710/218329317-e2d5a405-2a84-4e2e-957e-0dc978abb927.png)

## 4 - CUDA implementation of the network
In this section, the forward, backward, training, and testing of the network are explained. The code is written in a way that the network can be expanded to any number of layers with any activation functions or cost functions. Detailed information regarding the written code will be provided in the following subsections.

### 4.1	- Matrix class
As discussed in the previous section, the FC network contains lots of matrix multiplication. To better manipulate matrixe, a matrix class is implemented. Figure 2 shows the Matrix.hh file where private and public variables and functions are defined. As can be seen, inside this class, some useful functions for easier handling of the Matrix have been implemented. For example, two functions for transferring data between host and devices are written to make CUDA programming easier, as illustrated in figure 3. Detailed information about the functions of this class can be found in Matrix.cu file. It should be noted that the NNException is another class for handling CUDA error messages.

![image](https://user-images.githubusercontent.com/57262710/218329341-abf74869-fece-4fce-8f70-d4410ed37087.png)

Figure 2: Matrix class 

![image](https://user-images.githubusercontent.com/57262710/218329376-d7ad41ec-1a49-4071-8fb3-f53c86de66f9.png)

Figure 3: Transferring Matrix functions between GPU and CPU
### 4.2	- Dataset class 

In order to make batch processing easier, a dataset coordination class has been written. It converts the raw .txt train and test data into manageable format (Matrix class discussed in the previous section) and also does the one-hot labeling. This class and its functions are implemented in coordinates_dataset_mnist.cu and coordinates_dataset_mnist.hh files.

### 4.3	- Layers classes
Every layer in an FC neural network should have a name and two functions: forward and backward. It is realized in nn_layer.hh file, as shown in figure 4.

![image](https://user-images.githubusercontent.com/57262710/218329388-01b15cf9-d466-4657-bcef-b66b6a6d4c35.png)

Figure 4: layer class

In forward propagation, we first need to initialize the weight and biases of the layer with random numbers and zero arrays, respectively and multiply the weight with input to that layer, and sum the result with a bias vector, as previously demonstrated in equations (3) and (5). In the backward pass (or, simply speaking backprob) finally, we need to update the weight and biases. These are implemented in linear_layer.hh and linear_layer.cu files. Figure 5 shows the linear layer class (linear_layer.hh file).

![image](https://user-images.githubusercontent.com/57262710/218329404-be717bcb-5129-4873-a3bf-8f28d0dab7e1.png)

Figure 5: Linear layer class

Just for simplicity, some functions are implemented in the CPU because they are running just once when the network is created, for example, weights initialization. All biases were initiated with zero numbers, and all weights were initiated using the HE initialization method and random numbers, as depicted in figure 6. It has been experimentally demonstrated that this way of weight initialization will help networks with ReLU activation functions to converge [3].

![image](https://user-images.githubusercontent.com/57262710/218329417-d3dc953a-f2d5-489b-b2ce-30fdedf83a0b.png)

Figure 6: Weight and bias initialization

In forward propagation, equation (5) has been implemented, as illustrated in figures 7 and 8. In Lab 5, I noticed that a block size of 8×8 is somehow an optimized choice for GPU computation for 2-D thread configuration. (It makes sense because threads are dispatched in wrap in CUDA and every wrap contains 32 threads, and 64 threads will make two wraps). So, I have used a block size of 8×8, and grid size is computed based on matrix dimensions and block size in order to make sure threads will cover all required matrix indexes during matrix-vector multiplication.

![image](https://user-images.githubusercontent.com/57262710/218329432-7557ac2b-2fbc-43e5-ad61-a893f84f4b42.png)
 
Figure 7: Layer forward function

The kernel for linear layer forward pass calculation is shown in figure 8. 

![image](https://user-images.githubusercontent.com/57262710/218329440-06dc3c9f-15fd-441d-973f-2b0537c47ab2.png)
 
Figure 8: linear layer forward pass calculation

Backprob includes three steps: calculate partial derivation, update weights, and update biases, as shown in figure 9. Similar to forward pass, these three functions have their own definitions and kernels that can be found in the liniear_layer.cu.

![image](https://user-images.githubusercontent.com/57262710/218329447-fc205989-3570-4771-ad90-fd112c88b7bf.png)

Figure 9: Three steps of linear layer backprob.

### 4.4	- ReLU activation function
ReLU activation, like the linear layer, is inherent in NNLayer class that was explained previously (see figure 4). It just passes all Z vector elements from the ReLU function, as previously shown in equation (4). The ReLU function returns the maximum of the input value and zero. In other words, it passes all positive values as it is and returns zero for all negative values. Figure 10 indicates the kernel function for forward and backward ReLU activation. Further details about this activation function can be found in relu_activation.hh and relu_activation.cc. As can be seen, each CUDA thread is responsible for calculating a single entry of vectors A or dZ.

![image](https://user-images.githubusercontent.com/57262710/218329456-0f9431d4-cd95-4429-91b5-b0d03c7ff3c2.png)

Figure 10: ReLU activation function forward and backward CUDA kernels.
### 4.5 - Cost function 
As explained in previous sections, the categorical cross entropy (CCE) loss is used after the softmax layer for error computation for this project. Similar to known AI libraries (i.e., Pytorch), the softmax activation is implemented inside the CCE cost, as depicted in figure 11. Figure 11 shows the kernel of CCE cost in the forward pass where each thread firstly computes the sum of the exponential of one output of the network for calculating the softmax layer (see equation (2)), and subsequently computes the cost for this output. In line 25, the cost calculated for this output is added to the total cost for this batch of data using the atomicAdd function, which is implemented by NVIDIA, since it has a bit better performance rather than the simple addition.

![image](https://user-images.githubusercontent.com/57262710/218329474-9542b51a-754b-4781-b8db-f44cf5249012.png)

Figure 11: CCE cost forward

The backward pass of CCE cost, shown in figure 12, actually implements equation (9) with softmax calculation. It should be noted that we do not need synchronization in these two kernels as each thread just computes the cost and derivative of cost for one single output of the network.

![image](https://user-images.githubusercontent.com/57262710/218329534-0593dafe-27fe-4731-a306-a2a9fe6bbf0a.png)

Figure 12: CCE cost backprob

### 4.6	- Building the network and main function
In the main.cu file, first necessary libraries are included. Then the dataset is read from two separate .txt file followed by creating the train and test dataset from these raw data, as illustrated in figure 13. In line 68, the cost object is created, and lines 70 to 73 are devoted to creating the network. As can be seen, any number of layers or any kind of activation functions can be added to the network, just as shown in figure 13. The main limitation of this implementation is that we can only create sequential layers, and it is not possible to use recent deep learning techniques like skip connections.

![image](https://user-images.githubusercontent.com/57262710/218329550-9d149d88-88a8-4af2-9055-3118398b7e44.png)

Figure 13: main.cu: dataset, cost, and model development

![image](https://user-images.githubusercontent.com/57262710/218329565-4572a5de-a771-49d4-9016-b40a75e4631a.png)

Figure 14 shows the training section of main.cu file where the training time has been calculated just like taught in the class [4]. 
 
Figure 14: main.cu: training and timing
After training the accuracy of the model has been estimated by calculating the total number of correct predictions using the cumputeAccuracy function, as depicted in figure 15. 
 
Figure 15: main.cu: computing the accuracy of the model
Figure 16 illustrates the computeAccuracy function. This function receives the target and predicted labels into batches and sees if the maximum number in the output array of the network matches with the index of the target array, which is 1. If this condition is true, it adds one number to corrected predictions, and when it is done with this batch calculations, it returns the number of corrected predictions.
 
Figure 16: main.cu: computeAccuracy function
5	Results
In previous sections, we have seen detailed information about the network and its implementation using CUDA programming. This section provides the results of the code and compares them to the results of the same implementation of the network in Pytorch. You can find the python code of the PyTorch model in the mnist.py file.
The networks are trained using 5 epochs. Figure 17 shows the output of the written code in the command window. As can be seen, the training cost continuously decreased during the training processes, and finally, 88.67 % of accuracy was achieved in this run, which took around 209 s. Furthermore, 10 outputs of the network with corresponding labels have been printed in the command window (each column shows one output and one target). Interestingly, the model could successfully predict the correct labels in these ten samples as the output has the highest value (highest probability if we pass the output from the softmax function) in the index at which the target is one.

 
 
Figure 17: Training and test results of the network in c++
The same network was trained using PyTorch with the same data, cost function, learning rate, and batch size, and figure 18 shows the result. Compared to CUDA implementation in c++, the PyTorch model took around one-fourth time to train, but its accuracy is lesser by around 12% in the same number of epochs.

 
 
Figure 18: Training and test results of the pytorch implementation of the network in python
Finally, both networks in c++ and python were trained using 30 epochs (you can find the loss of each epoch in the two attached .txt files), and the results are tabulated in table 1. Regarding table 1, the accuracy of the CUDA code is more than the PyTorch, but the training time is roughly four times more than the PyTorch model in python. 

  
  
