#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <time.h>

#include "neural_network.hh"
#include "layers/linear_layer.hh"
#include "layers/relu_activation.hh"
#include "layers/sigmoid_activation.hh"
#include "nn_utils/nn_exception.hh"
#include "nn_utils/bce_cost.hh"
#include "nn_utils/cce_cost.hh"

#include "coordinates_dataset_mnist.hh"

#include <fstream>
#include <string>
#include <vector>
#include <sstream>

float computeAccuracy(const Matrix& predictions, const Matrix& targets);


int main() {

	int N_train = 60000;
	int N_test = 10000;
	int input_size = 28 * 28 +1;
	// read dataset
	std::cout << "Loading the dataset" << std::endl;

	std::ifstream file("mnist_train.txt");
	std::vector<std::vector<float>> train_data(N_train, std::vector<float>(input_size, 0));
	for (int i = 0; i < N_train; i++) {
		for (int j = 0; j < input_size; j++) {
			file >> train_data[i][j];
		}
		//std::cout<< data[i][0]<<std::endl;
	}
	std::cout << "Train data is loaded" << std::endl;

	std::ifstream file_test("mnist_test.txt");
	std::vector<std::vector<float>> test_data(N_test, std::vector<float>(input_size, 0));
	for (int i = 0; i < N_test; i++) {
		for (int j = 0; j < input_size; j++) {
			file_test >> test_data[i][j];
		}
		//std::cout<< data[i][0]<<std::endl;
	}
	std::cout << "Test data is loaded" << std::endl;
	
	int batch_size = 100;
	int total_betches = 600;
	int total_betches_test = 100;
	int n_epochs = 5;

	srand(time(NULL));

	CoordinatesDatasetMNIST dataset(batch_size, total_betches, train_data);           //train dataset
	CoordinatesDatasetMNIST test_dataset(batch_size, total_betches_test, test_data);  //test dataset
	std::cout << "Datasets created" << std::endl;
	//std::cout << "Train data total number of batches = " << dataset.getNumOfBatches() << std::endl;
	//std::cout << "Test data total number of batches = " << test_dataset.getNumOfBatches() << std::endl;

	CCECost cce_cost;

	NeuralNetwork nn;
	nn.addLayer(new LinearLayer("linear_1", Shape(784, 400)));
	nn.addLayer(new ReLUActivation("relu_1"));	
	nn.addLayer(new LinearLayer("linear_2", Shape(400, 10)));
	std::cout << "Model created" << std::endl;


	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	//// Network training
	std::cout << "Start training" << std::endl;
	Matrix Y;
	Matrix tar;
	for (int epoch = 0; epoch < n_epochs; epoch++) {
		float cost = 0.0;

		for (int batch = 0; batch < dataset.getNumOfBatches(); batch++) {
			Y = nn.forward(dataset.getBatches().at(batch));
			nn.backprop(Y, dataset.getTargets().at(batch));
			cost += cce_cost.cost(Y, dataset.getTargets().at(batch));

			if ((batch+1) % 100 ==0) {
				std::cout << "Epoch: " << epoch+1<< "/"<< n_epochs << ", step: " << batch+1 << "/" << dataset.getNumOfBatches()
					<< ", Cost: " << cost / (batch+1) << std::endl;
			}
		}
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float gpu_time;
	cudaEventElapsedTime(&gpu_time, start, stop);   //time in milliseconds
	gpu_time /= 1000.0; // time in seconds


	std::cout << std::endl;
	std::cout << "Training finished " << std::endl;
	std::cout << "Training time = " << gpu_time << " s" << std::endl;

	// Destroy time objects
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	// compute accuracy

	// Printing ten output of the network
	Matrix pred;
	Matrix target;
	pred = nn.forward(test_dataset.getBatches().at(10));
	target = test_dataset.getTargets().at(10);

	pred.copyDeviceToHost();
	target.copyDeviceToHost();
	std::cout << std::endl;
	std::cout << "Start testing" << std::endl;
	std::cout << "Predictions" << std::endl;
	for (int i = 0; i < 10;i++) // loop over classes
	{
		for (int j = 0; j < 10; j++) // loop over samples
		{
			std::cout << pred[i*batch_size+j] << ", ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
	std::cout << "Targets" << std::endl;

	for (int i = 0; i < 10; i++) // loop over classes
	{
		for (int j = 0; j < 10; j++) // loop over samples
		{
			std::cout << target[i * batch_size + j] << ", ";
		}
		std::cout << std::endl;
	}

	int correct_predictions = 0;
	for (int batch = 0; batch < test_dataset.getNumOfBatches(); batch++) {
		pred = nn.forward(test_dataset.getBatches().at(batch));
		target = test_dataset.getTargets().at(batch);
		pred.copyDeviceToHost();
		target.copyDeviceToHost();
		correct_predictions += computeAccuracy(pred, target);
	}
	
	std::cout << std::endl;
	std::cout <<"Number of corrected predictions = " << correct_predictions << std::endl;
	std::cout <<"Accuracy = " << (float)((float)correct_predictions/ N_test)*100 << " %" << std::endl;

	return 0;
}


float computeAccuracy(const Matrix& predictions, const Matrix& targets) {
	int batch_size = predictions.shape.x;
	int correct_predictions = 0;
	float max;
	int index;
	for (int i = 0; i < batch_size; i++) {
		max = 0;
		for (int j=0; j < 10; j++)
		{
			if (predictions[j * batch_size + i]>max) {
				max = predictions[j * batch_size + i];
				index = j;
			}
		}
		if (targets[index * batch_size + i] == 1)
		{
			correct_predictions++;
		}
	}
	return static_cast<float>(correct_predictions) ;
}
