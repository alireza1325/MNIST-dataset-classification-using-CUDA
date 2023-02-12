#include "coordinates_dataset_mnist.hh"


std::vector<float> one_hot_encodding(float label, int n)
{
	std::vector<float> one_hot_label(n);
	for (int i = 0; i < n; i++)
	{
		one_hot_label[i] = 0;
	}
	one_hot_label[label] = 1;
	return one_hot_label;
}


CoordinatesDatasetMNIST::CoordinatesDatasetMNIST(size_t batch_size, size_t number_of_batches, std::vector<std::vector<float>> data) :
	batch_size(batch_size), number_of_batches(number_of_batches)
{
	std::vector<float> one_hot_label(10);
	
	for (int i = 0; i < number_of_batches; i++) {
		batches.push_back(Matrix(Shape(batch_size, 784)));
		targets.push_back(Matrix(Shape(batch_size, 10)));

		batches[i].allocateMemory();
		targets[i].allocateMemory();

		for (int k = 0; k < batch_size; k++)
		{
			for (int row = 0; row < 784; row++)
			{
				batches[i][row * batch_size + k] = data[i* batch_size+k][row+1];
			}
			
			one_hot_label = one_hot_encodding(data[i * batch_size+k][0], 10);
			for (int row = 0; row < 10; row++)
			{
				targets[i][row * batch_size + k] = one_hot_label[row];
			}
			
		}
		batches[i].copyHostToDevice();
		targets[i].copyHostToDevice();
	}
}

int CoordinatesDatasetMNIST::getNumOfBatches() {
	return number_of_batches;
}

std::vector<Matrix>& CoordinatesDatasetMNIST::getBatches() {
	return batches;
}

std::vector<Matrix>& CoordinatesDatasetMNIST::getTargets() {
	return targets;
}
