#pragma once

#include "nn_utils/matrix.hh"
#include <vector>

class CoordinatesDatasetMNIST {
private:
	size_t batch_size;
	size_t number_of_batches;

	std::vector<Matrix> batches;
	std::vector<Matrix> targets;
	std::vector<std::vector<float>> data;
public:

	CoordinatesDatasetMNIST(size_t batch_size, size_t number_of_batches, std::vector<std::vector<float>> data);

	int getNumOfBatches();
	std::vector<Matrix>& getBatches();
	std::vector<Matrix>& getTargets();
};
