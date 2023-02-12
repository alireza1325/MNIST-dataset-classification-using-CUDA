#include "cce_cost.hh"
#include "nn_exception.hh"

#include <math.h>
#include <iostream>
#include <assert.h>


__global__ void CategoricalCrossEntropyCost(float* predictions, float* target,int batchsize,int c, float* cost) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	float sum_softmax = 0;

	float partial_cost = 0;
	if (index < batchsize) {
		// do softmax
		for (int i = 0; i < c; i++)
		{
			sum_softmax += expf(predictions[i * batchsize + index]);
		}
		for (int i = 0; i < c; i++)
		{
			partial_cost += target[i * batchsize + index] * logf(expf(predictions[i * batchsize + index]) / sum_softmax);
		}
		atomicAdd(cost, -partial_cost / batchsize);
	}
}

__global__ void dCategoricalCrossEntropyCost(float* predictions, float* target, float* dY, int batchsize,int c) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	float sum_softmax = 0;

	if (index < batchsize) {
		for (int i = 0; i < c; i++)
		{
			sum_softmax += expf(predictions[i * batchsize + index]);
		}
		for (int i = 0; i < c; i++)
		{
			dY[i * batchsize + index] = expf(predictions[i * batchsize + index])/sum_softmax - target[i * batchsize + index];
		}
	}
}

float CCECost::cost(Matrix predictions, Matrix target) {
	assert(predictions.shape.x == target.shape.x);

	float* cost;
	cudaMallocManaged(&cost, sizeof(float));
	*cost = 0.0f;

	
	dim3 block_size(256);
	dim3 num_of_blocks((predictions.shape.x + block_size.x - 1) / block_size.x);
	CategoricalCrossEntropyCost <<<num_of_blocks, block_size>>>(predictions.data_device.get(),
														  target.data_device.get(),
														  predictions.shape.x, predictions.shape.y, cost);
	cudaDeviceSynchronize();
	NNException::throwIfDeviceErrorsOccurred("Cannot compute binary cross entropy cost.");

	float cost_value = *cost;
	cudaFree(cost);

	return cost_value;
}

Matrix CCECost::dCost(Matrix predictions, Matrix target, Matrix dY) {
	assert(predictions.shape.x == target.shape.x);

	dim3 block_size(256);
	dim3 num_of_blocks((predictions.shape.x + block_size.x - 1) / block_size.x);
	dCategoricalCrossEntropyCost <<<num_of_blocks, block_size>>>(predictions.data_device.get(),
														   target.data_device.get(),
														   dY.data_device.get(),
														   predictions.shape.x, predictions.shape.y);
	NNException::throwIfDeviceErrorsOccurred("Cannot compute derivative for binary cross entropy.");
	
	return dY;
}
