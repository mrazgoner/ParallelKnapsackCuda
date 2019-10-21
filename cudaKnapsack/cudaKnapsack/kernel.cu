#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <fstream>
#include <math.h>
#include <fstream >
#include <stdio.h>
#include <tchar.h>

#pragma once

// Including SDKDDKVer.h defines the highest available Windows platform.

// If you wish to build your application for a previous Windows platform, include WinSDKVer.h and
// set the _WIN32_WINNT macro to the platform you wish to support before including SDKDDKVer.h.

#include <SDKDDKVer.h>

using namespace std;

cudaError_t knapsack_with_cuda(double *_gainMatrix, int *_weight, double *_gain, bool *_isIncluded, int _w, int _n);
__global__ void cudaKnapsack_kernel(double *gainMatrix, int *weight, double *gain, bool *isIncluded, int w, int n);

class ParallelKnapsack
{
public:

	ParallelKnapsack(int* _weight, double* _gain, int _items, int _capacity)
	{
		items = _items;
		weight = _weight;
		gain = _gain;
		capacity = _capacity;

		n = items + 1;
		w = capacity;
		resultVector = new bool[items];

		initMatrix();
	}

	void compute()
	{
		knapsack_with_cuda(gainMatrix, weight, gain, isIncluded, w, n);
		calculateVector();
	}

	//accessors 
	bool* getVector() { return resultVector; }
	double getMaxGain() { return maxGain; }

private:

	// input
	int items;
	int capacity;

	int n, w;

	// input vectors 
	int* weight;
	double* gain;

	// matrix 
	double* gainMatrix;
	bool* isIncluded;

	// results
	double maxGain;
	bool* resultVector;

	// define matrix of reduced problems
	void initMatrix()
	{
		gainMatrix = new double[n*w];
		isIncluded = new bool[n*w];

		for (int i = 0; i < n; i++)
		{
			double _gain = 0;
			for (int j = 0; j < w; j++)
			{
				gainMatrix[i*w + j] = 0;
				isIncluded[i*w + j] = false;

			}// for j
		}// for i 

	}// initMatrix()


	void calculateVector()
	{
		int i = n - 1;
		int j = w - 1;

		maxGain = gainMatrix[i*w + j];

		while (i > 0)
		{
			resultVector[i - 1] = isIncluded[i*w + j];
			if (isIncluded[i*w + j])
			{
				j -= weight[i - 1];
			}

			i--;
		}
	}// calculate vector

};


// cuda kernel for knapsack
__global__ void cudaKnapsack_kernel(double *gainMatrix, int *weight, double *gain, bool *isIncluded, int w, int n)
{

	//for (int row = 1; row < n; row++)

	int row = 1 + threadIdx.x;

	int add_weight = (row == 1 ? 0 : weight[row - 1]);

	for (int col = 1 - row; col < w; col++)
	{
		// threds responsible for lower rows wait for their predcessors to finish
		// the threads work as a pipe-line
		if (col > 0)   // skip if dependance rows not ready
		{

			if (add_weight > col) // new item is too heavy
			{
				// new item is not added - same gain as without it
				gainMatrix[w*row + col] = gainMatrix[w*(row - 1) + col];
				isIncluded[w*row + col] = false;
			}
			else // new item can fit
			{
				double exclude_gain = gainMatrix[w*(row - 1) + col];
				double include_gain = gainMatrix[w*(row - 1) + col - add_weight] + gain[row - 1];

				// select better option
				if (exclude_gain >= include_gain)
				{
					//exclude
					gainMatrix[w*row + col] = exclude_gain;
					isIncluded[w*row + col] = false;
				}
				else
				{
					//include
					gainMatrix[w*row + col] = include_gain;
					isIncluded[w*row + col] = true;
				}
			}// assigning new gain 

		} // endif for skip-check

	}// for

}// knapsack_kernel

int _tmain(int argc, char* argv[])
{

	int* weight = new int[20];
	double* gain = new double[20];

	for (int i = 0; i < 20; i++)
	{
		weight[i] = 2 + i + i % 3 - i % 2 + i % 7;
		gain[i] = 3 + i % 5 + i % 7 - i % 3 + 0.1*(i % 5) + 0.01*(i % 13);
	}

	ParallelKnapsack knap(weight, gain, 20, 45);

	ofstream outputFile;
	outputFile.open("output.txt");

	outputFile << "items:";
	for (int i = 0; i < 20; i++)
		outputFile << '\t' << "#" << i;
	outputFile << endl << "weights:";
	for (int i = 0; i < 20; i++)
		outputFile << '\t' << weight[i];
	outputFile << endl << "value:";
	for (int i = 0; i < 20; i++)
		outputFile << '\t' << gain[i];
	outputFile << endl;

	knap.compute();

	outputFile << "optimal set:" << knap.getMaxGain() << endl;

	outputFile << endl << "item:";
	for (int i = 0; i < 20; i++)
		outputFile << '\t' << knap.getVector()[i] ? "Yes" : "No";
	outputFile << endl;

	system("pause");
	return 0;
}


// knapsack kernel wraper function
cudaError_t knapsack_with_cuda(double *_gainMatrix, int *_weight, double *_gain, bool *_isIncluded, int _w, int _n)
{
	double *gainMatrix = 0;
	int *weight = 0;
	double *gain = 0;
	bool *isIncluded = 0;
	int w = _w;
	int n = _n;

	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system------------------.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	/*---------------- shared memory allocation ---------------------*/

	// Allocate GPU buffers for gainMatrix.
	cudaStatus = cudaMalloc((void**)&gainMatrix, n * w * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Allocate GPU buffers for isIncluded.
	cudaStatus = cudaMalloc((void**)&isIncluded, n * w * sizeof(bool));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Allocate GPU buffers for weight.
	cudaStatus = cudaMalloc((void**)&weight, (n - 1) * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Allocate GPU buffers for gain.
	cudaStatus = cudaMalloc((void**)&gain, (n - 1) * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	/*---------------- shared memory copy ---------------------------*/

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(gainMatrix, _gainMatrix, n * w * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(weight, _weight, (n - 1) * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(gain, _gain, (n - 1) * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}


	/*---------------- Kernel Execution -----------------------------*/

	// Launch a kernel on the GPU with one thread for each element.
	cudaKnapsack_kernel << < 1, n >> >(gainMatrix, weight, gain, isIncluded, w, n);

	/*---------------- Error check and synchronize ------------------*/

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	/*---------------- Copy data back to class ----------------------*/

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(_isIncluded, isIncluded, n * w * sizeof(bool), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(_gainMatrix, gainMatrix, n * w * sizeof(double), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	/*---------------- free cuda memory------------------------------*/

Error:
	cudaFree(gainMatrix);
	cudaFree(weight);
	cudaFree(gain);
	cudaFree(isIncluded);

	return cudaStatus;
}

