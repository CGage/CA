
#include "cuda_runtime.h"
#include "cuda.h"
#include "device_launch_parameters.h"
#include "iostream"
#include "stdlib.h"
#include <thread>         // std::this_thread::sleep_for
#include <chrono>         // std::chrono::seconds
#include "time.h"
#include <ctime>
#include "fstream"



using namespace std;

int getPos(int m, int n, const int width) {
	return m* width + n;
}

void printCells(int* cells, int const height, int const width) {
	for (int i = 0; i < height + 2; i++) {
		for (int j = 0; j < width + 2; j++) { 
			if (cells[getPos(i, j, width)] == 1) {
				cout << "O" << " ";
			}
			else {
				cout << "-" << " ";
			}
		}
		cout << endl;
	}
	cout << endl;
	std::this_thread::sleep_for(std::chrono::milliseconds(100));
	system("cls");
}

void populateArray(int* cellArray, int arraySize) {
	for (int i = 0; i < arraySize; i++) { 
		cellArray[i] = rand() % 2;
	}
}

__device__ int getX(int i, int width) {
	return i % width;
}

__device__ int getY(int i, int width) {
	return i / width;
}

__device__ int getI(int m, int n, int width) {
	return m * width + n;
}


//Gets the neigbour cells via von Neuman Neigbourhood 
__device__ int getNeigbours(int m, int n, int* cells, int width, int height) {
	int neigbours = 0;
	for (int i = m - 1; i <= m + 1; i++) {
		for (int j = n - 1; j <= n + 1; j++) {
			if (i >= 0 && i < height && j >= 0 && j < width) {
				neigbours += cells[getI(i, j, width)];
			}
			else {
				neigbours += cells[getI((i + height) % height, (j + width) % width, width)];
			}
		}
	}
	return neigbours;
}

// rules that determines the state of the cell
__device__ int rules(int neigbours, int state) {
	int n = neigbours - state;
	if (state == 1) {
		if (n > 1 && n < 4) {
			return 1;
		}
		else {
			return 0;
		}
	}
	else {
		if (n == 3){
			return 1;
		}
		return 0;
	}

}

// creates the new state of the world
__global__ void evolve(int* cells, const int height, const int width, const int arraySize, const int cellsPerThread) {
	extern __shared__ int sharedCells[];
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	for (int k = i * cellsPerThread; k < ((i + 1) * cellsPerThread); k++) {		
		sharedCells[k] = cells[k];
		int x, y, neigbours;
		x = getX(k, width);
		y = getY(k, width);
		neigbours = getNeigbours(y, x, sharedCells, width, height);
		cells[k] = rules(neigbours, sharedCells[getI(y, x, width)]);
		__syncthreads();
	}

	
}

// Runs the simulation
int main() {
	srand(1);
	const int height = 100, width = 100, arraySize = 10000, timeSteps = 10000, cellsPerThread = 10, gridSize = 10;
	char b;
	int* cells; // CPU
	int* cellsDev; // GPU

	cells = (int*)malloc(sizeof(int)*arraySize); // creating arrays
	populateArray(cells, arraySize);

	cudaMalloc((void**)&cellsDev, sizeof(float)*arraySize); // creating space on gpu

	cudaMemcpy(cellsDev, cells, sizeof(int)*arraySize, cudaMemcpyHostToDevice); // copying arrays to gpu
	clock_t begin = clock();
	for (int i = 1; i < timeSteps; i++) {
		evolve <<<gridSize, arraySize / cellsPerThread / gridSize >>>(cellsDev, height, width, arraySize, cellsPerThread); // running evolution iteration	
	}
	clock_t end = clock();
	
	cudaMemcpy(cells, cellsDev, sizeof(int)*arraySize, cudaMemcpyDeviceToHost); // copying cells back from gpu to cpu	
	cudaFree(cellsDev);

	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	cout << elapsed_secs;

	ofstream myfile;
	myfile.open("para4.txt");
	for (int i = 0; i < arraySize; i++) {
		myfile << cells[i] << endl;
	}
	free(cells);
	myfile.close();
	cin >> b;
	return 0;
}
