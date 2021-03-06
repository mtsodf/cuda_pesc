
#include <stdio.h>
#include <sys/time.h>
#include <sys/time.h>
#include <stdlib.h>
#include "matrixMult.h"
#include "rf-time.h"
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>


struct timeval  tv1, tv2;



int main(int argc, char const *argv[])
{

	double start_time, gpu_time;
    
    cudaEvent_t startCuda, stopCuda; float ms;

	cudaError_t err;

	int N = atoi(argv[1]);

	int size = N*N*sizeof(float);

	if(argc < 2) {
		printf("Passe o tamanho da matriz como referencia.\n");
		return -1;
	}

    cudaEventCreate(&startCuda); cudaEventCreate(&stopCuda);

	int deviceCount;
    cudaGetDeviceCount(&deviceCount);
	printf("Device Count %d\n", deviceCount);
    // This function call returns 0 if there are no CUDA capable devices.
    if( deviceCount == 0 ) {
        printf("There is no device supporting CUDA\n");
        exit( 1 );
    }

    if(deviceCount < 2){
    	printf("Nao tem placa grafica disponivel\n");
    }

    cudaSetDevice(deviceCount - 1);	


	printf("Dimensao da matriz %d\n", N);


	float *h_A, *h_B, *h_C;


	h_A = (float *) malloc(size);
	h_B = (float *) malloc(size);
	h_C = (float *) malloc(size);


	inicMat(h_A, N, 1.0);
	inicMat(h_B, N, 1.0);
	inicMat(h_C, N, 0.0);

	start_time = get_clock_msec();

	matMult(h_A, h_B, h_C, N);

	gpu_time = get_clock_msec() - start_time;  

	printf ("Total time = %f miliseconds\n\n", gpu_time);



	float *d_A, *d_B, *d_C;
	float *h_C_cuda;


	h_C_cuda = (float*) malloc(size);

	//Alocando vetores no cuda
	cudaMalloc((void **)&d_A, size);
	cudaMalloc((void **)&d_B, size);
	cudaMalloc((void **)&d_C, size);



	err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

	int block_size_x = TILE_SIZE, block_size_y = TILE_SIZE;

    dim3 threads(block_size_x, block_size_y, 1);
    int grid_x = N/block_size_x + (N%block_size_x==0?0:1);
    int grid_y = N/block_size_y + (N%block_size_y==0?0:1);
    dim3 grid(grid_x, grid_y, 1);

	printf("Grid(%d, %d) - Bloco(%d, %d)\n\n", grid_x, grid_y, block_size_x, block_size_y);

	cudaDeviceSynchronize();
	start_time = get_clock_msec();
    
//    cudaEventRecord(startCuda, 0);
	matMultCuda<<<grid, threads>>>(d_A, d_B, d_C, N);
//    cudaEventRecord(stopCuda);
//    cudaEventSynchronize(stopCuda);
//    cudaEventElapsedTime(&ms, startCuda, stopCuda);
  
    cudaDeviceSynchronize();
	gpu_time = get_clock_msec() - start_time;  



	err = cudaMemcpy(h_C_cuda, d_C, size, cudaMemcpyDeviceToHost);

	printf("Diff matMultCuda = %f\n", comparar(h_C, h_C_cuda, N));
	printf ("Total time = %f miliseconds %f\n\n", gpu_time, ms);

	zerarCuda<<<grid, threads>>>(d_C, N);

	cudaDeviceSynchronize();
	start_time = get_clock_msec();
	matMultTileCuda<<<grid, threads>>>(d_A, d_B, d_C, N); cudaDeviceSynchronize();
	gpu_time = get_clock_msec() - start_time;  

	err = cudaMemcpy(h_C_cuda, d_C, size, cudaMemcpyDeviceToHost);

	printf("Diff matMultTileCuda = %f\n", comparar(h_C, h_C_cuda, N));
	printf ("Total time = %f miliseconds\n\n", gpu_time);


	zerarCuda<<<grid, threads>>>(d_C, N);

	if(N <= 10) {
		
		printf("Matriz Calculada Normal\n");
		printMat(h_C, N);		

		printf("Matriz Calculada Cuda\n");
		printMat(h_C_cuda, N);
	}

	matTrans<<<grid, threads>>>(d_B, N);

	cudaDeviceSynchronize();
	start_time = get_clock_msec();
	matMultTransCuda<<<grid, threads>>>(d_A, d_B, d_C, N); cudaDeviceSynchronize();
	gpu_time = get_clock_msec() - start_time;  

	err = cudaMemcpy(h_C_cuda, d_C, size, cudaMemcpyDeviceToHost);

	printf("Diff matMultTransCuda = %f\n", comparar(h_C, h_C_cuda, N));
	printf ("Total time = %f miliseconds\n\n", gpu_time);

	
	free(h_A);
	free(h_B);
	free(h_C);

    // Free device global memory
    err = cudaFree(d_A);
    err = cudaFree(d_B);
    err = cudaFree(d_C);



	return 0;
}
