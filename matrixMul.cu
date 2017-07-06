
#include <stdio.h>
#include <sys/time.h>
#include <sys/time.h>
#include <stdlib.h>
#include "matrixMult.h"
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>






struct timeval  tv1, tv2;



int main(int argc, char const *argv[])
{

	cudaError_t err = cudaSuccess;

	int N = atoi(argv[1]);

	int size = N*N*sizeof(float);

	printf("Hello World!\n");

	if(argc < 2) {
		printf("Passe o tamanho da matriz como referencia.\n");
		return -1;
	}

	


	printf("Dimensao da matriz %d\n", N);


	float *h_A, *h_B, *h_C;


	h_A = (float *) malloc(size);
	h_B = (float *) malloc(size);
	h_C = (float *) malloc(size);


	inicMat(h_A, N, 1.0);
	inicMat(h_B, N, 1.0);
	inicMat(h_C, N, 0.0);

	gettimeofday(&tv1, NULL);

	matMult(h_A, h_B, h_C, N);

	gettimeofday(&tv2, NULL);

	printf ("Total time = %f seconds\n\n",
         (double) (tv2.tv_usec - tv1.tv_usec) / 1000 +
         (double) (tv2.tv_sec - tv1.tv_sec)*1000);



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

	gettimeofday(&tv1, NULL);

	matMultCuda<<<grid, threads>>>(d_A, d_B, d_C, N);
	
	gettimeofday(&tv2, NULL);

	printf ("Total time = %f seconds\n",
         (double) (tv2.tv_usec - tv1.tv_usec) / 1000 +
         (double) (tv2.tv_sec - tv1.tv_sec) * 1000);




	err = cudaMemcpy(h_C_cuda, d_C, size, cudaMemcpyDeviceToHost);




	printf("Diff matMultCuda = %f\n\n", comparar(h_C, h_C_cuda, N));
	zerarCuda<<<grid, threads>>>(d_C, N);

	gettimeofday(&tv1, NULL);

	matMultTileCuda<<<grid, threads>>>(d_A, d_B, d_C, N);

	gettimeofday(&tv2, NULL);

	printf ("Total time = %f seconds\n",
         (double) (tv2.tv_usec - tv1.tv_usec) / 1000 +
         (double) (tv2.tv_sec - tv1.tv_sec) * 1000);



	err = cudaMemcpy(h_C_cuda, d_C, size, cudaMemcpyDeviceToHost);

	printf("Diff matMultTileCuda = %f\n\n", comparar(h_C, h_C_cuda, N));
	zerarCuda<<<grid, threads>>>(d_C, N);

	if(N <= 10) {
		
		printf("Matriz Calculada Normal\n");
		printMat(h_C, N);		

		printf("Matriz Calculada Cuda\n");
		printMat(h_C_cuda, N);
	}

	matTrans<<<grid, threads>>>(d_B, N);

	gettimeofday(&tv1, NULL);

	matMultTransCuda<<<grid, threads>>>(d_A, d_B, d_C, N);
	
	gettimeofday(&tv2, NULL);

	printf ("Total time = %f seconds\n",
         (double) (tv2.tv_usec - tv1.tv_usec) / 1000 +
         (double) (tv2.tv_sec - tv1.tv_sec) * 1000);	



	err = cudaMemcpy(h_C_cuda, d_C, size, cudaMemcpyDeviceToHost);

	printf("Diff matMultTransCuda = %f\n\n", comparar(h_C, h_C_cuda, N));
	



	
	free(h_A);
	free(h_B);
	free(h_C);

    // Free device global memory
    err = cudaFree(d_A);
    err = cudaFree(d_B);
    err = cudaFree(d_C);



	return 0;
}
