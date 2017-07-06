
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
	int block_size_x = 16, block_size_y = 16;
	
	cudaError_t err = cudaSuccess;

	if(argc < 1) {
		printf("Passe o tamanho da matriz como referencia.\n");
		return -1;
	}

	if(argc > 2){
		block_size_x = atoi(argv[2]), block_size_y = atoi(argv[3]);
	}

	printf("Bloco dim = %d %d\n", block_size_x, block_size_y);
	printf("Qtd Threads/bloco = %d\n", block_size_x*block_size_y);


	int N = atoi(argv[1]);

	int size = N*N*sizeof(float);	


	printf("Dimensao da matriz %d\n", N);


	float *h_A, *h_B;


	h_A = (float *) malloc(size);
	h_B = (float *) malloc(size);



	inicMat(h_A, N, 1.0);
	inicMat(h_B, N, 1.0);


	float *d_A, *d_B, *d_C;


	//Alocando vetores no cuda
	cudaMalloc((void **)&d_A, size);
	cudaMalloc((void **)&d_B, size);
	cudaMalloc((void **)&d_C, size);



	err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);




    dim3 threads(block_size_x, block_size_y, 1);
    int grid_x = N/block_size_x + (N%block_size_x==0?0:1);
    int grid_y = N/block_size_y + (N%block_size_y==0?0:1);
    dim3 grid(grid_x, grid_y, 1);

    matTrans<<<grid, threads>>>(d_B, N);

	gettimeofday(&tv1, NULL);

	matMultTransCuda<<<grid, threads>>>(d_A, d_B, d_C, N);
	
	gettimeofday(&tv2, NULL);

	printf ("Total time = %f seconds\n",
         (double) (tv2.tv_usec - tv1.tv_usec) / 1000 +
         (double) (tv2.tv_sec - tv1.tv_sec) * 1000);



	
	free(h_A);
	free(h_B);

    // Free device global memory
    err = cudaFree(d_A);
    err = cudaFree(d_B);
    err = cudaFree(d_C);



	return 0;
}