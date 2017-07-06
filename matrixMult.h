#include <cuda_runtime.h>
#define DIM(i, j, N) i+j*N



float comparar(float* h_C, float* h_C_cuda, int N){
	float sum_diff = 0.0;

	for (int i = 0; i < N; ++i)
	{
		for (int j = 0; j < N; ++j)
		{
			sum_diff += fabs(h_C[DIM(i,j,N)] - h_C_cuda[DIM(i,j,N)]);
		}
	}

	return sum_diff;
}

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void
matMultCuda(const float *A, const float *B, float *C, int N)
{
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;


    float sum = 0.0;

    if(i < N && j < N){
    	for (int k = 0; k < N; ++k)
    	{
    		sum += A[DIM(i,k,N)] * B[DIM(k,j,N)];
    	}
    	C[DIM(i,j,N)] = sum;
    }
    
}

//Calcula A*B^T
__global__ void
matMultTransCuda(const float *A, const float *B, float *C, int N)
{
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;


    float sum = 0.0;

    if(i < N && j < N){
    	for (int k = 0; k < N; ++k)
    	{
    		sum += A[DIM(i,k,N)] * B[DIM(j,k,N)];
    	}
    	C[DIM(i,j,N)] = sum;
    }
    
}





__global__ void
matTrans(float *A, int N)
{
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;


    float aux = 0.0;

    if(i > j && i < N && j < N){
    	aux = A[DIM(i,j,N)];
    	A[DIM(i,j,N)] = A[DIM(j,i,N)];
    	A[DIM(j,i,N)] = aux;
    }
    
}



void matMult(const float *A, const float *B, float *C, int N){
	for (int i = 0; i < N; ++i)
	{
		for (int j = 0; j < N; ++j)
		{	
			C[DIM(i,j,N)] = 0.0;
			for (int k = 0; k < N; ++k)
			{
				C[DIM(i,j,N)] += A[DIM(i,k,N)]*B[DIM(k,j,N)];
			}
		}
	}
}


void inicMat(float *A, int N, float valor){
	for (int i = 0; i < N; ++i)
	{
		for (int j = 0; j < N; ++j)
		{
			A[DIM(i,j,N)] = rand()%6;
		}
	}
}

void printMat(float *A, int N){
	for (int i = 0; i < N; ++i)
	{
		for (int j = 0; j < N; ++j)
		{
			printf("%8.2f",A[DIM(i,j,N)]);
		}

		printf("\n");
	}
}
