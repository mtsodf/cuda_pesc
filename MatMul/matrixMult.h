#include <cuda_runtime.h>
#define DIM(i, j, N) (i)+(j)*(N)

#define TILE_SIZE 32



__global__ void zerarCuda(float *C, int N){


    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;

	if(i<N && j<N){
		C[DIM(i,j,N)] = 0.0;
	}

}

__global__ void matMultTileCuda(const float *A, const float *B, float *C, int N){
	__shared__ float a_tile[TILE_SIZE][TILE_SIZE], b_tile[TILE_SIZE][TILE_SIZE];

	int qtd_tiles = N/TILE_SIZE + (N%TILE_SIZE==0?0:1);

    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;

	int offset;

	float sum = 0.0;


		

		for (int tile_ind = 0; tile_ind < qtd_tiles; ++tile_ind) {
			offset = tile_ind*TILE_SIZE;
			if(i<N && offset+threadIdx.x< N){
				a_tile[threadIdx.y][threadIdx.x] = A[DIM(i, offset+threadIdx.x, N)];
			} else{
				a_tile[threadIdx.y][threadIdx.x] = 0.0;
			}

			if(threadIdx.y+offset<N && j< N){
				b_tile[threadIdx.y][threadIdx.x] = B[DIM(threadIdx.y+offset, j, N)];
			} else{
				b_tile[threadIdx.y][threadIdx.x] = 0.0;
			}
			
			__syncthreads();

			for (int k = 0; k < TILE_SIZE; ++k) {
				sum += a_tile[threadIdx.y][k]*b_tile[k][threadIdx.x];
			}

			__syncthreads();
		}

		if(i<N && j<N) C[DIM(i,j,N)] = sum;
	

}

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
	float aux;
	int i,j;
	for (j = 0; j < N; ++j)
	{
		for (i = 0; i < N; ++i)
		{	
			aux = 0.0;	
			for (int k = 0; k < N; ++k)
			{
				aux += A[DIM(i,k,N)]*B[DIM(k,j,N)];
			}
			C[DIM(i,j,N)] = aux;
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
