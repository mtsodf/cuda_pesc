/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Neste exemplo iremos ver como usar duas GPUs em paralelo, para
 *   calcular o produto interno de dois vetores, sendo que cada
 *   GPU executara a metade da tarefa. Usaremos pthread para
 *   permitir a chamada de cada GPU. Perceba que a funcao "routine"
 *   chamada por cada thread perfaz a alocacao dos seus buffers
 *   na GPU.
 * 
 */

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include "rf-time.h"
#include "book.h"

#define imin(a,b) (a<b?a:b)

#define   N  (33*1024*1024)
const int threadsPerBlock = 256;
const int blocksPerGrid = imin( 32, (N/2+threadsPerBlock-1) / threadsPerBlock );


__global__ void dot( int size, float *a, float *b, float *c ) {

	__shared__ float cache[threadsPerBlock];
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int cacheIndex = threadIdx.x;

	float   temp = 0;
	while( tid < size ) {
		temp += a[tid] * b[tid];
		tid += blockDim.x * gridDim.x;
	}
    
	// set the cache values
	cache[cacheIndex] = temp;
    
	// synchronize threads in this block
	__syncthreads();

	// for reductions, threadsPerBlock must be a power of 2
	// because of the following code
	int i = blockDim.x/2;
	while( i != 0 ) {
		if( cacheIndex < i )
			cache[cacheIndex] += cache[cacheIndex + i];
		__syncthreads();
		i /= 2;
	}

	if( cacheIndex == 0 )
		c[blockIdx.x] = cache[0];

}


struct DataStruct {
	int     deviceID;
	int     size;
	float   *a;
	float   *b;
	float   returnValue;
};


void *routine( void *pvoidData ) {

	DataStruct  *data = (DataStruct*)pvoidData;
	cudaSetDevice( data->deviceID );

	int     size = data->size;
	float   *h_a, *h_b, h_c, *h_partial_c;
	float   *d_a, *d_b, *d_partial_c;

	// allocate memory on the CPU side
	h_a = data->a;
	h_b = data->b;
	h_partial_c = (float*)malloc( blocksPerGrid*sizeof(float) );

	// allocate the memory on the GPU
	cudaMalloc( (void**)&d_a, size*sizeof(float) );
	cudaMalloc( (void**)&d_b, size*sizeof(float) );
	cudaMalloc( (void**)&d_partial_c, blocksPerGrid*sizeof(float) );

	// copy the arrays 'h_a' and 'h_b' to the GPU
	cudaMemcpy( d_a, h_a, size*sizeof(float), cudaMemcpyHostToDevice );
	cudaMemcpy( d_b, h_b, size*sizeof(float), cudaMemcpyHostToDevice ); 

	dot<<<blocksPerGrid,threadsPerBlock>>>( size, d_a, d_b, d_partial_c );

	// copy the array 'd_partial_c' back from the GPU to the CPU
	cudaMemcpy( h_partial_c, d_partial_c, blocksPerGrid*sizeof(float), cudaMemcpyDeviceToHost );

	// finish up on the CPU side
	h_c = 0;
	for( int i = 0 ; i < blocksPerGrid ; i++ ) {
		h_c += h_partial_c[i];
	}

	cudaFree( d_a );
	cudaFree( d_b );
	cudaFree( d_partial_c );

	// free memory on the CPU side
	free( h_partial_c );

	data->returnValue = h_c;
	return 0;

}



int main( void ) {

	int deviceCount;
	cudaGetDeviceCount( &deviceCount );
	if( deviceCount < 2 ) {
		printf( "We need at least two compute 1.0 or greater "
			"devices, but only found %d\n", deviceCount );
		return 0;
	}

	float   *a = (float*)malloc( sizeof(float) * N );
	if( a == NULL ) {
		printf( "Host memory failed in %s at line %d\n", __FILE__, __LINE__ );
		exit( EXIT_FAILURE );
	}


	float   *b = (float*)malloc( sizeof(float) * N );
	if( b == NULL ) {
		printf( "Host memory failed in %s at line %d\n", __FILE__, __LINE__ );
		exit( EXIT_FAILURE );
	}

	// fill in the host memory with data
	for( int i = 0 ; i < N ; i++ ) {
		a[i] = i;
		b[i] = i*2;
	}

	// prepare for multithread
	DataStruct  data[2];
	data[0].deviceID = 0;
	data[0].size = N/2;
	data[0].a = a;
	data[0].b = b;

	data[1].deviceID = 1;
	data[1].size = N/2;
	data[1].a = a + N/2;
	data[1].b = b + N/2;

	CUTThread   thread = start_thread( routine, &(data[0]) );
	routine( &(data[1]) );
	end_thread( thread );


	// free memory on the CPU side
	free( a );
	free( b );

	printf( "Value calculated:  %f\n", data[0].returnValue + data[1].returnValue );

	return 0;

}
