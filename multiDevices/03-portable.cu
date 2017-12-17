/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Este exemplo mostra como podemos utilizar mais de duas GPUs sem a necessidade
 *   de copiar os dados entre CPU e GPU. Para isso precisamos traduzir o endereco
 *   do espaco de memoria da CPU para seu equivalente no espaco de memoria da GPU.
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
	while (tid < size) {
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
	while (i != 0) {
		if (cacheIndex < i)
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
	int     offset;
	float   *a;
	float   *b;
	float   returnValue;
};



void* routine( void *pvoidData ) {

	DataStruct  *data = (DataStruct*)pvoidData;
	if( data->deviceID != 0 ) {
		cudaSetDevice( data->deviceID );
		cudaSetDeviceFlags( cudaDeviceMapHost );
	}

	int     size = data->size;
	float   *h_a, *h_b, h_c, *h_partial_c;
	float   *d_a, *d_b, *d_partial_c;

	// allocate memory on the CPU side
	h_a = data->a;
	h_b = data->b;
	h_partial_c = (float*)malloc( blocksPerGrid*sizeof(float) );

	// allocate the memory on the GPU
	cudaHostGetDevicePointer( &d_a, h_a, 0 );
	cudaHostGetDevicePointer( &d_b, h_b, 0 );
	cudaMalloc( (void**)&d_partial_c, blocksPerGrid*sizeof(float) );

	// offset 'd_a' and 'd_b' to where this GPU is gets it data
	d_a += data->offset;
	d_b += data->offset;

	dot<<<blocksPerGrid,threadsPerBlock>>>( size, d_a, d_b, d_partial_c );

	// copy the array 'c' back from the GPU to the CPU
	cudaMemcpy( h_partial_c, d_partial_c, blocksPerGrid*sizeof(float), cudaMemcpyDeviceToHost );

	// finish up on the CPU side
	h_c = 0;
	for( int i = 0 ; i < blocksPerGrid ; i++ ) {
		h_c += h_partial_c[i];
	}

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

	cudaDeviceProp  prop;
	for( int i = 0 ; i < deviceCount ; i++ ) {
		cudaGetDeviceProperties( &prop, i );
		if( prop.canMapHostMemory != 1 ) {
			printf( "Device %d can not map memory.\n", i );
			return 0;
		}
	}

	float *a, *b;
	cudaSetDevice( 0 );
	cudaSetDeviceFlags( cudaDeviceMapHost );
	cudaHostAlloc( (void**)&a, N*sizeof(float), 
				     cudaHostAllocWriteCombined | cudaHostAllocPortable | cudaHostAllocMapped );
	cudaHostAlloc( (void**)&b, N*sizeof(float), 
				     cudaHostAllocWriteCombined | cudaHostAllocPortable | cudaHostAllocMapped );

	// fill in the host memory with data
	for( int i = 0 ; i < N ; i++ ) {
		a[i] = i;
		b[i] = i*2;
	}

	// prepare for multithread
	DataStruct  data[deviceCount];
	for( int i = 0 ; i < deviceCount ; i++ ) {
		data[i].deviceID = i;
		data[i].offset = i*N/deviceCount;
		data[i].size = N/deviceCount;
		data[i].a = a;
		data[i].b = b;
	}

	CUTThread   thread[deviceCount];
	for( int i = 1 ; i < deviceCount ; i++ )
		thread[i] = start_thread( routine, &(data[i]) );
	routine( &(data[0]) );
	for( int i = 1 ; i < deviceCount ; i++ )
		end_thread( thread[i] );


	// free memory on the CPU side
	cudaFreeHost( a );
	cudaFreeHost( b );

	float resp = 0.0f;
	for( int i = 0 ; i < deviceCount ; i++ )
		resp += data[i].returnValue;
	printf( "Value calculated:  %f\n", resp );

	return 0;
}

