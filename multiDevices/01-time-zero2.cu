/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Neste exemplo, faremos um produto interno de dois vetores.
 * O kernel "dot()" sera chamado de duas formas.
 * A funcao "malloc_test()" faz a alocao dos vetores no espaco de memoria
 *   da GPU e copia via cudaMemCopy os vetores para a GPU, como foi visto.
 * Ja a funcao "cuda_host_alloc_test()", aloca os buffers apenas no espaco
 *   de memoria da CPU, tornando desnecessarias as copias entre GPU e CPU.
 *   No entanto para que a GPU consiga acessar a memoria da CPU, precisamos
 *   precisamos transformar os enderecos dos vetores para os enderecos
 *   equivalentes no espaco de memoria da GPU.
 * 
 */

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include "rf-time.h"

#define imin(a,b) (a<b?a:b)

const int N = 4 * 1024 * 1024;
const int threadsPerBlock = 32;
const int blocksPerGrid = imin( 32, (N+threadsPerBlock-1) / threadsPerBlock );


__global__ void dot( int size, float *a, float *b, float *c ) {

	__shared__ float cache[threadsPerBlock];

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int cacheIndex = threadIdx.x;

	if( cacheIndex == 0 )
		c[blockIdx.x] = 0;
	__syncthreads();

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




float malloc_test( int size ) {

	cudaEvent_t     start, stop;
	float           *h_a, *h_b, h_c, *h_partial_c;
	float           *d_a, *d_b, *d_partial_c;
	float           elapsedTime;

	cudaEventCreate( &start );
	cudaEventCreate( &stop );

	// allocate memory on the CPU side
	h_a = (float*)malloc( size*sizeof(float) );
	h_b = (float*)malloc( size*sizeof(float) );
	h_partial_c = (float*)malloc( blocksPerGrid*sizeof(float) );

	// allocate the memory on the GPU
	cudaMalloc( (void**)&d_a, size*sizeof(float) );
	cudaMalloc( (void**)&d_b, size*sizeof(float) );
	cudaMalloc( (void**)&d_partial_c, blocksPerGrid*sizeof(float) );

	// fill in the host memory with data
	for( int i = 0 ; i < size ; i++ ) {
		h_a[i] = i;
		h_b[i] = i*2;
	}

	cudaEventRecord( start, 0 );
	// copy the arrays 'a' and 'b' to the GPU
	cudaMemcpy( d_a, h_a, size*sizeof(float), cudaMemcpyHostToDevice );
	cudaMemcpy( d_b, h_b, size*sizeof(float), cudaMemcpyHostToDevice ); 

	dot<<<blocksPerGrid,threadsPerBlock>>>( size, d_a, d_b, d_partial_c );

	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	cudaEventElapsedTime( &elapsedTime, start, stop );

	// copy the array 'c' back from the GPU to the CPU
	cudaMemcpy( h_partial_c, d_partial_c, blocksPerGrid*sizeof(float), cudaMemcpyDeviceToHost );
	// finish up on the CPU side
	h_c = 0.0f;
	for( int i = 0 ; i < blocksPerGrid ; i++ ) {
		h_c += h_partial_c[i];
	}
	printf( "Value calculated:  %f\n", h_c );

	cudaFree( d_a );
	cudaFree( d_b );
	cudaFree( d_partial_c );

	// free memory on the CPU side
	free( h_a );
	free( h_b );
	free( h_partial_c );

	// free events
	cudaEventDestroy( start );
	cudaEventDestroy( stop );

	return elapsedTime;

}



float cuda_host_alloc_test( int size ) {

	cudaEvent_t     start, stop;
	float           *h_a, *h_b, h_c, *h_partial_c;
	float           *d_a, *d_b, *d_partial_c;
	float           elapsedTime;

	cudaEventCreate( &start );
	cudaEventCreate( &stop );

	// allocate the memory on the CPU
	cudaHostAlloc( (void**)&h_a, size*sizeof(float), 
				     cudaHostAllocWriteCombined | cudaHostAllocMapped );
	cudaHostAlloc( (void**)&h_b, size*sizeof(float), 
				     cudaHostAllocWriteCombined | cudaHostAllocMapped );
	cudaHostAlloc( (void**)&h_partial_c, blocksPerGrid*sizeof(float), 
				     cudaHostAllocMapped );

	// find out the GPU pointers
	cudaHostGetDevicePointer( &d_a, h_a, 0 );
	cudaHostGetDevicePointer( &d_b, h_b, 0 );
	cudaHostGetDevicePointer( &d_partial_c, h_partial_c, 0 );

	// fill in the host memory with data
	for( int i = 0 ; i < size ; i++ ) {
		h_a[i] = i;
		h_b[i] = i*2;
	}

	cudaEventRecord( start, 0 );

	dot<<< blocksPerGrid, threadsPerBlock >>>( size, d_a, d_b, d_partial_c );

	cudaThreadSynchronize();

	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	cudaEventElapsedTime( &elapsedTime, start, stop );

	// finish up on the CPU side
	h_c = 0;
	for( int i = 0 ; i < blocksPerGrid ; i++ ) {
		h_c += h_partial_c[i];
	}

	cudaFreeHost( h_a );
	cudaFreeHost( h_b );
	cudaFreeHost( h_partial_c );

	// free events
	cudaEventDestroy( start );
	cudaEventDestroy( stop );

	printf( "Value calculated:  %f\n", h_c );

	return elapsedTime;

} 



int main( int argc, char **argv ) {

	cudaDeviceProp  prop;
	int whichDevice;

	cudaGetDevice( &whichDevice );

	cudaGetDeviceProperties( &prop, whichDevice );
	if( prop.canMapHostMemory != 1 ) {
		printf( "Device can not map memory.\n" );
		return 0;
	}

	int size = N;
	if( argc > 1 ) {
		int n = atoi( argv[1] );
		if( n > 0 )
			size *= n;
	}
	printf( "Vector size = %d elements\n", size );
	printf( "Memoria necessaria = %lud bytes\n", size*sizeof( float ) );

	float elapsedTime;

	cudaSetDeviceFlags( cudaDeviceMapHost );

	// try it with malloc
	elapsedTime = malloc_test( size );
	printf( "Time using cudaMalloc:  %5.2f ms\n", elapsedTime );

	// now try it with cudaHostAlloc
	elapsedTime = cuda_host_alloc_test( size );
	printf( "Time using cudaHostAlloc:  %5.2f ms\n", elapsedTime );

}
