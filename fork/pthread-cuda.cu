//--------------------------------------------------
// Autor: Ricardo Farias
// Data : 29 Out 2011
// Goal : Increment a variable in the graphics card
//--------------------------------------------------

/***************************************************************************************************
	Includes
***************************************************************************************************/
#include <cuda.h>
#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>
#include <pthread.h> 


bool volatile lock = true;


__global__ void somaUm( int *a ) {

	atomicAdd( a, 1 );

}


void *f1( void *x ) {

	int i;
	i = *(int *)x;
	printf("\tf1: %d\n",i);

	int h_a = 0;
	int deviceCount = 0;
		
	cudaGetDeviceCount( &deviceCount );
	// This function call returns 0 if there are no CUDA capable devices.
	if( deviceCount == 0 ) {
		printf("There is no device supporting CUDA\n");
		exit( 1 );
	}
 
  cudaSetDevice(1);
		
	int   *d_a;       // Pointer to host & device arrays
		
	 cudaMalloc( (void **) &d_a, sizeof( int ) ) ;
		
	// Copy array to device
	 cudaMemcpy( d_a, &h_a, sizeof(int), cudaMemcpyHostToDevice ) ;
		
	printf( "Valor de a antes = %d\n", h_a );
	//------------------------------------------------

	lock = true;

	somaUm<<< 1, 5 >>>( d_a );

	 cudaMemcpy( &h_a, d_a, sizeof(int), cudaMemcpyDeviceToHost ) ;
	printf( "\tValor de a depois da chamada do filho f1 = %d\n", h_a );
	lock = false;
	pthread_exit(0); 

}


void *f2( void *x ) {

	int i;
	i = *(int *)x;
	while( lock );
	printf("\tf2: %d\n",i);
	int h_a = 0;
	int deviceCount = 0;
		
	cudaGetDeviceCount( &deviceCount );
	// This function call returns 0 if there are no CUDA capable devices.
	if( deviceCount == 0 ) {
		printf("There is no device supporting CUDA\n");
		exit( 1 );
	}
		
	int   *d_a;       // Pointer to host & device arrays
		
	 cudaMalloc( (void **) &d_a, sizeof( int ) ) ;
		
	// Copy array to device
	 cudaMemcpy( d_a, &h_a, sizeof(int), cudaMemcpyHostToDevice ) ;
		
	printf( "Valor de a antes = %d\n", h_a );
	//------------------------------------------------

	lock = true;

	somaUm<<< 5, 5 >>>( d_a );

	 cudaMemcpy( &h_a, d_a, sizeof(int), cudaMemcpyDeviceToHost ) ;
	printf( "\tValor de a depois da chamada do filho f2 = %d\n", h_a );
	pthread_exit(0); 

}


int main() {

	pthread_t f2_thread, f1_thread; 
	void *f2(void*), *f1(void*);
	int i1,i2;

	i1 = 1;
	i2 = 2;

	pthread_create( &f1_thread, NULL, f1, &i1 );
	pthread_create( &f2_thread, NULL, f2, &i2 );

	pthread_join( f1_thread, NULL );
	pthread_join( f2_thread, NULL );
	
	return 1;
	
}
