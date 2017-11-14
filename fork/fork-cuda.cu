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

/* Descritores herdados pelos processos filhos */

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>


__global__ void somaUm( int *a ) {

	atomicAdd( a, 1 );

}


int main() {
	
	bool *lock;
	int pid ;
  int teste = 10;
	
 
  lock = (bool*)malloc(1*sizeof(bool));
  lock[0] = true; 
	int h_a = 0;
	int deviceCount = 0;

	printf("Creating the son process\n");
	
	pid = fork();
 
  printf("Endereco do Lock %d\n", &lock);
	
	if( pid == -1 ) { /* erro */
		
		perror("impossivel de criar um filho") ;
		exit(-1); 
		
	} else if( pid == 0 ) { /* filho */
    teste = 20;
		printf("Endereco do Lock do filho %d\n", &teste);
		printf("\tO filho espera o pai chamar o kernel primeiro.\n") ;
		sleep( 10 );
		while( lock[0] ){
      //printf("Valor do Lock no filho %d \n.", lock);
      sleep(1);
    };
		printf("\tO Pai liberou o lock.\nChamando o kernel pelo filho com 5 threads.\n") ;

		//-------------------------------------------------
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

		lock[0] = true;

		somaUm<<< 1, 5 >>>( d_a );
		 cudaMemcpy( &h_a, d_a, sizeof(int), cudaMemcpyDeviceToHost ) ;
		printf( "\tValor de a depois da chamada do filho = %d\n", h_a );

		printf("\tO Filho se mata!!!\n") ;
		lock[0] = false;
		exit(1) ;

	} else { /* pai */
		printf("Endereco do Lock do pai %d\n", &lock);
		lock[0] = true;
		printf( "O pid do meu filho e': %d\n", pid );
		printf( "O Pai pega o controle para chamar o kernel com 1 thread...\n" );
		
		cudaGetDeviceCount( &deviceCount );
		// This function call returns 0 if there are no CUDA capable devices.
		if( deviceCount == 0 ) {
			printf("There is no device supporting CUDA\n");
			exit( 1 );
		}
		cudaSetDevice(1);
		int   *d_a;       // Pointer to host & device arrays
		
		 cudaMalloc( (void **) &d_a, sizeof( int ) );
		
		// Copy array to device
		 cudaMemcpy( d_a, &h_a, sizeof(int), cudaMemcpyHostToDevice ) ;
		
		printf( "Valor de a antes = %d\n", h_a );

		somaUm<<< 1, 1 >>>( d_a );
		 cudaMemcpy( &h_a, d_a, sizeof(int), cudaMemcpyDeviceToHost ) ;
		printf( "Valor de a depois da chamada do Pai = %d\n", h_a );

		cudaFree( d_a ); 

		printf( "O Pai libera o lock e espera o filho chamar o kernel...\n" );

		lock[0] = false;
    teste = 30;
    printf(" Lock liberado pelo pai lock = %d\n", &teste);
		sleep( 3 );
		while( lock[0] );
		printf( "O Pai se mata!!!\n" );

	}

	exit(0);

}
