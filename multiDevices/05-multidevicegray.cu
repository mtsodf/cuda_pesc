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

#define BLOCK_SIZE 32

using namespace std;


/**************************************************************************************************/

__host__ void erro( const char tipoDeErro[] ) {

	fprintf( stderr, "%s\n", tipoDeErro );
	exit(0);

}


/**************************************************************************************************/
__host__ void savePPM( char *fname, unsigned char *buffer, int width, int height ) {

	if( !buffer ) {
		cout << "Image not saved. This ViewPoint Class in NOT FULL." << endl;
		return;
	}
	FILE *f = fopen( fname, "wb" );
	if( f == NULL ) erro( "Error writting PPM file." );

	fprintf( f, "P6\n# Written in the CUDA course\n%u %u\n%u\n", width, height, 255 );
	fwrite( buffer, 3, width*height, f );
	fclose(f);

}

__host__ void readPPM( char *fname, unsigned char **buffer, int *width, int *height ) {

	char aux[256];
	FILE *f = fopen( fname, "rb" );
	if( f == NULL )
		erro( "Error reading PPM image" );

	fgets( aux, 256, f );
	fgets( aux, 256, f );
	fgets( aux, 256, f );
	sscanf( aux, "%d %d", width, height );
	fgets( aux, 256, f );

	if( *buffer ) {
		free( *buffer );
	}

	int size = 3*(*width)*(*height)*sizeof( unsigned char );
	cout << "Image dimension: (" << *width << "," << *height <<")\n";

	if( ( *buffer = (unsigned char *)malloc( size ) ) == NULL )
		erro( "Error allocating image" );

	fread( *buffer, 3, (*width)*(*height), f );
	fclose( f );

}


/**************************************************************************************************/
__global__ void cinzaGPU1d( unsigned char *image1,
                          unsigned char *res, int pixels ) {

        int i = threadIdx.x + blockIdx.x*blockDim.x;
        int cinza;

        if( i < pixels ) {

                int idx = 3*i;
                int r = image1[ idx+2 ];
                int g = image1[ idx+1 ];
                int b = image1[ idx   ];
                
                cinza  = (30*r + 59*g + 11*b)/100;

                res[ idx+2 ] = (unsigned char)cinza;
                res[ idx+1 ] = (unsigned char)cinza;
                res[ idx   ] = (unsigned char)cinza;

         }
}


struct DataStruct {
	int     deviceID;
	int     init;
	int     qtdPixels;
	unsigned char   *image;
	unsigned char   *out;
};


void *routine( void *pvoidData ) {

	DataStruct  *data = (DataStruct*)pvoidData;
	cudaSetDevice( data->deviceID );

	int     size = 3*data->qtdPixels*sizeof(unsigned char);
	unsigned char   *d_image, *d_res;
	int qtdPixels = data->qtdPixels;


	// allocate the memory on the GPU
	cudaMalloc( (void**)&d_image, size);
	cudaMalloc( (void**)&d_res, size);

	// copy the arrays 'h_a' and 'h_b' to the GPU
	cudaMemcpy( d_image, data->image, size*sizeof(float), cudaMemcpyHostToDevice ); 


	int threadsPerBlock = BLOCK_SIZE;
	int blocksPerGrid = qtdPixels/BLOCK_SIZE + qtdPixels%BLOCK_SIZE==0?0:1;

	cinzaGPU1d<<<blocksPerGrid,threadsPerBlock>>>( d_image, d_res, qtdPixels);

	// copy the array 'd_partial_c' back from the GPU to the CPU
	cudaMemcpy( d_res, data->out+3*data->init, size, cudaMemcpyDeviceToHost );


	cudaFree( d_image );
	cudaFree( d_res );

}



int main( int argc, char *argv[] ) {

	int deviceCount;
	cudaGetDeviceCount( &deviceCount );
	if( deviceCount < 2 ) {
		printf( "We need at least two compute 1.0 or greater "
			"devices, but only found %d\n", deviceCount );
		return 0;
	}

	cout << "Program for image treatment " << endl;

	int h_width, h_height;
	unsigned char*  h_image = NULL, *h_res = NULL;

	readPPM( argv[1], &h_image, &h_width, &h_height );

	int qtdPixels = h_width*h_height;

	int size = 3*h_width*h_height*sizeof( unsigned char );

	// Buffer for result image
	if( ( h_res = (unsigned char *)malloc( size ) ) == NULL )
		erro( "Erro allocating result imagem buffer." );

	// prepare for multithread
	DataStruct  data[2];
	data[0].deviceID = 0;
	data[0].init = 0;
	data[0].qtdPixels = qtdPixels;
	data[0].image = h_image;
	data[0].out = h_res;

	//CUTThread   thread = start_thread( routine, &(data[0]) );
	//routine( &(data[1]) );
	//end_thread( thread );

	routine( &(data[0]) );


	savePPM( (char *)"out.ppm", data.out, h_width, h_height );

	// free memory on the CPU side
	free( h_image );
	free( h_res );

	return 0;

}
