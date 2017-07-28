//-----------------------------------------
// Autor: Farias
// Data : January 2012
// Goal : Image treatment
//-----------------------------------------

/***************************************************************************************************
	Includes
***************************************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <iostream>
#include <iomanip>

#include "rf-time.h"


/***************************************************************************************************
	Defines
***************************************************************************************************/

#define ELEM(i,j,DIMX_) (i+(j)*(DIMX_))


/***************************************************************************************************
	Functions
***************************************************************************************************/

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
__global__ void funcGPU( int width, int height, unsigned char *src, unsigned char *dest ) {

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;

	if( i < width && j < height ) {

		int idx = 3*ELEM( i, j, width );
		int r = src[ idx+2 ];
		int g = src[ idx+1 ];
		int b = src[ idx   ];
		dest[ idx   ] = (unsigned char)b;
		dest[ idx+1 ] = (unsigned char)g;
		dest[ idx+2 ] = (unsigned char)r;

	}

}

/**************************************************************************************************/
__global__ void mergeImages( int width, int height, unsigned char *src1, unsigned char *src2, unsigned char *dest ) {

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;

	if( i < width && j < height ) {

		int idx = 3*ELEM( i, j, width );
		int r1 = src1[ idx+2 ]; int r2 = src2[ idx+2 ];
		int g1 = src1[ idx+1 ]; int g2 = src2[ idx+1 ];
		int b1 = src1[ idx   ]; int b2 = src2[ idx   ];
	
		dest[ idx   ] = (unsigned char)(0.5*b1 + 0.5*b2);
		dest[ idx+1 ] = (unsigned char)(0.5*g1 + 0.5*g2);
		dest[ idx+2 ] = (unsigned char)(0.5*r1 + 0.5*r2);

	}

}


/**************************************************************************************************/
__host__ int main( int argc, char *argv[] ) {

	double start_time, gpu_time;
	int    h_width, h_height;
	unsigned char*  h_image_1 = NULL, *h_image_2 = NULL, *h_res = NULL;
	
	if( argc != 3 ) {
		
		erro( "Sintaxe: template image" );
		
	}


        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        // This function call returns 0 if there are no CUDA capable devices.
        if( deviceCount == 0 ) {
                printf("There is no device supporting CUDA\n");
                exit( 1 );
        }

        if(deviceCount < 2){
                printf("Nao tem placa grafica disponivel\n");
        }

        printf("Device Count %d\n", deviceCount);

        cudaSetDevice(1);


	cout << "Program for image treatment " << endl;

	readPPM( argv[1], &h_image_1, &h_width, &h_height );
	readPPM( argv[2], &h_image_2, &h_width, &h_height );

	int size = 3*h_width*h_height*sizeof( unsigned char );

	// Buffer for result image
	if( ( h_res = (unsigned char *)malloc( size ) ) == NULL )
		erro( "Erro allocating result imagem buffer." );

	// Allocate memory for buffers in the GPU
	unsigned char *d_image_1;
	cudaMalloc( (void**)&d_image_1, size );
	cudaMemcpy( d_image_1, h_image_1, size, cudaMemcpyHostToDevice );
	unsigned char *d_res;
	cudaMalloc( (void**)&d_res, size );

	// Allocate memory for buffers in the GPU
	unsigned char *d_image_2;
	cudaMalloc( (void**)&d_image_2, size );
	cudaMemcpy( d_image_2, h_image_2, size, cudaMemcpyHostToDevice );


	// Calcula dimensoes da grid e dos blocos
	dim3 blockSize( 16, 16 );

	int numBlocosX = h_width  / blockSize.x + ( h_width  % blockSize.x == 0 ? 0 : 1 );
	int numBlocosY = h_height / blockSize.y + ( h_height % blockSize.y == 0 ? 0 : 1 );
	dim3 gridSize( numBlocosX, numBlocosY, 1 );

	cout << "Blocks (" << blockSize.x << "," << blockSize.y << ")\n";
	cout << "Grid   (" << gridSize.x << "," << gridSize.y << ")\n";

	start_time = get_clock_msec();
	mergeImages<<< gridSize, blockSize >>>( h_width, h_height, d_image_1, d_image_2, d_res );
	cudaThreadSynchronize();
	gpu_time = get_clock_msec() - start_time;

	// Copy result buffer back to cpu memory
	cudaMemcpy( h_res, d_res, size, cudaMemcpyDeviceToHost );

	// Salva imagem resultado
	savePPM( (char *)"template.ppm", h_res, h_width, h_height );
	
	// Imprime tempo
	cout << "\tTempo de execucao da GPU: " << gpu_time << endl;
	cout << "-------------------------------------------" << endl;

	// Free buffers
	cudaFree( d_image_1 );
	cudaFree( d_image_2 );
	cudaFree( d_res   );
	free( h_image_1 );
	free( h_image_2 );
	free( h_res );

	//system( "eog template.ppm" );

	return 0;

}
