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
#define BLOCK_SIZE 16


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
/*
 Filtro Blur com os seguintes pesos
    2
  2 4 2 /12
    2
*/
__global__ void filter1( int width, int height, unsigned char *src, unsigned char *dest ) {

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;

	int aux, idx;

	if(i > 0 && j > 0 && i < width - 1 && j < height - 1) {
		for (int k = 0; k < 3; ++k)
		{
			aux = 0;
			idx = 3*ELEM( i, j, width );

			aux += 4*src[ idx+k ];

			idx = 3*ELEM( i-1, j, width );
			aux+= 2*src[ idx+k ];

			idx = 3*ELEM( i, j-1, width );
			aux+= 2*src[ idx+k ];

			idx = 3*ELEM( i+1, j, width );
			aux+= 2*src[ idx+k ];

			idx = 3*ELEM( i, j+1, width );
			aux+= 2*src[ idx+k ];

			aux /= 12;

			idx = 3*ELEM( i, j, width );
			dest[ idx+k ] = (unsigned char)aux;

		}

	}
}

/**************************************************************************************************/
__global__ void filter2( int width, int height, unsigned char *src, unsigned char *dest ) {

		int i = threadIdx.x + blockIdx.x*blockDim.x;
		int j = threadIdx.y + blockIdx.y*blockDim.y;

		int aux, idx;

		__shared__ int pesos[3][3];

		// Setando Pesos
		pesos[0][0] = 0; pesos[0][1] = 2; pesos[0][2] = 0;
		pesos[1][0] = 2; pesos[1][1] = 4; pesos[1][2] = 2;
		pesos[2][0] = 0; pesos[2][1] = 2; pesos[2][2] = 0;



		if(i > 0 && j > 0 && i < width - 1 && j < height - 1) {
			for (int k = 0; k < 3; ++k)
			{

				aux = 0;
				for (int lin = 0; lin < 3; lin++)
				{
					for (int col = 0; col < 3; col++){
						idx = 3*ELEM( i + lin - 1, j + col - 1, width );
						aux += pesos[lin][col]*src[ idx+k ];
					}
				}
				aux /= 12;
				idx = 3*ELEM( i, j , width );
				dest[ idx+k ] = (unsigned char)aux;

			}

		}
	}

/**************************************************************************************************/
__global__ void filter3( int width, int height, unsigned char *src, unsigned char *dest ) {
	
			int i = threadIdx.x + blockIdx.x*blockDim.x;
			int j = threadIdx.y + blockIdx.y*blockDim.y;
	
			int r, g, b, idx;
	
			__shared__ int pesos[3][3];
	
			// Setando Pesos
			pesos[0][0] = 0; pesos[0][1] = 2; pesos[0][2] = 0;
			pesos[1][0] = 2; pesos[1][1] = 4; pesos[1][2] = 2;
			pesos[2][0] = 0; pesos[2][1] = 2; pesos[2][2] = 0;
	
	
	
			if(i > 0 && j > 0 && i < width - 1 && j < height - 1) {
				
					r = g = b = 0;
					for (int lin = 0; lin < 3; lin++)
					{
						for (int col = 0; col < 3; col++){
							idx = 3*ELEM( i + lin - 1, j + col - 1, width );
							b += pesos[lin][col]*src[ idx+0 ];
							g += pesos[lin][col]*src[ idx+1 ];
							r += pesos[lin][col]*src[ idx+2 ];
						}
					}
					aux /= 12;
					idx = 3*ELEM( i, j , width );
					dest[ idx+0 ] = (unsigned char)b;
					dest[ idx+1 ] = (unsigned char)g;
					dest[ idx+2 ] = (unsigned char)r;
	
				
	
			}
		}

/**************************************************************************************************/
__host__ int main( int argc, char *argv[] ) {

	float gpu_time;
	int    h_width, h_height;
	unsigned char*  h_image = NULL, *h_res = NULL;

	cudaEvent_t startCuda, stopCuda;


	if( argc != 2 ) {

		erro( "Sintaxe: template image" );

	}


	int deviceCount;
    cudaGetDeviceCount(&deviceCount);
	printf("Device Count %d\n", deviceCount);
    // This function call returns 0 if there are no CUDA capable devices.
    if( deviceCount == 0 ) {
        printf("There is no device supporting CUDA\n");
        exit( 1 );
    }

    if(deviceCount < 2){
    	printf("Nao tem placa grafica disponivel\n");
    }

    cudaSetDevice(deviceCount - 1);


	cout << "Program for image treatment " << endl;

	readPPM( argv[1], &h_image, &h_width, &h_height );

	int size = 3*h_width*h_height*sizeof( unsigned char );

	// Buffer for result image
	if( ( h_res = (unsigned char *)malloc( size ) ) == NULL )
		erro( "Erro allocating result imagem buffer." );

	// Allocate memory for buffers in the GPU
	unsigned char *d_image;
	cudaMalloc( (void**)&d_image, size );
	cudaMemcpy( d_image, h_image, size, cudaMemcpyHostToDevice );
	unsigned char *d_res, *d_res_cinza;
	cudaMalloc( (void**)&d_res, size );
	cudaMalloc( (void**)&d_res_cinza, size );

	// Calcula dimensoes da grid e dos blocos
	dim3 blockSize( BLOCK_SIZE, BLOCK_SIZE );

	int numBlocosX = h_width  / blockSize.x + ( h_width  % blockSize.x == 0 ? 0 : 1 );
	int numBlocosY = h_height / blockSize.y + ( h_height % blockSize.y == 0 ? 0 : 1 );
	dim3 gridSize( numBlocosX, numBlocosY, 1 );

	cout << "Blocks (" << blockSize.x << "," << blockSize.y << ")\n";
	cout << "Grid   (" << gridSize.x << "," << gridSize.y << ")\n";

	cudaEventCreate(&startCuda); cudaEventCreate(&stopCuda);

	cudaThreadSynchronize();
	cudaEventRecord(startCuda, 0);
    filter1<<< gridSize, blockSize >>>( h_width, h_height, d_image, d_res );
	cudaThreadSynchronize();
    cudaEventRecord(stopCuda);
    cudaEventSynchronize(stopCuda);
	cudaEventElapsedTime(&gpu_time, startCuda, stopCuda);

	printf("\tTempo filtro normal: %f ms\n", gpu_time);
    cudaMemcpy( h_res, d_res, size, cudaMemcpyDeviceToHost );
    savePPM( (char *)"filtro_blur.ppm", h_res, h_width, h_height );


	cudaThreadSynchronize();
	cudaEventRecord(startCuda, 0);
    filter2<<< gridSize, blockSize >>>( h_width, h_height, d_image, d_res );
	cudaThreadSynchronize();
    cudaEventRecord(stopCuda);
    cudaEventSynchronize(stopCuda);
	cudaEventElapsedTime(&gpu_time, startCuda, stopCuda);

	printf("\tTempo filtro shared: %f ms\n", gpu_time);
    cudaMemcpy( h_res, d_res, size, cudaMemcpyDeviceToHost );
    savePPM( (char *)"filtro_blur_2.ppm", h_res, h_width, h_height );

	cudaThreadSynchronize();
	cudaEventRecord(startCuda, 0);
    filter3<<< gridSize, blockSize >>>( h_width, h_height, d_image, d_res );
	cudaThreadSynchronize();
    cudaEventRecord(stopCuda);
    cudaEventSynchronize(stopCuda);
	cudaEventElapsedTime(&gpu_time, startCuda, stopCuda);

	printf("\tTempo filtro shared sem loop canais: %f ms\n", gpu_time);
    cudaMemcpy( h_res, d_res, size, cudaMemcpyDeviceToHost );
    savePPM( (char *)"filtro_blur_3.ppm", h_res, h_width, h_height );

	// Free buffers
	cudaFree( d_image );
	cudaFree( d_res   );
	free( h_image );
	free( h_res );
	return 0;

}
