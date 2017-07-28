//-----------------------------------------
// Autor: Farias
// Data : January 2012
// Goal : Image treatment
//-----------------------------------------

/***************************************************************************************************
	Includes
***************************************************************************************************/

#include <cuda.h>
#include <cutil.h>
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

    	int i = blockDim.y * blockIdx.y + threadIdx.y;
	int j = blockDim.x * blockIdx.x + threadIdx.x;
	
	int idx = 3*ELEM(i,j,width);

	if( i < width && j < height){
		dest[idx]   = src[idx];
		dest[idx+1] = src[idx+1];
		//dest[idx+2] = src[idx+2];
		dest[idx+2]=(unsigned char) 0;
	}
}


/**************************************************************************************************/
__host__ int main( int argc, char *argv[] ) {

	double start_time, gpu_time;
	int    h_width, h_height;
	unsigned char*  h_image = NULL, *h_res = NULL;
	
	if( argc != 2 ) {
		
		erro( "Sintaxe: template image" );
		
	}
	cout << "Program for image treatment " << endl;

	readPPM( argv[1], &h_image, &h_width, &h_height );

	int size = 3*h_width*h_height*sizeof( unsigned char );

	// Buffer for result image
	if( ( h_res = (unsigned char *)malloc( size ) ) == NULL )
		erro( "Erro allocating result imagem buffer." );

	// Allocate memory for buffers in the GPU
	unsigned char *d_image;

	unsigned char *d_res;

	// Calcula dimensoes da grid e dos blocos

	start_time = get_clock_msec();
	funcGPU<<< gridSize, blockSize >>>( h_width, h_height, d_image, d_res );
	cudaThreadSynchronize();
	gpu_time = get_clock_msec() - start_time;

	// Copy result buffer back to cpu memory

	// Salva imagem resultado
	savePPM( "template.ppm", h_res, h_width, h_height );
	
	// Imprime tempo
	cout << "\tTempo de execucao da GPU: " << gpu_time << endl;
	cout << "-------------------------------------------" << endl;

	// Free buffers
	free( h_image );
	free( h_res );

	//system( "eog template.ppm" );

	return 0;

}
