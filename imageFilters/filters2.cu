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
__global__ void filter1( int width, int height, unsigned char *src, unsigned char *dest ) {

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;

	int aux, idx;

	if( i < width && j < height ) {
		for (int k = 0; k < 3; ++k)
		{
			aux = 0;
			idx = 3*ELEM( i, j, width );

			aux += 4*src[ idx+k ];

			if(i > 0) {
				idx = 3*ELEM( i-1, j, width );
				aux+= src[ idx+k ];
			}

			if(j > 0){
				idx = 3*ELEM( i, j-1, width );
				aux+= src[ idx+k ];	
			}

			if(i < width - 1){
				idx = 3*ELEM( i+1, j, width );
				aux+= src[ idx+k ];			
			}

			if(j < height - 1){
				idx = 3*ELEM( i, j+1, width );
				aux+= src[ idx+k ];		
			}			

			aux /= 8;

			dest[ idx+k ] = (unsigned char)aux;		
		}
		



	}

}

/**************************************************************************************************/
__global__ void filter_x( int width, int height, float p, unsigned char *src, unsigned char *dest ) {

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;

	int aux, idx;

	if( i < width && j < height ) {
		for (int k = 0; k < 3; ++k)
		{
			aux = 0;


			if(i > 0) {
				idx = 3*ELEM( i-1, j, width );
				aux-= p*src[ idx+k ];
			}


			if(i < width - 1){
				idx = 3*ELEM( i+1, j, width );
				aux+= p*src[ idx+k ];			
			}



			dest[ idx+k ] = (unsigned char)abs(aux);		
		}
		

	}

}



__global__ void filter_y( int width, int height, float p, unsigned char *src, unsigned char *dest ) {

        int i = threadIdx.x + blockIdx.x*blockDim.x;
        int j = threadIdx.y + blockIdx.y*blockDim.y;

        int aux, idx;

        if( i < width && j < height ) {
                for (int k = 0; k < 3; ++k)
                {
                        aux = 0;


                        if(j > 0) {
                                idx = 3*ELEM( i, j-1, width );
                                aux-= p*src[ idx+k ];
                        }


                        if(j < height - 1){
                                idx = 3*ELEM( i, j+1, width );
                                aux+= p*src[ idx+k ];
                        }


                        dest[ idx+k ] = (unsigned char)abs(aux);
                }


        }

}




/**************************************************************************************************/
__global__ void filter_x_y( int width, int height, float p, unsigned char *src, unsigned char *dest ) {

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;

	int aux_x, aux_y, idx;

	if( i < width && j < height ) {
		for (int k = 0; k < 3; ++k)
		{
			aux_x = 0;
			aux_y = 0;


			if(j > 0) {
				idx = 3*ELEM( i, j-1, width );
				aux_y-= p*src[ idx+k ];
			}


			if(j < height - 1){
				idx = 3*ELEM( i, j+1, width );
				aux_y+= p*src[ idx+k ];			
			}


                        if(i > 0) {
                                idx = 3*ELEM( i-1, j, width );
                                aux_x-= p*src[ idx+k ];
                        }


                        if(i < width - 1){
                                idx = 3*ELEM( i+1, j, width );
                                aux_x+= p*src[ idx+k ];
                        }


			dest[ idx+k ] = (unsigned char)sqrt((float)aux_x*aux_x+aux_y*aux_y);		
		}
		

	}

}



__global__ void tom_cinza( int width, int height,  unsigned char *src, unsigned char *dest ) {
	
	int i = threadIdx.x + blockIdx.x*blockDim.x;
        int j = threadIdx.y + blockIdx.y*blockDim.y;

        int cinza, idx;
	int r,g,b;
	
	if(i < width && j < height){
		idx = 3*ELEM( i, j, width );
		r = src[idx+2];
		g = src[idx+1];
		b = src[idx];
		cinza  = (30*r + 59*g + 11*b)/100;
		for (int k = 0; k < 3; ++k)
                {
                       dest[idx+k] = (unsigned char) cinza;
                }

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
	dim3 blockSize( 16, 16 );

	int numBlocosX = h_width  / blockSize.x + ( h_width  % blockSize.x == 0 ? 0 : 1 );
	int numBlocosY = h_height / blockSize.y + ( h_height % blockSize.y == 0 ? 0 : 1 );
	dim3 gridSize( numBlocosX, numBlocosY, 1 );

	cout << "Blocks (" << blockSize.x << "," << blockSize.y << ")\n";
	cout << "Grid   (" << gridSize.x << "," << gridSize.y << ")\n";

	start_time = get_clock_msec();
	tom_cinza<<< gridSize, blockSize >>>( h_width, h_height, d_image, d_res_cinza );
	cudaThreadSynchronize();
	gpu_time = get_clock_msec() - start_time;

	// Copy result buffer back to cpu memory
	cudaMemcpy( h_res, d_res, size, cudaMemcpyDeviceToHost );
	//cudaMemcpy( h_res, d_image, size, cudaMemcpyDeviceToHost );

	// Salva imagem resultado
	savePPM( (char *)"cinza.ppm", h_res, h_width, h_height );

	cudaMemcpy( d_image, d_res_cinza, size, cudaMemcpyDeviceToDevice );
	
	//Filtro em x
	filter_x<<< gridSize, blockSize >>>( h_width, h_height, 1.0, d_image, d_res );
	cudaMemcpy( h_res, d_res, size, cudaMemcpyDeviceToHost );
	savePPM( (char *)"filtro_x.ppm", h_res, h_width, h_height );	

	
	//Filtro em y
        filter_y<<< gridSize, blockSize >>>( h_width, h_height, 1.0, d_image, d_res );
        cudaMemcpy( h_res, d_res, size, cudaMemcpyDeviceToHost );
        savePPM( (char *)"filtro_y.ppm", h_res, h_width, h_height );

	
	//Filtro em x e y
        filter_x_y<<< gridSize, blockSize >>>( h_width, h_height, 1.0,  d_image, d_res );
        cudaMemcpy( h_res, d_res, size, cudaMemcpyDeviceToHost );
        savePPM( (char *)"filtro_x_y.ppm", h_res, h_width, h_height );



	// Imprime tempo
	cout << "\tTempo de execucao da GPU: " << gpu_time << endl;
	cout << "-------------------------------------------" << endl;

	// Free buffers
	cudaFree( d_image );
	cudaFree( d_res   );
	free( h_image );
	free( h_res );

	//system( "eog template.ppm" );

	return 0;

}