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

	if( i > 0 && i < width-1 && j < height ) {
		for (int k = 0; k < 3; ++k)
		{
			aux = 0;

			idx = 3*ELEM( i-1, j, width );
			aux-= p*src[ idx+k ];
	
			idx = 3*ELEM( i+1, j, width );
			aux+= p*src[ idx+k ];			

			dest[ idx+k ] = (unsigned char)abs(aux);		
		}
		

	}

}



__global__ void filter_y( int width, int height, float p, unsigned char *src, unsigned char *dest ) {

        int i = threadIdx.x + blockIdx.x*blockDim.x;
        int j = threadIdx.y + blockIdx.y*blockDim.y;

        int aux, idx;

        if( i < width && j < height-1 && j > 0) {
                for (int k = 0; k < 3; ++k)
                {
                        aux = 0;
                        idx = 3*ELEM( i, j-1, width );
                        aux-= p*src[ idx+k ];
                        
                        idx = 3*ELEM( i, j+1, width );
                        aux+= p*src[ idx+k ];


                        dest[ idx+k ] = (unsigned char)abs(aux);
                }


        }

}




/**************************************************************************************************/
__global__ void filter_x_y( int width, int height, float p, unsigned char *src, unsigned char *dest ) {

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;

	int aux_x, aux_y, idx;

	if( i > 0 && i < width-1 && j < height-1 && j > 0) {
		aux_x = 0;
		aux_y = 0;

		idx = 3*ELEM( i, j-1, width );
		aux_y-= p*src[ idx+0 ];
		idx = 3*ELEM( i, j+1, width );
		aux_y+= p*src[ idx+0 ];

        idx = 3*ELEM( i-1, j, width );		
		aux_x-= p*src[ idx+0 ];				
        idx = 3*ELEM( i+1, j, width );
        aux_x+= p*src[ idx+0 ];			
        
		idx = 3*ELEM( i, j, width );
		dest[ idx+0 ] = (unsigned char)sqrt((float)aux_x*aux_x+aux_y*aux_y);		

		aux_x = 0;
		aux_y = 0;

		idx = 3*ELEM( i, j-1, width );
		aux_y-= p*src[ idx+1 ];
		idx = 3*ELEM( i, j+1, width );
		aux_y+= p*src[ idx+1 ];

        idx = 3*ELEM( i-1, j, width );		
		aux_x-= p*src[ idx+1 ];				
        idx = 3*ELEM( i+1, j, width );
        aux_x+= p*src[ idx+1 ];			
        
		idx = 3*ELEM( i, j, width );
		dest[ idx+1 ] = (unsigned char)sqrt((float)aux_x*aux_x+aux_y*aux_y);	



		aux_x = 0;
		aux_y = 0;

		idx = 3*ELEM( i, j-1, width );
		aux_y-= p*src[ idx+2 ];
		idx = 3*ELEM( i, j+1, width );
		aux_y+= p*src[ idx+2 ];

        idx = 3*ELEM( i-1, j, width );		
		aux_x-= p*src[ idx+2 ];				
        idx = 3*ELEM( i+1, j, width );
        aux_x+= p*src[ idx+2 ];		

		idx = 3*ELEM( i, j, width );
		dest[ idx+2 ] = (unsigned char)sqrt((float)aux_x*aux_x+aux_y*aux_y);		
	}
		

}



__global__ void filter_x_y_sm( int width, int height, float p, unsigned char *src, unsigned char *dest ) {
	__shared__ float r[BLOCK_SIZE+2][BLOCK_SIZE+2], g[BLOCK_SIZE+2][BLOCK_SIZE+2], b[BLOCK_SIZE+2][BLOCK_SIZE+2];
	int idx;
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	float aux_x, aux_y;
	//Copia dos valores para o tile

	if(i < width && j < height){
		idx = 3*ELEM(i,j,width);
		r[threadIdx.x+1][threadIdx.y+1] = src[idx+2];
		g[threadIdx.x+1][threadIdx.y+1] = src[idx+1];
		b[threadIdx.x+1][threadIdx.y+1] = src[idx+0];
	

		if(blockIdx.y > 0 && threadIdx.y == 0){
			idx = 3*ELEM(i,j-1,width);
			r[threadIdx.x+1][threadIdx.y] = src[idx+2];
			g[threadIdx.x+1][threadIdx.y] = src[idx+1];
			b[threadIdx.x+1][threadIdx.y] = src[idx+0];
		}

		if(blockIdx.y < gridDim.y - 1 && threadIdx.y == blockDim.y-1){
			idx = 3*ELEM(i,j+1,width);
			r[threadIdx.x+1][threadIdx.y+2] = src[idx+2];
			g[threadIdx.x+1][threadIdx.y+2] = src[idx+1];
			b[threadIdx.x+1][threadIdx.y+2] = src[idx+0];
		}

		if(blockIdx.x > 0 && threadIdx.x == 0){
			idx = 3*ELEM(i-1,j,width);
			r[threadIdx.x][threadIdx.y+1] = src[idx+2];
			g[threadIdx.x][threadIdx.y+1] = src[idx+1];
			b[threadIdx.x][threadIdx.y+1] = src[idx+0];
		}

		if(blockIdx.x < gridDim.x - 1 && threadIdx.x == blockDim.x-1){
			idx = 3*ELEM(i+1,j,width);
			r[threadIdx.x+2][threadIdx.y+1] = src[idx+2];
			g[threadIdx.x+2][threadIdx.y+1] = src[idx+1];
			b[threadIdx.x+2][threadIdx.y+1] = src[idx+0];
		}
	}
	__syncthreads();

	if( i > 0 && i < width-1 && j < height-1 && j > 0){
		idx = 3*ELEM(i,j,width);
		
		aux_x = 0;
		aux_y = 0;

		aux_y-= p*b[threadIdx.x+1][threadIdx.y+0];
		aux_y+= p*b[threadIdx.x+1][threadIdx.y+2];

		aux_x-= p*b[threadIdx.x+0][threadIdx.y+1];			
		aux_x+= p*b[threadIdx.x+2][threadIdx.y+1];			
            
		dest[ idx+0 ] = (unsigned char)sqrt((float)aux_x*aux_x+aux_y*aux_y);		


		aux_x = 0;
		aux_y = 0;

		aux_y-= p*g[threadIdx.x+1][threadIdx.y+0];
		aux_y+= p*g[threadIdx.x+1][threadIdx.y+2];

		aux_x-= p*g[threadIdx.x+0][threadIdx.y+1];			
		aux_x+= p*g[threadIdx.x+2][threadIdx.y+1];			
            
		dest[ idx+1 ] = (unsigned char)sqrt((float)aux_x*aux_x+aux_y*aux_y);	
	
		aux_x = 0;
		aux_y = 0;

		aux_y-= p*r[threadIdx.x+1][threadIdx.y+0];
		aux_y+= p*r[threadIdx.x+1][threadIdx.y+2];

		aux_x-= p*r[threadIdx.x+0][threadIdx.y+1];			
		aux_x+= p*r[threadIdx.x+2][threadIdx.y+1];			
            
		dest[ idx+2 ] = (unsigned char)sqrt((float)aux_x*aux_x+aux_y*aux_y);	

	}


}


__global__ void filter_x_y_sm( int width, int height, float p, unsigned char *src, unsigned char *dest , int threshold) {
	__shared__ float r[BLOCK_SIZE+2][BLOCK_SIZE+2], g[BLOCK_SIZE+2][BLOCK_SIZE+2], b[BLOCK_SIZE+2][BLOCK_SIZE+2];
	int idx;
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	float aux_x, aux_y;
	//Copia dos valores para o tile

	if(i < width && j < height){
		idx = 3*ELEM(i,j,width);
		r[threadIdx.x+1][threadIdx.y+1] = src[idx+2];
		g[threadIdx.x+1][threadIdx.y+1] = src[idx+1];
		b[threadIdx.x+1][threadIdx.y+1] = src[idx+0];
	

		if(blockIdx.y > 0 && threadIdx.y == 0){
			idx = 3*ELEM(i,j-1,width);
			r[threadIdx.x+1][threadIdx.y] = src[idx+2];
			g[threadIdx.x+1][threadIdx.y] = src[idx+1];
			b[threadIdx.x+1][threadIdx.y] = src[idx+0];
		}

		if(blockIdx.y < gridDim.y - 1 && threadIdx.y == blockDim.y-1){
			idx = 3*ELEM(i,j+1,width);
			r[threadIdx.x+1][threadIdx.y+2] = src[idx+2];
			g[threadIdx.x+1][threadIdx.y+2] = src[idx+1];
			b[threadIdx.x+1][threadIdx.y+2] = src[idx+0];
		}

		if(blockIdx.x > 0 && threadIdx.x == 0){
			idx = 3*ELEM(i-1,j,width);
			r[threadIdx.x][threadIdx.y+1] = src[idx+2];
			g[threadIdx.x][threadIdx.y+1] = src[idx+1];
			b[threadIdx.x][threadIdx.y+1] = src[idx+0];
		}

		if(blockIdx.x < gridDim.x - 1 && threadIdx.x == blockDim.x-1){
			idx = 3*ELEM(i+1,j,width);
			r[threadIdx.x+2][threadIdx.y+1] = src[idx+2];
			g[threadIdx.x+2][threadIdx.y+1] = src[idx+1];
			b[threadIdx.x+2][threadIdx.y+1] = src[idx+0];
		}
	}
	__syncthreads();

	if( i > 0 && i < width-1 && j < height-1 && j > 0){
		idx = 3*ELEM(i,j,width);
		
		aux_x = 0;
		aux_y = 0;

		aux_y-= p*b[threadIdx.x+1][threadIdx.y+0];
		aux_y+= p*b[threadIdx.x+1][threadIdx.y+2];

		aux_x-= p*b[threadIdx.x+0][threadIdx.y+1];			
		aux_x+= p*b[threadIdx.x+2][threadIdx.y+1];			
            
		dest[ idx+0 ] = ((unsigned char)sqrt((float)aux_x*aux_x+aux_y*aux_y) >= threshold)*255;	


		aux_x = 0;
		aux_y = 0;

		aux_y-= p*g[threadIdx.x+1][threadIdx.y+0];
		aux_y+= p*g[threadIdx.x+1][threadIdx.y+2];

		aux_x-= p*g[threadIdx.x+0][threadIdx.y+1];			
		aux_x+= p*g[threadIdx.x+2][threadIdx.y+1];			
            
		dest[ idx+1 ] = ((unsigned char)sqrt((float)aux_x*aux_x+aux_y*aux_y) >= threshold)*255;		
	
		aux_x = 0;
		aux_y = 0;

		aux_y-= p*r[threadIdx.x+1][threadIdx.y+0];
		aux_y+= p*r[threadIdx.x+1][threadIdx.y+2];

		aux_x-= p*r[threadIdx.x+0][threadIdx.y+1];			
		aux_x+= p*r[threadIdx.x+2][threadIdx.y+1];			
            
		dest[ idx+2 ] = ((unsigned char)sqrt((float)aux_x*aux_x+aux_y*aux_y) >= threshold)*255;	

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

	//Filtro em x

	start_time = get_clock_msec();
    filter_x<<< gridSize, blockSize >>>( h_width, h_height, 1.0, d_image, d_res );
	cudaThreadSynchronize();
	gpu_time = get_clock_msec() - start_time;
	printf("\tTempo filtro x: %f\n", gpu_time);
    cudaMemcpy( h_res, d_res, size, cudaMemcpyDeviceToHost );
    savePPM( (char *)"filtro_x.ppm", h_res, h_width, h_height );


    //Filtro em y
	start_time = get_clock_msec();
    filter_y<<< gridSize, blockSize >>>( h_width, h_height, 1.0, d_image, d_res );
	cudaThreadSynchronize();
	gpu_time = get_clock_msec() - start_time;
	printf("\tTempo filtro y: %f\n", gpu_time);	
    cudaMemcpy( h_res, d_res, size, cudaMemcpyDeviceToHost );
    savePPM( (char *)"filtro_y.ppm", h_res, h_width, h_height );


    //Filtro em x e y
	start_time = get_clock_msec();
    filter_x_y<<< gridSize, blockSize >>>( h_width, h_height, 1.0,  d_image, d_res );
	cudaThreadSynchronize();
	gpu_time = get_clock_msec() - start_time;   
	printf("\tTempo filtro x e y: %f\n", gpu_time); 
	cudaMemcpy( h_res, d_res, size, cudaMemcpyDeviceToHost );
    savePPM( (char *)"filtro_x_y.ppm", h_res, h_width, h_height );

	printf("---------------------------------------------------------\n");
	printf("Filtro para imagem tom de cinza\n");
	printf("---------------------------------------------------------\n");

	start_time = get_clock_msec();
	tom_cinza<<< gridSize, blockSize >>>( h_width, h_height, d_image, d_res_cinza );
	cudaThreadSynchronize();
	gpu_time = get_clock_msec() - start_time;

	printf("Tempo tom cinza: %f\n", gpu_time); 

	cudaMemcpy( h_res, d_res_cinza, size, cudaMemcpyDeviceToHost );
	savePPM( (char *)"cinza.ppm", h_res, h_width, h_height );

	//Copia imagem cinza para d_image
	cudaMemcpy( d_image, d_res_cinza, size, cudaMemcpyDeviceToDevice );


	//Filtro em x

	start_time = get_clock_msec();
    filter_x<<< gridSize, blockSize >>>( h_width, h_height, 1.0, d_image, d_res );
	cudaThreadSynchronize();
	gpu_time = get_clock_msec() - start_time;
	printf("\tTempo filtro x: %f\n", gpu_time);
    cudaMemcpy( h_res, d_res, size, cudaMemcpyDeviceToHost );
    savePPM( (char *)"filtro_cinza_x.ppm", h_res, h_width, h_height );


    //Filtro em y
	start_time = get_clock_msec();
    filter_y<<< gridSize, blockSize >>>( h_width, h_height, 1.0, d_image, d_res );
	cudaThreadSynchronize();
	gpu_time = get_clock_msec() - start_time;
	printf("\tTempo filtro y: %f\n", gpu_time);	
    cudaMemcpy( h_res, d_res, size, cudaMemcpyDeviceToHost );
    savePPM( (char *)"filtro_cinza_y.ppm", h_res, h_width, h_height );


    //Filtro em x e y
	start_time = get_clock_msec();
    filter_x_y<<< gridSize, blockSize >>>( h_width, h_height, 1.0,  d_image, d_res );
	cudaThreadSynchronize();
	gpu_time = get_clock_msec() - start_time;   
	printf("\tTempo filtro x e y: %f\n", gpu_time); 
	cudaMemcpy( h_res, d_res, size, cudaMemcpyDeviceToHost );
    savePPM( (char *)"filtro_cinza_x_y.ppm", h_res, h_width, h_height );


    //Filtro em x e y
	start_time = get_clock_msec();
    filter_x_y_sm<<< gridSize, blockSize >>>( h_width, h_height, 1.0,  d_image, d_res );
	cudaThreadSynchronize();
	gpu_time = get_clock_msec() - start_time;   
	printf("\tTempo filtro x e y sm: %f\n", gpu_time); 
	cudaMemcpy( h_res, d_res, size, cudaMemcpyDeviceToHost );
    savePPM( (char *)"filtro_cinza_x_y_sm.ppm", h_res, h_width, h_height );	


	char name[100];
	for(int threshold = 1; threshold < 255; threshold+= 20){
		filter_x_y_sm<<< gridSize, blockSize >>>( h_width, h_height, 1.0,  d_image, d_res , threshold);
		cudaMemcpy( h_res, d_res, size, cudaMemcpyDeviceToHost );
		sprintf(name, "filtro_cinza_x_y_sm_%d.ppm", threshold);
    	savePPM( name, h_res, h_width, h_height );			
	}

	// Free buffers
	cudaFree( d_image );
	cudaFree( d_res   );
	free( h_image );
	free( h_res );

	//system( "eog template.ppm" );

	return 0;

}
