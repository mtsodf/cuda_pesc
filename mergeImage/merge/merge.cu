//-----------------------------------------
// Autor: Farias
// Data : May 2011
// Goal : Merge duas imagens PPM
//-----------------------------------------

/***************************************************************************************************
	Includes
***************************************************************************************************/

#include <stdio.h>
#include <cuda.h>
#include <iostream>
#include <iomanip>

#include "rf-time.h"


/***************************************************************************************************
	Defines
***************************************************************************************************/

#define ELEM(i,j,DIMX_) ((i)+(j)*(DIMX_))
//#define STREAM_SIZE 351232
//#define STREAM_SIZE 1024

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
__host__ void salvaPPM( char *fname, unsigned char *buffer, int width, int height ) {

	if( !buffer ) {
		cout << "Image not saved. This ViewPoint Class in NOT FULL." << endl;
		return;
	}
	FILE *f = fopen( fname, "wb" );
	if( f == NULL ) erro( "Erro escrevendo o PPM" );

	fprintf( f, "P6\n# Gravado no curso de CUDA\n%u %u\n%u\n", width, height, 255 );
	fwrite( buffer, 3, width*height, f );
	fclose(f);

}

__host__ void lerPPM( char *fname, unsigned char **buffer, int *width, int *height ) {

	char aux[256];
	FILE *f = fopen( fname, "r" );
	if( f == NULL ) 
		erro( "Erro lendo o PPM" );

	fgets( aux, 256, f );
	fgets( aux, 256, f );
	fgets( aux, 256, f );
	sscanf( aux, "%d %d", width, height );
	fgets( aux, 256, f );

	if( !(*buffer) ) {
		free( *buffer );
	}
	
	int size = 3*(*width)*(*height)*sizeof( char );
	cout << "Dimensao da imagem: (" << *width << "," << *height <<")\n";

	if( ( *buffer = (unsigned char *)malloc( size ) ) == NULL )
		erro( "Erro lendo o buffer da imagem" );

	fread( *buffer, 3, (*width)*(*height), f );
	fclose( f );

}

/**************************************************************************************************/
__global__ void mergeGPU( unsigned char *image1, unsigned char *image2, 
			  unsigned char *res, int width, int height ) {

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;

	if( i < width && j < height ) {

	 	int idx = 3*ELEM( i, j, width );
	 	int r1 = image1[ idx+2 ];
	 	int g1 = image1[ idx+1 ];
	 	int b1 = image1[ idx   ];
	 	int r2 = image2[ idx+2 ];
	 	int g2 = image2[ idx+1 ];
	 	int b2 = image2[ idx   ];
	 	int r = (int)( ( (float)r1 + (float)r2 )*0.5f );
		int g = (int)( ( (float)g1 + (float)g2 )*0.5f );
	 	int b = (int)( ( (float)b1 + (float)b2 )*0.5f );
	 	res[ idx+2 ] = (unsigned char)r;
	 	res[ idx+1 ] = (unsigned char)g;
	 	res[ idx   ] = (unsigned char)b;
		
	 }
	
}

__global__ void mergeGPU1d( unsigned char *image1, unsigned char *image2,
                          unsigned char *res, int pixels ) {

        int i = threadIdx.x + blockIdx.x*blockDim.x;

        if( i < pixels ) {

                int idx = 3*i;
                int r1 = image1[ idx+2 ];
                int g1 = image1[ idx+1 ];
                int b1 = image1[ idx   ];
                int r2 = image2[ idx+2 ];
                int g2 = image2[ idx+1 ];
                int b2 = image2[ idx   ];
                int r = (int)( ( (float)r1 + (float)r2 )*0.5f );
                int g = (int)( ( (float)g1 + (float)g2 )*0.5f );
                int b = (int)( ( (float)b1 + (float)b2 )*0.5f );
                res[ idx+2 ] = (unsigned char)r;
                res[ idx+1 ] = (unsigned char)g;
                res[ idx   ] = (unsigned char)b;

         }

}


/**************************************************************************************************/
__host__ int main( int argc, char *argv[] ) {

	int blSizeX = 16, blSizeY = 16;
	double start_time, gpu_time;
	int    h_width1, h_height1;
	int    h_width2, h_height2;
	unsigned char   *h_imagem1 = NULL, *h_imagem2 = NULL;
	unsigned char   *h_imagem_resultado = NULL;
	
	if( argc < 3 ) {
		
		erro( "Syntaxe: merge fig1 fig2 [numBlocoX numBlocoY]" );
		
	}


  cudaSetDevice(1);


	cout << "Programa para Merge duas Imagens PPM " << endl;

	switch( argc ) {

	case 4:
		blSizeX = blSizeY = atoi( argv[ 3 ] );
		break;
	case 5:
		blSizeX = atoi( argv[ 3 ] );
		blSizeY = atoi( argv[ 4 ] );
	}
 
  int STREAM_SIZE = 351232;

	lerPPM( argv[1], &h_imagem1, &h_width1, &h_height1 );
	lerPPM( argv[2], &h_imagem2, &h_width2, &h_height2 );

	if( h_width1 != h_width2 || h_height1 != h_height2 )
		erro( "Imagens tem dimensoes diferentes.\nAbortando." );

	int size = 3*h_width1*h_height1*sizeof( char );

	if( ( h_imagem_resultado = (unsigned char *)malloc( size ) ) == NULL )
		erro( "Erro alocando imagem resultado." );

	// Aloca memória no device e copia vetorA e vetorB para lá
	unsigned char *d_imagem1 = NULL;
	cudaMalloc( (void**)&d_imagem1, size );
	cudaMemcpy( d_imagem1, h_imagem1, size, cudaMemcpyHostToDevice );

	unsigned char *d_imagem2 = NULL;
	cudaMalloc( (void**)&d_imagem2, size );
	cudaMemcpy( d_imagem2, h_imagem2, size, cudaMemcpyHostToDevice );

	unsigned char *d_res = NULL;
	cudaMalloc( (void**)&d_res, size );

	// Calcula dimensoes da grid e dos blocos
	dim3 blockSize( blSizeX);
	int pixels = h_width1*h_height1;
  printf("Quantidade de Pixels %d\n", pixels);
	int numBlocosX = pixels  / blockSize.x + ( pixels  % blockSize.x == 0 ? 0 : 1 );
	dim3 gridSize( numBlocosX, 1, 1 );

	cout << "Blocks (" << blockSize.x << ")\n";
	cout << "Grid   (" << gridSize.x << "," << gridSize.y << ")\n";

	// Chama SomarVetoresGPU
	start_time = get_clock_msec();
	mergeGPU1d<<< gridSize, blockSize >>>( d_imagem1, d_imagem2, d_res, pixels );
	cudaThreadSynchronize();
	gpu_time = get_clock_msec() - start_time;

	// Copia o resultado de volta para o host
	cudaMemcpy( h_imagem_resultado, d_res, size, cudaMemcpyDeviceToHost );

	// Salva imagem resultado
	salvaPPM( "merge.ppm", h_imagem_resultado, h_width1, h_height1 );
	
	// Imprime tempo
	cout << "\tTempo de execucao da GPU: " << gpu_time << "ms" << endl;
	cout << "-------------------------------------------" << endl;

	//system( "eog merge.ppm" );	

	// Libera memória do device
	cudaFree( d_imagem1 );
	cudaFree( d_imagem2 );
	cudaFree( d_res     );
	
	free( h_imagem_resultado );

	//Merge das imagens com stream
	cudaStream_t    stream0, stream1;

	// initialize the streams
	cudaStreamCreate( &stream0 );
	//cudaStreamCreate( &stream1 );	

	
	unsigned char   *h_imagem1_pin = NULL, *h_imagem2_pin = NULL;
	
	// Alocando o valor de 
	cudaHostAlloc( (void**)&h_imagem1_pin, size, cudaHostAllocDefault );
	cudaHostAlloc( (void**)&h_imagem2_pin, size, cudaHostAllocDefault );
	cudaHostAlloc( (void**)&h_imagem_resultado, size, cudaHostAllocDefault );

  unsigned char *d_res1 = NULL;
  unsigned char *d_res2 = NULL;
  
	cudaMalloc( (void**)&d_imagem1, STREAM_SIZE*3*sizeof( char ) );
	cudaMalloc( (void**)&d_imagem2, STREAM_SIZE*3*sizeof( char ) );
	cudaMalloc( (void**)&d_res1, STREAM_SIZE*3*sizeof( char ));
  cudaMalloc( (void**)&d_res2, STREAM_SIZE*3*sizeof( char ));

  printf("Esta aqui %d. Size char %d\n", __LINE__, sizeof(char));

	for(int i = 0; i < size; i++){

		h_imagem1_pin[i] = h_imagem1[i];
		h_imagem2_pin[i] = h_imagem2[i];

	}

	printf("Esta aqui %d\n", __LINE__);

/*	
  cudaMemcpyAsync( d_imagem1, h_imagem1_pin, 3*STREAM_SIZE * sizeof(char), cudaMemcpyHostToDevice, stream0 );
  cudaMemcpyAsync( d_imagem2, h_imagem2_pin, 3*STREAM_SIZE * sizeof(char), cudaMemcpyHostToDevice, stream0 );
  mergeGPU1d<<< STREAM_SIZE/blSizeX, blSizeX, 0, stream0 >>>( d_imagem1, d_imagem2, d_res1, STREAM_SIZE );
  cudaMemcpyAsync( h_imagem_resultado, d_res1, 3*STREAM_SIZE * sizeof(char), cudaMemcpyDeviceToHost, stream0 );


  cudaMemcpyAsync( d_imagem1, h_imagem1_pin+3*STREAM_SIZE, 3*STREAM_SIZE * sizeof(char), cudaMemcpyHostToDevice, stream0 );
  cudaMemcpyAsync( d_imagem2, h_imagem2_pin+3*STREAM_SIZE, 3*STREAM_SIZE * sizeof(char), cudaMemcpyHostToDevice, stream0 );
  mergeGPU1d<<< STREAM_SIZE/blSizeX, blSizeX, 0, stream0 >>>( d_imagem1, d_imagem2, d_res2, STREAM_SIZE );
  cudaMemcpyAsync( h_imagem_resultado+3*STREAM_SIZE, d_res2, 3*STREAM_SIZE * sizeof(char), cudaMemcpyDeviceToHost, stream0 );
*/
 
	cudaEvent_t     start, stop;
	float           elapsedTime;
     
  // start the timers
	cudaEventCreate( &start );
	cudaEventCreate( &stop );
  
	cudaEventRecord( start, 0 );
 
  for(int i = 0; i < pixels; i += STREAM_SIZE){
  		cudaMemcpyAsync( d_imagem1, h_imagem1_pin+3*i, 3*STREAM_SIZE * sizeof(char), cudaMemcpyHostToDevice, stream0 );		
  		cudaMemcpyAsync( d_imagem2, h_imagem2_pin+3*i, 3*STREAM_SIZE * sizeof(char), cudaMemcpyHostToDevice, stream0 );
  		mergeGPU1d<<< STREAM_SIZE/blSizeX, blSizeX, 0, stream0 >>>( d_imagem1, d_imagem2, d_res1, STREAM_SIZE );
      cudaMemcpyAsync( h_imagem_resultado+3*i, d_res1, 3*STREAM_SIZE * sizeof(char), cudaMemcpyDeviceToHost, stream0 );
  }


	
  printf("Esta aqui %d\n", __LINE__);
	cudaStreamSynchronize( stream0 );
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	cudaEventElapsedTime( &elapsedTime, start, stop );
	printf( "Time taken:  %3.1f ms\n", elapsedTime);
 
  salvaPPM( "merge_stream.ppm", h_imagem_resultado, h_width1, h_height1 );

  cudaFree( d_res1    );
  cudaFree( d_res2    );

	cudaFreeHost( h_imagem1_pin );
	cudaFreeHost( h_imagem2_pin );
  cudaFreeHost( h_imagem_resultado );

	// Libera memória do host
	free( h_imagem1 );
	free( h_imagem2 );
	return 0;

}
