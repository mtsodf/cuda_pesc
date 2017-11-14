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
#include "rf-time.h"

/***************************************************************************************************
	Functions
***************************************************************************************************/

using namespace std;

__host__ void erro( const char tipoDeErro[] ) {

	fprintf( stderr, "%s\n", tipoDeErro );
	exit(0);

}


/***************************************************************************************************
	Defines
***************************************************************************************************/

#define ELEM(i,j,DIMX_) (i+(j)*(DIMX_))


__global__ void somaUm( int *a ) {

	atomicAdd( a, 1 );

}


/**************************************************************************************************/
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



int main( int argc, char *argv[] ) {
	
	int pid ;
  int blSizeX = 16;
	double start_time, gpu_time;
	int    h_width, h_height;	
 	int h_a = 0;
	int deviceCount = 0;
	unsigned char*  h_image_1 = NULL, *h_image_2 = NULL, *h_res = NULL;
	printf("Creating the son process\n");
	
 
 if(argc < 3) {
   printf("Necessario duas imagens como parametro para fazer o merge.\n");
   return;
 }
 
	cout << "Program for image treatment " << endl;
  
  printf("Fazendo merge das imagens: \n");
  printf("\t%s\n", argv[1]);
  printf("\t%s\n", argv[2]);
  
  
  
  printf("Lendo imagens\n");
	readPPM( argv[1], &h_image_1, &h_width, &h_height );
	readPPM( argv[2], &h_image_2, &h_width, &h_height );


 
  cudaSetDevice(1);
	int pixelsTotal = h_width*h_height*sizeof( unsigned char );

  
  
  printf("Alocando imagens na gpu e realizando copias\n");
	unsigned char *d_image_1;
	cudaMalloc( (void**)&d_image_1, 3*pixelsTotal );
	unsigned char *d_image_2;
	cudaMalloc( (void**)&d_image_2, 3*pixelsTotal );
	cudaMemcpy( d_image_2, h_image_2, 3*pixelsTotal, cudaMemcpyHostToDevice );

	cudaMemcpy( d_image_1, h_image_1, 3*pixelsTotal, cudaMemcpyHostToDevice );
	unsigned char *d_res;
	cudaMalloc( (void**)&d_res, 3*pixelsTotal );

	// Buffer for result image
	if( ( h_res = (unsigned char *)malloc( 3*pixelsTotal ) ) == NULL )
		erro( "Erro allocating result imagem buffer." );

  printf("Realizando o fork.\n");
	pid = fork();



	if( pid == -1 ) { /* erro */
		
		perror("impossivel de criar um filho") ;
		exit(-1); 
		
	} else if( pid == 0 ) { /* filho */
     cudaSetDevice(1);
     printf("Processo filho\n");
  
     int pixelsLocal = pixelsTotal/2 + pixelsTotal%2;
     int offset = 3*(pixelsTotal/2);
     
     dim3 blockSize( blSizeX);   
  	 int numBlocosX = pixelsLocal  / blockSize.x + ( pixelsLocal  % blockSize.x == 0 ? 0 : 1 );
  	 dim3 gridSize( numBlocosX, 1, 1 );
     
     printf("Realizando o merge no filho.\n");
     mergeGPU1d<<< gridSize, blockSize >>>(d_image_1+offset, d_image_2+offset, d_res+offset, pixelsLocal);
     printf("Fim do merge no filho.\n");

	} else { /* pai */
     printf("Processo pai\n");
     
     int pixelsLocal = pixelsTotal/2;
     int offset = 0;      

     
     printf("Offset do pai %d\n", offset);
     
     dim3 blockSize( blSizeX);  
  	 int numBlocosX = pixelsLocal  / blockSize.x + ( pixelsLocal  % blockSize.x == 0 ? 0 : 1 );
  	 dim3 gridSize( numBlocosX, 1, 1 );
    
     printf("Realizando o merge no pai.\n");
     mergeGPU1d<<< gridSize, blockSize >>>(d_image_1+offset, d_image_2+offset, d_res+offset, pixelsLocal);
     printf("Fim do merge no pai.\n");
     sleep(3);
  	 // Copy result buffer back to cpu memory
  	 cudaMemcpy( h_res, d_res, 3*pixelsTotal, cudaMemcpyDeviceToHost );
     
  	 // Salva imagem resultado
  	 savePPM( (char *)"template.ppm", h_res, h_width, h_height );     
	}

	exit(0);

}
