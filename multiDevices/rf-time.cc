//-------------------------------------------------------------
// Medicao de tempo em linux e windows
//-------------------------------------------------------------
#ifndef WINDOWS

#include <sys/time.h>

double get_clock_sec( void ) {
	
	struct timeval t;
	struct timezone tz;
	gettimeofday(&t,&tz);
	return (double) t.tv_sec + (double) t.tv_usec * 1E-6;
	
}

double get_clock_msec( void ) {
	
	struct timeval t;
	struct timezone tz;
	gettimeofday(&t,&tz);
	return (double) t.tv_sec * 1E+3 + (double) t.tv_usec * 1E-3;
	
}

#else

#include <windows.h>

double get_clock_sec( LARGE_INTEGER fim, LARGE_INTEGER inicio ) {

	LARGE_INTEGER frequency;
	QueryPerformanceFrequency(&frequency);
	return (double)((fim.QuadPart - inicio.QuadPart) / frequency.QuadPart);

}

double get_clock_msec( LARGE_INTEGER fim, LARGE_INTEGER inicio ) {

	return (double)( get_clock_sec( fim, inicio ) * 1000.0 );

}

// Exemplo
// int main() {
// 	LARGE_INTEGER inicio, fim;
// 	QueryPerformanceCounter(&inicio);
// 	// PROCESSAMENTO
// 	QueryPerformanceCounter(&fim);
// 	double tempoDecorrido_ms = getTempo_ms(fim, inicio);
// 	return 0;
// }

#endif
