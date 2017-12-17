#-----------------------------------------------------------------------------
#
# Makefile CPU / CUDA
#
#-----------------------------------------------------------------------------

all: 	time_zero2 multidevice portable portableTimed deleteObjs

time_zero2:	Makefile-time_zero2
	make -f Makefile-time_zero2

multidevice:	Makefile-multidevice
	make -f Makefile-multidevice

portable:	Makefile-portable
	make -f Makefile-portable

portableTimed:	Makefile-portableTimed
	make -f Makefile-portableTimed

deleteObjs:
	rm *.o

clean:
	make -f Makefile-time_zero2    clean
	make -f Makefile-multidevice   clean
	make -f Makefile-portable      clean
	make -f Makefile-portableTimed clean
