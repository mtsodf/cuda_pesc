#
# Makefile CPU / CUDA
#

#-----------------------------------------------------------------------------

all: 	fork pthread merge

fork:	Makefile-fork
	make -f Makefile-fork

pthread:	Makefile-pthread
	make -f Makefile-pthread

pthread:	Makefile-merge-fork
	make -f Makefile-merge-fork

clean:
	make -f Makefile-fork      clean
	make -f Makefile-pthread   clean
