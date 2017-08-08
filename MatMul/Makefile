all: testeBlockSizeTrans testeBlockSize matrixMul
	 @echo "BUILD TYPE"
	 @echo $(build_type)

testeBlockSizeTrans: 
	mkdir -p bin
	nvcc -o testeBlockSizeTrans testeBlockSizeTrans.cu rf-time.cc
	mv testeBlockSizeTrans ./bin

testeBlockSize: 
	mkdir -p bin
	nvcc -o testeBlockSize testeBlockSize.cu rf-time.cc
	mv testeBlockSize ./bin
	
matrixMul: 
	mkdir -p bin
	nvcc -o matrixMul matrixMul.cu rf-time.cc
	mv matrixMul ./bin
	
clean:
	rm -f testeBlockSizeTrans
	rm -f testeBlockSize
	rm -f matrixMul