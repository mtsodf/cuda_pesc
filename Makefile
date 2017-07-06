all: testeBlockSizeTrans testeBlockSize matrixMul

testeBlockSizeTrans: 
	nvcc -o testeBlockSizeTrans testeBlockSizeTrans.cu

testeBlockSize: 
	nvcc -o testeBlockSize testeBlockSize.cu

matrixMul: 
	nvcc -o matrixMul matrixMul.cu