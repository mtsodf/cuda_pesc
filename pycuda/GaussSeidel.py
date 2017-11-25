import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from scipy.sparse import csr_matrix, identity
from time import time

mod = SourceModule("""
    __global__ void doublify(double *a)
    {
        int idx = threadIdx.x + threadIdx.y*4;
        a[idx] *= 2;
    }

    __global__ void GaussSeidelRb(int *ia, int *ja, double *a, int n){
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
    }

    __global__ void MatVec(int *ia, int *ja, double *a, double *b, double *c, int n){
        int idx = blockDim.x * blockIdx.x + threadIdx.x;



        if(idx < n){
            double aux = 0.0;
            for(int i=ia[idx]; i < ia[idx+1]; i++){
                aux += a[i]*b[ja[i]];
            }
            c[idx] = aux;
        }

    }

    __global__ void MatVecSp(int *ia, int *ja, float *a, float *b, float *c, int n){
        int idx = blockDim.x * blockIdx.x + threadIdx.x;



        if(idx < n){
            double aux = 0.0;
            for(int i=ia[idx]; i < ia[idx+1]; i++){
                aux += a[i]*b[ja[i]];
            }
            c[idx] = aux;
        }

    }

  """
  )


def MatVecSp(A, b):

    n = np.int32(b.size)


    b_sp = b.astype(np.float32)
    data_sp = A.data.astype(np.float32)

    # Alocando memoria na gpu
    ia_gpu = cuda.mem_alloc(A.indptr.nbytes)
    ja_gpu = cuda.mem_alloc(A.indices.nbytes)
    data_gpu = cuda.mem_alloc(data_sp.nbytes)
    b_gpu = cuda.mem_alloc(b_sp.nbytes)
    c_gpu = cuda.mem_alloc(b_sp.nbytes)


    cuda.memcpy_htod(ia_gpu, A.indptr)
    cuda.memcpy_htod(ja_gpu, A.indices)
    cuda.memcpy_htod(data_gpu, data_sp)
    cuda.memcpy_htod(b_gpu, b_sp)

    blocksize = 64

    gridsize = n/blocksize
    gridsize += 0 if n%blocksize == 0 else 1


    func = mod.get_function("MatVecSp")
    grid = (gridsize, 1, 1)
    block = (blocksize, 1, 1)
    t1 = time()
    func(ia_gpu, ja_gpu, data_gpu, b_gpu, c_gpu, n,
        grid=grid, block=block)
    pycuda.driver.Context.synchronize()
    t2 = time()


    c = np.empty_like(b_sp)

    cuda.memcpy_dtoh(c, c_gpu)

    return c, t2-t1


def MatVec(A, b):

    n = np.int32(b.size)


    # Alocando memoria na gpu
    ia_gpu = cuda.mem_alloc(A.indptr.nbytes)
    ja_gpu = cuda.mem_alloc(A.indices.nbytes)
    data_gpu = cuda.mem_alloc(A.data.nbytes)
    b_gpu = cuda.mem_alloc(b.nbytes)
    c_gpu = cuda.mem_alloc(b.nbytes)

    cuda.memcpy_htod(ia_gpu, A.indptr)
    cuda.memcpy_htod(ja_gpu, A.indices)
    cuda.memcpy_htod(data_gpu, A.data)
    cuda.memcpy_htod(b_gpu, b)


    blocksize = 64

    gridsize = n/blocksize
    gridsize += 0 if n%blocksize==0 else 1


    func = mod.get_function("MatVec")
    grid = (gridsize, 1, 1)
    block = (blocksize, 1, 1)

    t1 = time()
    func(ia_gpu, ja_gpu, data_gpu, b_gpu, c_gpu, n,
        grid=grid, block=block)
    pycuda.driver.Context.synchronize()
    t2 = time()

    c = np.empty_like(b)
    cuda.memcpy_dtoh(c, c_gpu)
    return c, t2-t1

n = np.int32(10000000)

A = identity(n, format='csr')

b = np.ones(n)

t1 = time()
c1, dt1 = MatVec(A, b)
t2 = time()
d = A*b
t3 = time()
c2, dt2 = MatVecSp(A,b)
t4 = time()

print "Norma da Diferenca = ", np.linalg.norm(c1-d)
print "Norma da Diferenca = ", np.linalg.norm(c2-d)


print "Tempo Numpy SP = ", t4-t3, dt2

print "Tempo Gpu DP = ", t2-t1, dt1

print "Tempo Numpy  = ", t3-t2



