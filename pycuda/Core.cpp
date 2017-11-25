    __global__ void doublify(double *a)
    {
        int idx = threadIdx.x + threadIdx.y*4;
        a[idx] *= 2;
    }

    __global__ void GaussSeidelRb(int *ia, int *ja, double *a, double *b, double *x, int rOub, int n){
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        idx = 2*idx + rOub%2;
        if( idx < n){

            double aux, diag;

            aux = b[idx];

            for (int i = ia[idx]; i < ia[idx+1]; ++i)
            {
                if(ja[i] == idx){
                    diag = a[i];
                }
                else{
                    aux -= a[i];
                }
            }

            x[idx] = aux/diag;

        }
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