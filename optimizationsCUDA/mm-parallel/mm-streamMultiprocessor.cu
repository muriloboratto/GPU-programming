/*
@(#)File:           mm-streamMultiprocessador.cu
@(#)Last changed:   2024/01/11 15:48:00
@(#)Purpose:        Sample Matrix Multiply using C and CUDA
@(#)Author:         Murilo Boratto - murilo.boratto@fieb.org.br
@(#)Usage:         
 (*) Hotocompile:   nvcc mm-dimensionSM.cu -o mm-dimensionSM -Xcompiler -fopenmp -O3
 (*) Hotoexecute:  ./mm-dimensionSM  <size> <blockSize>  
                   ./mm-dimensionSM   10000      64
@(#)Comments:        
 (*) Dimension SMs.
*/

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

__global__ void kernelGridStriderLoop(int *A, int *B,  int *C, int n)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  int stride = gridDim.x * blockDim.x;
  int i, j, k;

  for(i = idx; i < n; i += stride)
    for(j = idy; j < n; j += stride)
    {
       for(k = 0; k < n; k++) 
         C[i*n+j] += A[i*n+k] * B[k*n+j];
    }  

}

void initializeMatrix(int *A, int size)
{
  for(int i = 0; i < size; i++)
    for(int j = 0; j < size; j++)
      A[i * size + j] = rand() % (10 - 1) * 1;
}

void printMatrix(int *A, int size)
{
  for(int i = 0; i < size; i++){
    for(int j = 0; j < size; j++)
      printf("%d\t", A[i * size + j]);
    printf("\n");
  }
  printf("\n");
}

int main (int argc, char **argv)
{
 /*Usage*/ 
 if (argc < 3) {
   printf("%s [SIZE] [BLOCKSIZE]\n", argv[0]);
   exit(-1);
 } 

 int n = atoi(argv[1]); 
 int sizeblock = atoi(argv[2]); ; 
 double t1, t2;
 int *A,  *B, *C;

 cudaMallocManaged(&A, sizeof(int) * n * n);
 cudaMallocManaged(&B, sizeof(int) * n * n);
 cudaMallocManaged(&C, sizeof(int) * n * n);

 initializeMatrix(A, n);
 initializeMatrix(B, n);

 //printMatrix(A, n);
 //printMatrix(B, n);
 //printMatrix(C, n);

t1 = omp_get_wtime();

 int deviceId, numberOfSMs;
 cudaGetDevice(&deviceId);
 cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

 int NUMBER_OF_BLOCKS = numberOfSMs * 32;
 int NUMBER_OF_THREADS = 1024; 

      kernelGridStriderLoop<<< NUMBER_OF_BLOCKS, NUMBER_OF_THREADS>>>(A, B, C, n);
      cudaDeviceSynchronize();
 
 t2 = omp_get_wtime();

 printf("%d\t%f\n", n, t2-t1);

//printMatrix(A, n);
//printMatrix(B, n);
//printMatrix(C, n);

cudaFree(A);
cudaFree(B);
cudaFree(C);

return 0;
}
