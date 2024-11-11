/*
@(#)File:           mm-cudaMallocManaged.cu
@(#)Last changed:   2024/01/11 15:48:00
@(#)Purpose:        Sample Matrix Multiply using C and CUDA
@(#)Author:         Murilo Boratto - murilo.boratto 'at' fieb.org.br
@(#)Usage:         
 (*) Hotocompile:   nvcc mm-cudaMallocManaged.cu -o mm-cudaMallocManaged -Xcompiler -fopenmp -O3
 (*) Hotoexecute:  ./mm-cudaMallocManaged  <size> <blockSize>  
                   ./mm-cudaMallocManaged  10000      64
@(#)Comments:        
 (*) Using the concept of Unified Memory (cudaMallocManaged).
*/

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

__global__ void kernel(int *A, int *B, int *C, int size)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y; 

  if((i < size) && (j < size))
     for(int k = 0; k < size; k++)
        C[i * size + j] += A[i * size + k] * B[k * size + j];

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

int main(int argc, char **argv)
{
 /*Usage*/ 
 if (argc < 3) {
   printf("%s [SIZE] [BLOCKSIZE]\n", argv[0]);
   exit(-1);
 } 

 int n = atoi(argv[1]); 
 int blockSize = atoi(argv[2]); ; 
 double t1, t2;
 int *A,  *B,  *C;

 cudaMallocManaged(&A, sizeof(int) * n * n);
 cudaMallocManaged(&B, sizeof(int) * n * n);
 cudaMallocManaged(&C, sizeof(int) * n * n);

 initializeMatrix(A, n);
 initializeMatrix(B, n);

 //printMatrix(A, n);
 //printMatrix(B, n);
 //printMatrix(C, n);

t1 = omp_get_wtime();

 dim3 dimGrid( (int) ceil( (float) n / (float) blockSize ), (int) ceil( (float) n / (float) blockSize ) );
 dim3 dimBlock( blockSize, blockSize);

      kernel<<<dimGrid, dimBlock>>>(A, B, C, n);
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
