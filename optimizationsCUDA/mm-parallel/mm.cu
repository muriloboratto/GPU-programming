/*
@(#)File:           mm.cu
@(#)Last changed:   2024/01/15 09:27:00
@(#)Purpose:        Sample Matrix Multiply using C and CUDA
@(#)Author:         Murilo Boratto - murilo.boratto 'at' fieb.org.br
@(#)Usage:         
 (*) Hotocompile:   nvcc mm.cu -o mm -Xcompiler -fopenmp -O3
 (*) Hotoexecute:  ./mm  <size> <blockSize>  
                   ./mm   10000     64
@(#)Comments:        
 (*)  The workload is divided in threads manually.
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
  if (argc < 3) 
  {
    printf("%s [SIZE] [BLOCKSIZE]\n", argv[0]);
    exit(-1);
  } 

  int n = atoi(argv[1]); 
  int blockSize = atoi(argv[2]); 
  double t1, t2;
  
 //Memory Allocation in the Host
  int  *A = (int *) malloc (sizeof(int)*n*n);
  int  *B = (int *) malloc (sizeof(int)*n*n);
  int  *C = (int *) malloc (sizeof(int)*n*n);

  initializeMatrix(A, n);
  initializeMatrix(B, n);

 //printMatrix(A, n);
 //printMatrix(B, n);
 //printMatrix(C, n);

 // Memory Allocation in the Device 
  int *d_A, *d_B, *d_C;
  cudaMalloc((void **) &d_A, n * n * sizeof(int) ) ;
  cudaMalloc((void **) &d_B, n * n * sizeof(int) ) ;
  cudaMalloc((void **) &d_C, n * n * sizeof(int) ) ;

  t1 = omp_get_wtime();

 // Copy of data from host to device
  cudaMemcpy( d_A, A, n * n * sizeof(int), cudaMemcpyHostToDevice ) ;
  cudaMemcpy( d_B, B, n * n * sizeof(int), cudaMemcpyHostToDevice ) ;
  cudaMemcpy( d_C, C, n * n * sizeof(int), cudaMemcpyHostToDevice ) ;

 // 2D Computational Grid
  dim3 dimGrid( (int) ceil( (float) n / (float) blockSize ), (int) ceil( (float) n / (float) blockSize ) );
  dim3 dimBlock( blockSize, blockSize);

        kernel<<<dimGrid, dimBlock>>>(A, B, C, n);

 // Copy of data from device to host
  cudaMemcpy( C, d_C, n * n * sizeof(float), cudaMemcpyDeviceToHost ) ;
 
  t2 = omp_get_wtime();

  printf("%d\t%f\n", n, t2-t1);

 //printMatrix(A, n);
 //printMatrix(B, n);
 //printMatrix(C, n);

// Memory Allocation in the Device
 cudaFree(d_A) ;
 cudaFree(d_B) ;
 cudaFree(d_C) ;

// Memory Allocation in the Host
 free(A); 
 free(B); 
 free(C); 

 return 0;
}
