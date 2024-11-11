/*
@(#)File:           addMatrixMultiGPU.cu
@(#)Version:        CUDA version for add matrix on Multi-GPU
@(#)Last changed:   2023/08/02 10:00:00 
@(#)Purpose:        Add Matrix on Multi-GPU
@(#)Author:         Murilo Boratto - murilo.boratto@fieb.org.br
@(#)Usage:         
 (*) Hotocompile:   nvcc addMatrixMultiGPU.cu -o add
 (*) Hotoexecute:   ./add <size> <nGPUs>
 (*) Hotoexecute:   ./add   16      2
@(#)Comment:  
    - Synchronous Events.

*/


#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void kernel(float *A, float *B, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    B[i] = A[i] + B[i];
  }
}

void initializeMatrix(float *A, int n) {
  
  for(int i = 0; i < n; i++) 
    for(int j = 0; j < n; j++) 
      A[i * n + j]= rand()%(10-1)*1;
  
}

void printMatrix(float *A, int n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      printf("%1.2f\t", A[i * n + j]);
    }
    printf("\n");
  }
  printf("\n");
}

void addMatrixMultiGPU(float *A, float *B, int n, int halfSize, int nGPUs){

  float *d_A[nGPUs], *d_B[nGPUs];
  
  for (int i = 0; i < nGPUs; i++) {
    cudaSetDevice(i);
    cudaMalloc(&d_A[i], halfSize * sizeof(float));
    cudaMalloc(&d_B[i], halfSize * sizeof(float));
  
    cudaMemcpy(d_A[i], &A[halfSize * i], halfSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B[i], &B[halfSize * i], halfSize * sizeof(float), cudaMemcpyHostToDevice);
  }

  int threadsPerBlock = 256;
  int blocksPerGrid = (halfSize + threadsPerBlock - 1) / threadsPerBlock;

  for (int i = 0; i < nGPUs; i++) {
    cudaSetDevice(i);
    kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A[i], d_B[i], halfSize);
    cudaMemcpy(&B[halfSize * i], d_B[i], halfSize * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A[i]);
    cudaFree(d_B[i]);
  }

}

int main(int argc, char *argv[]) {
 
  int n = atoi(argv[1]);
  int nGPUs = atoi(argv[2]);
  
  int halfSize = (n * n) / nGPUs;

  float *A = (float *)malloc(n * n * sizeof(float));
  float *B = (float *)malloc(n * n * sizeof(float));

  initializeMatrix(A, n);
  initializeMatrix(B, n);

  printMatrix(A, n);
  printMatrix(B, n);

     addMatrixMultiGPU(A, B, n, halfSize, nGPUs);
     
  printMatrix(B, n);

  free(A);
  free(B);

  return 0;
}
