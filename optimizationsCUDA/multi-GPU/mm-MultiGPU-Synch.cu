/*
@(#)File:           mmMultiGPU.cu
@(#)Version:        CUDA version for matrix multiple on Multi-GPU using Synchronous Events
@(#)Last changed:   2024/10/30 10:00:00 
@(#)Purpose:        Matrix multiple on Multi-GPU
@(#)Author:         Murilo Boratto - murilo.boratto@fieb.org.br
@(#)Usage:         
 (*) Hotocompile:   nvcc mmMultiGPU.cu -o mmMultiGPU
 (*) Hotoexecute:   ./mmMultiGPU <size> <nGPUs>
 (*) Hotoexecute:   ./mmMultiGPU   16      2
@(#)Comment:  
    - Synchronous Events.

*/


#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void kernel(int *A, int *B, int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y; 
  
  if((i < n) && (j < n))
    for(int k = 0; k < n; k++)
       B[i * n + j] += A[i * n + k] * B[k * n + j];

}


void initializeMatrix(int *A, int n) {
  
  for(int i = 0; i < n; i++) 
    for(int j = 0; j < n; j++) 
      A[i * n + j]= rand()%(10-1)*1;
  
}

void printMatrix(int *A, int n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      printf("%1.2f\t", A[i * n + j]);
    }
    printf("\n");
  }
  printf("\n");
}

void mmMultiGPU(int *A, int *B, int n, int halfSize, int nGPUs){

  int *d_A[nGPUs], *d_B[nGPUs];
  
  for (int i = 0; i < nGPUs; i++) {
    cudaSetDevice(i);
    cudaMalloc(&d_A[i], halfSize * sizeof(int));
    cudaMalloc(&d_B[i], halfSize * sizeof(int));
  
    cudaMemcpy(d_A[i], &A[halfSize * i], halfSize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B[i], &B[halfSize * i], halfSize * sizeof(int), cudaMemcpyHostToDevice);
  }

  int threadsPerBlock = 256;
  int blocksPerGrid = (halfSize + threadsPerBlock - 1) / threadsPerBlock;

  for (int i = 0; i < nGPUs; i++) {
    cudaSetDevice(i);
    kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A[i], d_B[i], halfSize);
    cudaMemcpy(&B[halfSize * i], d_B[i], halfSize * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_A[i]);
    cudaFree(d_B[i]);
  }

}

int main(int argc, char *argv[]) {
 
  int n = atoi(argv[1]);
  int nGPUs = atoi(argv[2]);
  
  int halfSize = (n * n) / nGPUs;

  int *A = (int *)malloc(n * n * sizeof(int));
  int *B = (int *)malloc(n * n * sizeof(int));

  initializeMatrix(A, n);
  initializeMatrix(B, n);

  printMatrix(A, n);
  printMatrix(B, n);

     mmMultiGPU(A, B, n, halfSize, nGPUs);
     
  printMatrix(B, n);

  free(A);
  free(B);

  return 0;
}
