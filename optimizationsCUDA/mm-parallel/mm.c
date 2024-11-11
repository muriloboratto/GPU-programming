#include <stdio.h>
#include <stdlib.h>

void initializeMatrix(int *matrix, int size)
{
  for (int i = 0; i < size; i++)
    for (int j = 0; j < size; j++)
      matrix[i * size + j] = rand() % (10 - 1) * 1;
}

void printMatrix(int *matrix, int size)
{
  for (int i = 0; i < size; i++)
  {
    for (int j = 0; j < size; j++)
      printf("%d\t", matrix[i * size + j]);
    printf("\n");
  }
  printf("\n");
}

int main (int argc, char **argv)
{

 int size = atoi(argv[1]);  
 int i, j, k;
 double t1, t2;

 int  *A = (int *) malloc (sizeof(int)*size*size);
 int  *B = (int *) malloc (sizeof(int)*size*size);
 int  *C = (int *) malloc (sizeof(int)*size*size);

 initializeMatrix(A, size);
 initializeMatrix(B, size);

 t1 = omp_get_wtime();

 
   for(i = 0; i < size; i++)
    for(j = 0; j < size; j++)
      for(k = 0; k < size; k++)
        C[i * size + j] += A[i * size + k] * B[k * size + j];

 t2 = omp_get_wtime();

 printf("%d\t%f\n",size, t2-t1);

// printMatrix(A,size);
// printMatrix(B,size);
// printMatrix(C,size);

 return 0;

}

