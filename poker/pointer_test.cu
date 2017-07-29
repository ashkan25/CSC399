#include <stdio.h>
#define DSIZE 256
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)
__global__ void kernel_test(float **ptrs, float *values){
 
  printf(" value d1[0] = %f\n", ptrs[0][0]);
  printf(" value d1[1] = %f\n", ptrs[0][1]);
  printf(" value d2[0] = %f\n", ptrs[1][0]);
  printf(" value d2[1] = %f\n", ptrs[1][1]);

  values[0] = ptrs[0][0];
  values[1] = ptrs[0][1];
  values[2] = ptrs[1][0];
  values[3] = ptrs[1][1];

}
 
int main(){
  float *data_1, *data_2, *data_3, *data_4;
  data_1 = (float *)malloc(DSIZE);
  data_2 = (float *)malloc(DSIZE);
  data_3 = (float *)malloc(DSIZE);
  data_4 = (float *)malloc(DSIZE);
 
  float *data_d1,*data_d2,*data_d3,*data_d4;
 
  cudaMalloc((void **)&data_d1,DSIZE);
  cudaMalloc((void **)&data_d2,DSIZE);
  cudaMalloc((void **)&data_d3,DSIZE);
  cudaMalloc((void **)&data_d4,DSIZE);
  data_1[0] = 54321.0f;
  data_1[1] = 4321.0f;
  data_2[0] = 12345.0f;
  data_2[1] = 1234.0f;
  cudaMemcpy(data_d1,data_1,DSIZE,cudaMemcpyHostToDevice);
  cudaMemcpy(data_d2,data_2,DSIZE,cudaMemcpyHostToDevice);
  cudaMemcpy(data_d3,data_3,DSIZE,cudaMemcpyHostToDevice);
  cudaMemcpy(data_d4,data_4,DSIZE,cudaMemcpyHostToDevice);
 
  float *ptrs[4];
 
  ptrs[0] = data_d1;
  ptrs[1] = data_d2;
  ptrs[2] = data_d3;
  ptrs[3] = data_d4;

  float *values_d;
  cudaMalloc((void **) &values_d, sizeof(float)*4); 

 
  float **ptrs_d;
  size_t size = 4 * sizeof(float*);
  cudaMalloc((void ***)&ptrs_d,size);
  cudaMemcpy(ptrs_d,ptrs,size,cudaMemcpyHostToDevice);
  kernel_test<<<1,1>>>(ptrs_d, values_d);
  cudaDeviceSynchronize();
  cudaCheckErrors("some error");

  float *values;
  values = (float *) malloc(sizeof(float)*4);
  cudaMemcpy(values, values_d, sizeof(float)*4, cudaMemcpyDeviceToHost);
  for(int i = 0; i < 4; i++)
    printf("VALUE AT %d is %f\n", i, values[i]);

  return 0;
}
