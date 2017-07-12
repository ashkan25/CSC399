#include <stdio.h>
#include <stdlib.h>


// TILE_DIM and BLOCK_ROWS MUST BE SET TO THE 
#define TILE_DIM    2
#define BLOCK_ROWS  2

__global__ void transposeNaive(float *odata, float* idata, int width, int height)
{
  int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
  int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;

  int index_in  = xIndex + width * yIndex;
  int index_out = yIndex + height * xIndex;
    for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) {
      printf("%d, %d\n", index_out+i, index_in+i*width);
      odata[index_out+i] = idata[index_in+i*width];
    }
}

__global__ void transposeFineGrained(float *odata, float *idata, int width, int height)
{
    __shared__ float block[TILE_DIM][TILE_DIM+1];

    int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
    int index = xIndex + (yIndex)*width;

    for (int i=0; i < TILE_DIM; i += BLOCK_ROWS) {
      block[threadIdx.y+i][threadIdx.x] = idata[index+i*width];
    }

    __syncthreads();

    for (int i=0; i < TILE_DIM; i += BLOCK_ROWS) {
      odata[index+i*height] = block[threadIdx.x][threadIdx.y+i];
    }
}


__global__ void transposeDiagonal(float *odata, float *idata, int width, int height)
{
  __shared__ float tile[TILE_DIM*(TILE_DIM+1)];

  int blockIdx_x, blockIdx_y;

  // do diagonal reordering
  if (width == height) {
    blockIdx_y = blockIdx.x;
    blockIdx_x = (blockIdx.x+blockIdx.y)%gridDim.x;
  } else {
    int bid = blockIdx.x + gridDim.x*blockIdx.y;
    blockIdx_y = bid%gridDim.y;
    blockIdx_x = ((bid/gridDim.y)+blockIdx_y)%gridDim.x;
  }

  // from here on the code is same as previous kernel except blockIdx_x replaces blockIdx.x
  // and similarly for y

  int xIndex = blockIdx_x * TILE_DIM + threadIdx.x;
  int yIndex = blockIdx_y * TILE_DIM + threadIdx.y;
  int index_in = xIndex + (yIndex)*width;

  xIndex = blockIdx_y * TILE_DIM + threadIdx.x;
  yIndex = blockIdx_x * TILE_DIM + threadIdx.y;
  int index_out = xIndex + (yIndex)*height;

    for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) {
      tile[(threadIdx.y+i)*TILE_DIM + threadIdx.x] = idata[index_in+i*width];
    }

    __syncthreads();

    for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) {
      odata[index_out+i*height] = tile[threadIdx.x*TILE_DIM + (threadIdx.y+i)];
      //printf("%f", idata[i]);
    }
}


__global__ void transposeDiagonal2(float *odata, float *idata, int width, int height)
{
  extern __shared__ float tile[];//[];

  //int n = sizeof(tile);
  //printf("SIZE is: %d \n", n);

  int blockIdx_x, blockIdx_y;

  // do diagonal reordering
  if (width == height) {
    blockIdx_y = blockIdx.x;
    blockIdx_x = (blockIdx.x+blockIdx.y)%gridDim.x;
  } else {
    int bid = blockIdx.x + gridDim.x*blockIdx.y;
    blockIdx_y = bid%gridDim.y;
    blockIdx_x = ((bid/gridDim.y)+blockIdx_y)%gridDim.x;
  }

  // from here on the code is same as previous kernel except blockIdx_x replaces blockIdx.x
  // and similarly for y

  int xIndex = blockIdx_x * TILE_DIM + threadIdx.x;
  int yIndex = blockIdx_y * TILE_DIM + threadIdx.y;
  int index_in = xIndex + (yIndex)*width;

  xIndex = blockIdx_y * TILE_DIM + threadIdx.x;
  yIndex = blockIdx_x * TILE_DIM + threadIdx.y;
  int index_out = xIndex + (yIndex)*height;

    for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) {
      tile[(threadIdx.y+i)*TILE_DIM + threadIdx.x] = idata[index_in+i*width];
    }

    __syncthreads();

    for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) {
      odata[index_out+i*height] = tile[threadIdx.x*TILE_DIM + (threadIdx.y+i)];
      //printf("%f", idata[i]);
    }
}


int main( void ) {
    int N = 17;
    int M = TILE_DIM;
    float a[N*M], b[M*N];
    float *dev_a, *dev_b;
    cudaMalloc((void**)&dev_a, M*N * sizeof(float));
    cudaMalloc((void**)&dev_b, M*N * sizeof(float));

    for (int i=0; i<N; i++) {
        for (int j = 0; j<M; j++) {
        a[i*M + j] = i*M + j;
        //printf("%d, %d\n", i, j);
        }
    }

    cudaMemcpy( dev_a, a, N * M * sizeof(float), cudaMemcpyHostToDevice);
    dim3 grid(9,1), threads(TILE_DIM,BLOCK_ROWS);
    transposeDiagonal2<<<grid,threads, sizeof(float)*2*3>>>(dev_b, dev_a, N, M);

    cudaMemcpy(b, dev_b, N * M * sizeof(float), cudaMemcpyDeviceToHost);

        
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            printf("%d ", int(b[i*M + j]));
        }
        printf("\n");
    }

  //  for(int i =0; i < 10; i++) {
  //      printf("%d \n", int(b[i]));
  //  }

    cudaFree(dev_a);
    cudaFree(dev_b);
    return 0;

}
