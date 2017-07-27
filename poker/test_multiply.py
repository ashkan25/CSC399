#from __future__ import division
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import math
import numpy as np

mod = SourceModule("""
#define TILE_DIM 32
#include <stdio.h>

__global__ void MatMul(float* A, float* B, float* C, int ARows, int ACols, int BRows,
    int BCols, int CRows, int CCols)
{
    float CValue = 0;

    int Row = blockIdx.y*TILE_DIM + threadIdx.y;
    int Col = blockIdx.x*TILE_DIM + threadIdx.x;

    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    for (int k = 0; k < (TILE_DIM + ACols - 1)/TILE_DIM; k++) {

         if (k*TILE_DIM + threadIdx.x < ACols && Row < ARows)
             As[threadIdx.y][threadIdx.x] = A[Row*ACols + k*TILE_DIM + threadIdx.x];
         else
             As[threadIdx.y][threadIdx.x] = 0.0;

         if (k*TILE_DIM + threadIdx.y < BRows && Col < BCols)
             Bs[threadIdx.y][threadIdx.x] = B[(k*TILE_DIM + threadIdx.y)*BCols + Col];
         else
             Bs[threadIdx.y][threadIdx.x] = 0.0;

         __syncthreads();

         for (int n = 0; n < TILE_DIM; ++n)
             CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];

         __syncthreads();
    }

    if (Row < CRows && Col < CCols)
        C[((blockIdx.y * blockDim.y + threadIdx.y)*CCols) +
           (blockIdx.x * blockDim.x)+ threadIdx.x] = CValue;
}

    __global__ void forward_pass(const float* a, const float* b, float* c,
                                const int a_width, const int b_width, const int c_width, const int a_height) {

        float c_value = 0.0;

        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        // TODO pass in Height of matrix A. It is a vector, so it always should be 1
        if(row >= a_height || col >= b_width) {
            return;
        }

        for (int i = 0; i < a_width; i++) {
            c_value += (a[row * a_width + i]) * (b[i * b_width + col]);
        }

        c[row*c_width+col] = c_value;

        // TODO It might be better to export ReLU to another kernel since it is not needed for the final hidden layer
        // ReLU function: max(x, 0)
        //c[row * c_width + col] = c_value > 0 ? c_value  : 0;

    }


""")

forward_pass = mod.get_function("MatMul")
#forward_pass = mod.get_function("forward_pass")

import time

time_a = time.time()

a = np.random.random((7643,1)).astype(np.float32)
b = np.random.random((845,727)).astype(np.float32)
c = np.zeros((a.shape[0], b.shape[1])).astype(np.float32)

a_gpu = cuda.mem_alloc(a.nbytes)
b_gpu = cuda.mem_alloc(b.nbytes)
c_gpu = cuda.mem_alloc(c.nbytes)


cuda.memcpy_htod(a_gpu, a)
cuda.memcpy_htod(b_gpu, b)
cuda.memcpy_htod(c_gpu, c)

block_x, block_y = 16,16
block = (block_x, block_y, 1)
grid = ((b.shape[1] + block_x - 1) / block_x, (a.shape[0] + block_y - 1) / block_y)

print(grid)

forward_pass(a_gpu, b_gpu, c_gpu, np.int32(a.shape[0]), np.int32(a.shape[1]), np.int32(b.shape[0]),
             np.int32(b.shape[1]), np.int32(c.shape[0]), np.int32(c.shape[1]), block=block, grid=grid)

#forward_pass(a_gpu, b_gpu, c_gpu, np.int32(a.shape[1]), np.int32(b.shape[1]), np.int32(c.shape[1]), np.int32(a.shape[0]), block=block, grid=grid)

cuda.memcpy_dtoh(c, c_gpu)

print((abs(c-np.dot(a,b)) > 0.1).sum())

time_b = time.time()
print(time_b - time_a)
print(np.dot(a,b)[0][:10])
print(c[0][:10])
