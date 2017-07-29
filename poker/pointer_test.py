from __future__ import division
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import math
import numpy as np

mod = SourceModule("""
#include <stdio.h>

__global__ void transpose(float *odata, const float *idata,
    const int width, const int height)
{
    __shared__ float tile[32][33];
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    if (x < width && y < height) {
        printf("INDEX: %f\\n", idata[y*width + x]);
        tile[threadIdx.y][threadIdx.x] = idata[y*width + x];
    }
    __syncthreads();
    x = blockIdx.y * 32 + threadIdx.x; // transpose block offset
    y = blockIdx.x * 32 + threadIdx.y;
    if (y < width && x < height) {
        odata[y*height + x] = tile[threadIdx.x][threadIdx.y];
    }
}

__global__ void kernel_test(float *ptrs, float *values){
 
  printf(" value d1[0] = %f\\n", ptrs[0]);
  printf(" value d1[1] = %f\\n", ptrs[1]);
  printf(" value d2[0] = %f\\n", ptrs[2]);
  printf(" value d2[1] = %f\\n", ptrs[3]);
  printf(" value d3[0] = %f\\n", ptrs[4]);
  printf(" value d3[1] = %f\\n", ptrs[5]);
  printf(" value d4[0] = %f\\n", ptrs[6]);
  printf(" value d4[1] = %f\\n", ptrs[7]);

  values[0] = ptrs[0];
  values[1] = ptrs[1];
  values[2] = ptrs[2];
  values[3] = ptrs[3];
  values[4] = ptrs[4];
  values[5] = ptrs[5];
  values[6] = ptrs[6];
  values[7] = ptrs[7];

}


""")

#test = mod.get_function("kernel_test")

#d1 = np.array([1.1, 1.2]).astype(np.float32)
#d2 = np.array([2.1, 2.2]).astype(np.float32)
#d3 = np.array([3.1, 3.2]).astype(np.float32)
#d4 = np.array([4.1, 4.2]).astype(np.float32)

#d1_gpu = cuda.mem_alloc(d1.nbytes)
#d2_gpu = cuda.mem_alloc(d2.nbytes)
#d3_gpu = cuda.mem_alloc(d3.nbytes)
#d4_gpu = cuda.mem_alloc(d4.nbytes)

#cuda.memcpy_htod(d1_gpu, d1)
#cuda.memcpy_htod(d2_gpu, d2)
#cuda.memcpy_htod(d3_gpu, d3)
#cuda.memcpy_htod(d4_gpu, d4)

#ptrs = np.array([d1, d2, d3, d4])

#ptrs_gpu = cuda.mem_alloc(ptrs.nbytes)
#cuda.memcpy_htod(ptrs_gpu, ptrs)

#values = np.zeros((4,2)).astype(np.float32)
#values_d = cuda.mem_alloc(values.nbytes)


#block = (1,1,1)
#grid = (1,1)
#test(ptrs_gpu, values_d, block=block, grid=grid)


#cuda.memcpy_dtoh(values, values_d)
#print(values)


transpose = mod.get_function("transpose")

dim = (5,1)
a = []
c = []
for i in range(10):
    b = np.random.random(dim).astype(np.float32)
    b_gpu = cuda.mem_alloc(b.nbytes)
    cuda.memcpy_htod(b_gpu, b)
    a.append(b_gpu)
    c.append(b)

ptrs = np.array(a)
c = np.array(c)

ptrs_gpu = cuda.mem_alloc(ptrs.nbytes)
cuda.memcpy_htod(ptrs_gpu, ptrs)

d = np.zeros((c.shape[1], c.shape[0])).astype(np.float32)
d_gpu = cuda.mem_alloc(d.nbytes)


block_x, block_y = 32, 32
block = (block_x, block_y, 1)
grid = (int(math.ceil(c.shape[1] / block_x)), int(math.ceil(c.shape[0]/block_y)))
print(block)
print(grid)
transpose(d_gpu, ptrs_gpu, np.int32(c.shape[1]), np.int32(c.shape[0]), block=block, grid=grid)

cuda.memcpy_dtoh(d, d_gpu)

print(d)
print(c.T)

print(ptrs.shape)
print(c.shape)
print(d.shape)
