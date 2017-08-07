from __future__ import division
from pycuda.compiler import SourceModule
import pycuda.driver as cuda
import pycuda.autoinit
import math
import numpy as np

mod = SourceModule("""
#include <stdio.h>

__global__ void transpose(float *odata, const float **idata,
    const int width, const int height)
{
    __shared__ float tile[32][33];
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;

//    printf("pointer is: %p\\n", odata
    if (x < width && y < height) {
//         printf("1\\n");
//        printf("INDEX: %d,%d, VALUE: \\n", ,x);
//        printf("INDEX: x:%d,y:%d, VALUE: \\n", x,y);
        tile[threadIdx.y][threadIdx.x] = idata[y][x];
    }
    __syncthreads();
    x = blockIdx.y * 32 + threadIdx.x; // transpose block offset
    y = blockIdx.x * 32 + threadIdx.y;
    if (y < width && x < height) {
        odata[y*height + x] = tile[threadIdx.x][threadIdx.y];
    }
}

__global__ void kernel_test(float **ptrs, float *value){
 
  printf(" value d1 pointer = %p\\n", ptrs[0]);
  printf(" value d2 pointer = %p\\n", ptrs[1]);
  printf(" value d1[0] pointer = %f\\n", ptrs[0][0]);
  printf(" value d1[0] pointer = %f\\n", ptrs[0][1]);
  printf(" value d2[1] pointer = %f\\n", ptrs[1][0]);
  printf(" value d2[1] pointer = %f\\n", ptrs[1][1]);

//  values[0] = ptrs[0];
//  values[1] = ptrs[1];
//  values[2] = ptrs[2];
//  values[3] = ptrs[3];
//  values[4] = ptrs[4];
//  values[5] = ptrs[5];
//  values[6] = ptrs[6];
//  values[7] = ptrs[7];

}


""")

#test = mod.get_function("kernel_test")

#d1 = np.array([1.1, 1.2]).astype(np.float32)
#d2 = np.array([2.1, 2.2]).astype(np.float32)

#d1_gpu = cuda.mem_alloc(d1.nbytes)
#d2_gpu = cuda.mem_alloc(d2.nbytes)

#cuda.memcpy_htod(d1_gpu, d1)
#cuda.memcpy_htod(d2_gpu, d2)

#d1_gpu_1 = np.uint64(int(d1_gpu))
#d2_gpu_1 = np.uint64(int(d2_gpu))

#ptrs = np.array([d1_gpu_1, d2_gpu_1]).astype(np.uint64)

#print(ptrs)


#ptrs_gpu = cuda.mem_alloc(ptrs.nbytes)
#cuda.memcpy_htod(ptrs_gpu, ptrs)

#values = np.zeros((2,2)).astype(np.float32)
#values_d = cuda.mem_alloc(values.nbytes)


#block = (1,1,1)
#grid = (1,1)
#test(ptrs_gpu, np.float32(1), block=block, grid=grid)

#cuda.memcpy_dtoh(values, d1_gpu)
#print(values)


#########################################################################################

transpose = mod.get_function("transpose")

dim = (5,)
a = []
c = []
for i in range(10):
    b = np.random.random(dim).astype(np.float32)
    b_gpu = cuda.mem_alloc(b.nbytes)
    cuda.memcpy_htod(b_gpu, b)
    a.append(b_gpu)
    c.append(b)

a_fixed = []
for i in a:
    a_fixed.append(np.uint64(int(i)))

ptrs = np.array(a_fixed).astype(np.uint64)
print(ptrs)
print(a_fixed)
c1 = np.array(c)

ptrs_gpu = cuda.mem_alloc(ptrs.nbytes)
cuda.memcpy_htod(ptrs_gpu, ptrs)

d = np.zeros((c1.shape[1], c1.shape[0])).astype(np.float32)
d_gpu = cuda.mem_alloc(d.nbytes)


block_x, block_y = 32, 32
block = (block_x, block_y, 1)
grid = (int(math.ceil(c1.shape[1] / block_x)), int(math.ceil(c1.shape[0]/block_y)))
print(block)
print(grid)

#block=(32,3,1)
#grid = (1,1)

transpose(d_gpu, ptrs_gpu, np.int32(c1.shape[1]), np.int32(c1.shape[0]), block=block, grid=grid)

#ptrs1 = np.zeros(c1.shape).astype(np.float32)
#cuda.memcpy_dtoh(ptrs1, d_gpu)
#print(ptrs1.shape)
#print(c1.shape)

cuda.memcpy_dtoh(d, d_gpu)

print(d)
print(c1.T)

print(((d-c1.T) > 0.001).sum())

#print(ptrs.shape)
#print(c.shape)
#print(d.shape)

